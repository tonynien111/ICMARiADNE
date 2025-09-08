import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from icm import ICMAgent

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, icm_agent=None):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)
        
        # ICM integration
        self.icm_agent = icm_agent
        self.use_icm = USE_ICM and icm_agent is not None

        self.episode_buffer = []
        self.perf_metrics = dict()
        # Add buffer for intrinsic rewards (index 15)
        buffer_size = 16 if self.use_icm else 15
        for i in range(buffer_size):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):

            self.save_observation(observation)

            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
                
            intrinsic_reward = 0.0
            next_observation = self.robot.get_observation()
            
            # Update intrinsic reward calculation to use the correct next_observation
            if self.use_icm and i > 0:
                try:
                    prev_obs_index = len(self.episode_buffer[0]) - 1
                    if prev_obs_index >= 0:
                        prev_observation = self.get_observation_from_buffer(prev_obs_index)
                        
                        intrinsic_reward_tensor = self.icm_agent.compute_intrinsic_reward(
                            prev_observation, next_observation, action_index, ICM_ACTION_DIM
                        )
                        
                        if ICM_ADAPTIVE:
                            eta_adaptive = self.compute_adaptive_icm_weight()
                        else:
                            eta_adaptive = ICM_ETA
                            
                        intrinsic_reward = float(intrinsic_reward_tensor.item()) * eta_adaptive
                        reward += intrinsic_reward
                except Exception as e:
                    print(f"ICM computation failed: {e}")
                    # intrinsic_reward remains 0.0

            self.save_reward_done(reward, done, intrinsic_reward)
            self.save_next_observations(next_observation)
            observation = next_observation

            if self.save_image:
                self.robot.plot_env()
                self.env.plot_env(i+1)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done, intrinsic_reward=0.0):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)
        
        # Save intrinsic reward if ICM is used
        if self.use_icm:
            self.episode_buffer[15] += torch.FloatTensor([intrinsic_reward]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()
    
    def compute_adaptive_icm_weight(self):
        """Compute adaptive ICM weight based on exploration progress"""
        current_explored_rate = self.env.explored_rate
        
        # Strategy 1: Linear decay from max to min as exploration progresses
        if current_explored_rate >= ICM_EXPLORATION_THRESHOLD:
            # In well-explored areas, reduce curiosity significantly
            progress = (current_explored_rate - ICM_EXPLORATION_THRESHOLD) / (1.0 - ICM_EXPLORATION_THRESHOLD)
            eta_adaptive = ICM_ETA_MAX * (1 - progress) + ICM_ETA_MIN * progress
        else:
            # In unexplored areas, maintain high curiosity with gradual decay
            progress = current_explored_rate / ICM_EXPLORATION_THRESHOLD
            eta_adaptive = ICM_ETA_MAX * (1 - progress * 0.3)  # Only reduce by 30% before threshold
        
        # Strategy 2: Boost curiosity when exploration stagnates
        # Calculate recent exploration progress (if we have step history)
        step = len(self.episode_buffer[0])
        if step > 10:  # Need some history
            prev_rate = getattr(self, '_prev_explored_rate', 0.0)
            exploration_velocity = current_explored_rate - prev_rate
            
            # If exploration is stagnating, boost curiosity
            if exploration_velocity < 0.005:  # Very slow exploration progress
                eta_adaptive *= 1.5  # Boost by 50%
        
        # Store current rate for next step
        self._prev_explored_rate = current_explored_rate
        
        # Ensure within bounds
        eta_adaptive = max(ICM_ETA_MIN, min(ICM_ETA_MAX, eta_adaptive))
        
        return eta_adaptive
    
    def get_observation_from_buffer(self, index):
        """Reconstruct observation tuple from buffer at given index"""
        if index < 0 or index >= len(self.episode_buffer[0]):
            return None
            
        return (
            self.episode_buffer[0][index],  # node_inputs
            self.episode_buffer[1][index],  # node_padding_mask
            self.episode_buffer[2][index],  # edge_mask
            self.episode_buffer[3][index],  # current_index
            self.episode_buffer[4][index],  # current_edge
            self.episode_buffer[5][index]   # edge_padding_mask
        )


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=True)
    worker.run_episode()
