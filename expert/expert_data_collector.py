import numpy as np
import pickle
import os
import time
import signal
import random
from contextlib import contextmanager

import torch 
from agent import Agent
from env import Env
# Try to import OR-Tools based planner, fallback to simple planner
try:
    from expert.expert_planner import ExpertPlanner
    EXPERT_PLANNER_AVAILABLE = True
except ImportError as e:
    print(f"OR-Tools planner not available ({e}), using simple expert planner")
    from expert.simple_expert import SimpleExpertPlanner as ExpertPlanner
    EXPERT_PLANNER_AVAILABLE = False
from parameter import *


class ExpertDataCollector:
    def __init__(self, device='cpu'):
        self.device = device
        self.expert_demonstrations = []

    def collect_expert_demonstrations(self, num_episodes=EXPERT_EPISODES, save_path=BC_DEMONSTRATIONS_PATH):
        """Collect expert demonstrations using ExpertPlanner"""
            
        print(f"Collecting {num_episodes} expert demonstrations...")
        
        for episode in range(num_episodes):
            print(f"Collecting episode {episode + 1}/{num_episodes}...")
            
            try:
                # Create environment
                env = Env(episode_index=episode, plot=False)
                print(f"  Environment initialized for episode {episode + 1}")
                
                # Create expert agent (without policy_net since we'll use ExpertPlanner)
                expert_agent = Agent(policy_net=None, device=self.device, plot=False)
                print(f"  Expert agent initialized")
                
                episode_data = []
                step_count = 0
                max_steps = BC_MAX_STEPS_PER_EPISODE
                
                while step_count < max_steps:
                    print(f"  Step {step_count + 1}/{max_steps}")
                    
                    # Update agent's planning state
                    expert_agent.update_planning_state(env.belief_info, env.robot_location)
                    print(f"    Planning state updated. Frontiers: {len(expert_agent.frontier)}")
                    
                    # Check if exploration is complete
                    if len(expert_agent.frontier) == 0 or expert_agent.utility.sum() == 0:
                        print("    No frontiers left or no utility, episode complete")
                        break
                    
                    # Get current observation
                    try:
                        observation = expert_agent.get_observation()
                        print(f"    Got observation with {len(observation)} components")
                    except Exception as e:
                        print(f"    Error getting observation: {e}")
                        break
                    
                    # Use expert planner to get next action with timeout
                    try:
                        with timeout_context(15):  # 15 second timeout
                            expert_planner = ExpertPlanner(expert_agent.node_manager)
                            paths = expert_planner.plan_coverage_paths([env.robot_location])
                            print(f"    Expert planner returned {len(paths[0]) if paths and paths[0] else 0} waypoints")
                    except TimeoutError:
                        print("    Expert planning timed out, using fallback")
                        # Simple fallback: move to node with highest utility
                        if len(expert_agent.node_coords) > 0:
                            utility = expert_agent.utility
                            if utility.sum() > 0:
                                best_node_idx = np.argmax(utility)
                                next_waypoint = expert_agent.node_coords[best_node_idx]
                                paths = [[env.robot_location, next_waypoint]]
                            else:
                                break
                        else:
                            break
                    except Exception as e:
                        print(f"    Error in expert planning: {e}")
                        break
                    
                    if not paths or not paths[0] or len(paths[0]) < 2:
                        print("    No valid paths found")
                        break
                    
                    # Get next waypoint from expert path
                    next_waypoint = paths[0][1]  # First waypoint in the path
                    print(f"    Next waypoint: {next_waypoint}")
                    
                    # Find the corresponding action index
                    try:
                        action_index = self._find_action_index(observation, next_waypoint, expert_agent)
                        print(f"    Action index: {action_index}")
                    except Exception as e:
                        print(f"    Error finding action index: {e}")
                        break
                    
                    if action_index is None:
                        print("    No valid action found")
                        break
                    
                    # Store state-action pair
                    try:
                        state_action = {
                            'observation': [obs.clone().detach() if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32) 
                                          for obs in observation],
                            'action': torch.tensor([action_index], dtype=torch.long).reshape(1, 1, 1),
                            'next_position': next_waypoint.copy()
                        }
                        episode_data.append(state_action)
                        print(f"    State-action pair stored")
                    except Exception as e:
                        print(f"    Error storing state-action pair: {e}")
                        break
                    
                    # Execute action in environment
                    try:
                        reward = env.step(next_waypoint)
                        step_count += 1
                        print(f"    Environment step executed. Reward: {reward:.3f}, Explored: {env.explored_rate:.3f}")
                    except Exception as e:
                        print(f"    Error executing environment step: {e}")
                        break
                    
                    # Early termination if exploration rate is high
                    if env.explored_rate > 0.95:
                        print("    High exploration rate reached, terminating episode")
                        break
                
                print(f"  Episode {episode + 1} completed with {len(episode_data)} transitions")
                self.expert_demonstrations.extend(episode_data)
                
            except Exception as e:
                print(f"  Error in episode {episode + 1}: {e}")
                continue
            
            if (episode + 1) % 5 == 0:
                print(f"Collected {episode + 1} episodes, total transitions: {len(self.expert_demonstrations)}")
        
        # Save demonstrations
        self.save_demonstrations(save_path)
        print(f"Expert demonstrations saved to {save_path}")
        return self.expert_demonstrations
    
    def _find_action_index(self, observation, target_position, agent):
        """Find the action index that corresponds to moving to target_position"""
        node_coords = agent.node_coords
        current_index = agent.current_index
        neighbor_indices = agent.neighbor_indices
        
        # Find the neighbor node closest to target position
        best_action = 0
        min_distance = float('inf')
        
        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx < len(node_coords):
                neighbor_pos = node_coords[neighbor_idx]
                distance = np.linalg.norm(neighbor_pos - target_position)
                if distance < min_distance:
                    min_distance = distance
                    best_action = i
        
        # Check if the action is valid (not pointing to current position)
        if len(neighbor_indices) > best_action and neighbor_indices[best_action] != current_index:
            return best_action
        else:
            return None
    
    def save_demonstrations(self, save_path):
        """Save expert demonstrations to file"""
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self.expert_demonstrations, f)
            print(f"Successfully saved {len(self.expert_demonstrations)} demonstrations")
        except Exception as e:
            print(f"Error saving demonstrations: {e}")
    
    def load_demonstrations(self, load_path):
        """Load expert demonstrations from file"""
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    self.expert_demonstrations = pickle.load(f)
                print(f"Loaded {len(self.expert_demonstrations)} expert demonstrations")
                return self.expert_demonstrations
            except Exception as e:
                print(f"Error loading demonstrations: {e}")
                return []
        else:
            print(f"File {load_path} not found")
            return []
    
    def get_demonstration_batch(self, batch_size):
        """Get a batch of expert demonstrations for behavior cloning training"""
        if len(self.expert_demonstrations) == 0:
            return []
            
        if len(self.expert_demonstrations) < batch_size:
            # If not enough data, sample with replacement
            demo_batch = random.choices(self.expert_demonstrations, k=batch_size)
        else:
            demo_batch = random.sample(self.expert_demonstrations, batch_size)
        
        return demo_batch
    
    def create_bc_dataset(self):
        """Create dataset suitable for behavior cloning training"""
        observations = []
        actions = []
        
        for demo in self.expert_demonstrations:
            observations.append(demo['observation'])
            actions.append(demo['action'])
        
        return observations, actions
    
    def get_dataset_stats(self):
        """Get statistics about the collected dataset"""
        if len(self.expert_demonstrations) == 0:
            return {}
        
        # Count action distribution
        action_counts = {}
        for demo in self.expert_demonstrations:
            action = demo['action'].item()
            action_counts[action] = action_counts.get(action, 0) + 1
        
        stats = {
            'total_demonstrations': len(self.expert_demonstrations),
            'action_distribution': action_counts,
            'unique_actions': len(action_counts)
        }
        
        return stats


@contextmanager
def timeout_context(seconds):
    """Context manager for timeouts"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Clear the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def collect_expert_data_script():
    """Script to collect expert demonstrations"""
    collector = ExpertDataCollector()
    
    # Check if demonstrations already exist
    demo_path = BC_DEMONSTRATIONS_PATH
    if os.path.exists(demo_path) and not BC_COLLECT_DATA_ON_START:
        print("Expert demonstrations already exist. Loading existing data...")
        collector.load_demonstrations(demo_path)
    else:
        print("Collecting new expert demonstrations...")
        collector.collect_expert_demonstrations(num_episodes=EXPERT_EPISODES, save_path=demo_path)  
    
    # Print dataset statistics
    stats = collector.get_dataset_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test batch generation
    if len(collector.expert_demonstrations) > 0:
        batch_size = min(BC_BATCH_SIZE, len(collector.expert_demonstrations))
        batch = collector.get_demonstration_batch(batch_size)
        print(f"\nTest batch generated: {len(batch)} samples")
    else:
        print("No expert demonstrations collected!")
    
    print("Expert data collection completed successfully!")
    return collector


if __name__ == "__main__":
    collect_expert_data_script()