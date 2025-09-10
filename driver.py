import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *
from icm import ICMAgent
from expert.expert_action_generator import ExpertActionGenerator

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # initialize ICM if enabled
    global_icm_agent = None
    global_icm_optimizer = None
    if USE_ICM:
        global_icm_agent = ICMAgent(NODE_INPUT_DIM, EMBEDDING_DIM, ICM_ACTION_DIM, device)
        global_icm_optimizer = optim.Adam(global_icm_agent.icm.parameters(), lr=ICM_LR)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha'] 
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']
        
        # Load ICM if available and enabled
        if USE_ICM and global_icm_agent is not None and 'icm_model' in checkpoint:
            global_icm_agent.icm.load_state_dict(checkpoint['icm_model'])
            if 'icm_optimizer' in checkpoint:
                global_icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
            print("ICM model loaded successfully")

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)
    
    # add ICM weights if enabled
    if USE_ICM and global_icm_agent is not None:
        if device != local_device:
            icm_weights = global_icm_agent.icm.to(local_device).state_dict()
            global_icm_agent.icm.to(device)
        else:
            icm_weights = global_icm_agent.icm.state_dict()
        weights_set.append(icm_weights)

    # distributed training if multiple GPUs are available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'intrinsic_reward']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    buffer_size = 15
    if USE_ICM:
        buffer_size += 1  # index 15 for intrinsic rewards
    if USE_BC:
        buffer_size += 1  # index 16 (or 15 if no ICM) for expert actions
    for i in range(buffer_size):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)
                    
                    # ICM training data if enabled
                    intrinsic_reward = None
                    if USE_ICM and len(rollouts) > 15:
                        intrinsic_reward = torch.stack(rollouts[15]).to(device)
                    
                    # BC training data if enabled
                    expert_actions = None
                    if USE_BC:
                        expert_idx = 15
                        if USE_ICM:
                            expert_idx += 1  # Expert actions at index 16 if ICM enabled
                        if len(rollouts) > expert_idx:
                            expert_actions = torch.stack(rollouts[expert_idx]).to(device)

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index,
                                   current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                        next_current_index, next_current_edge, next_edge_padding_mask]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*observation)
                        q_values2 = dp_q_net2(*observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*observation)
                    
                    # SAC policy loss
                    sac_policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()
                    
                    # Behavior cloning loss
                    bc_loss = torch.tensor(0.0).to(device)
                    if USE_BC and expert_actions is not None:
                        # Get current BC weight
                        expert_action_gen = ExpertActionGenerator(None, device)
                        bc_weight = expert_action_gen.get_bc_weight(curr_episode)
                        
                        # Create mask for valid expert actions (not -1)
                        expert_actions_flat = expert_actions.squeeze(-1).squeeze(-1)  # Shape: [batch_size]
                        valid_expert_mask = (expert_actions_flat >= 0)
                        
                        if valid_expert_mask.sum() > 0:
                            # Extract valid expert actions and corresponding log probabilities
                            valid_expert_actions = expert_actions_flat[valid_expert_mask]  # Shape: [valid_samples]
                            valid_logp = logp[valid_expert_mask]  # Shape: [valid_samples, action_dim]
                            
                            # Negative log-likelihood loss for behavior cloning
                            bc_loss = -torch.gather(valid_logp, 1, valid_expert_actions.unsqueeze(1)).sum() / BATCH_SIZE
                            bc_loss = bc_weight * bc_loss
                    
                    # Combined policy loss
                    policy_loss = sac_policy_loss + bc_loss

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_observation)
                        next_q_values2 = dp_target_q_net2(*next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*observation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*observation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    
                    # ICM training
                    icm_inverse_loss = torch.tensor(0.0)
                    icm_forward_loss = torch.tensor(0.0)
                    if USE_ICM and global_icm_agent is not None:
                        try:
                            #Prepare action indices for ICM
                            action_indices = action.squeeze(-1)
                            
                            #ICM losses
                            icm_inverse_loss, icm_forward_loss = global_icm_agent.compute_losses(
                                observation, next_observation, action_indices, ICM_ACTION_DIM
                            )
                            
                            # Combined ICM loss with beta weighting
                            icm_total_loss = (1 - ICM_BETA) * icm_inverse_loss + ICM_BETA * icm_forward_loss
                            
                            # ICM optimization step
                            global_icm_optimizer.zero_grad()
                            icm_total_loss.backward()
                            icm_grad_norm = torch.nn.utils.clip_grad_norm_(global_icm_agent.icm.parameters(), max_norm=100, norm_type=2)
                            global_icm_optimizer.step()
                            
                        except Exception as e:
                            print(f"ICM training failed: {e}")
                            icm_inverse_loss = torch.tensor(0.0)
                            icm_forward_loss = torch.tensor(0.0)

                    target_q_update_counter += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                
                # setting icm and bc
                if USE_ICM and USE_BC:
                    data = [reward.mean().item(), value_prime.mean().item(), sac_policy_loss.item(), bc_loss.item(),
                            policy_loss.item(), q1_loss.item(), entropy.mean().item(), policy_grad_norm.item(), 
                            q_grad_norm.item(), log_alpha.item(), alpha_loss.item(), icm_inverse_loss.item(), 
                            icm_forward_loss.item(), *perf_data]
                elif USE_ICM:
                    data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                            entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                            alpha_loss.item(), icm_inverse_loss.item(), icm_forward_loss.item(), *perf_data]
                elif USE_BC:
                    data = [reward.mean().item(), value_prime.mean().item(), sac_policy_loss.item(), bc_loss.item(),
                            policy_loss.item(), q1_loss.item(), entropy.mean().item(), policy_grad_norm.item(), 
                            q_grad_norm.item(), log_alpha.item(), alpha_loss.item(), *perf_data]
                else:
                    data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                            entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                            alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)
            
            # update icm weight
            if USE_ICM and global_icm_agent is not None:
                if device != local_device:
                    icm_weights = global_icm_agent.icm.to(local_device).state_dict()
                    global_icm_agent.icm.to(device)
                else:
                    icm_weights = global_icm_agent.icm.state_dict()
                weights_set.append(icm_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                
                # save icm model
                if USE_ICM and global_icm_agent is not None:
                    checkpoint["icm_model"] = global_icm_agent.icm.state_dict()
                    checkpoint["icm_optimizer"] = global_icm_optimizer.state_dict()
                
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    
    if USE_ICM and USE_BC:
        reward, value, sac_policy_loss, bc_loss, total_policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, icm_inverse_loss, icm_forward_loss, travel_dist, success_rate, explored_rate, intrinsic_reward = tensorboard_data
        writer.add_scalar(tag='ICM/Inverse Loss', scalar_value=icm_inverse_loss, global_step=curr_episode)
        writer.add_scalar(tag='ICM/Forward Loss', scalar_value=icm_forward_loss, global_step=curr_episode)
        writer.add_scalar(tag='ICM/Intrinsic Reward', scalar_value=intrinsic_reward, global_step=curr_episode)
        writer.add_scalar(tag='BC/BC Loss', scalar_value=bc_loss, global_step=curr_episode)
        writer.add_scalar(tag='BC/SAC Policy Loss', scalar_value=sac_policy_loss, global_step=curr_episode)
        policy_loss = total_policy_loss
    elif USE_ICM:
        reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, icm_inverse_loss, icm_forward_loss, travel_dist, success_rate, explored_rate, intrinsic_reward = tensorboard_data
        writer.add_scalar(tag='ICM/Inverse Loss', scalar_value=icm_inverse_loss, global_step=curr_episode)
        writer.add_scalar(tag='ICM/Forward Loss', scalar_value=icm_forward_loss, global_step=curr_episode)
        writer.add_scalar(tag='ICM/Intrinsic Reward', scalar_value=intrinsic_reward, global_step=curr_episode)
    elif USE_BC:
        reward, value, sac_policy_loss, bc_loss, total_policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate, intrinsic_reward = tensorboard_data
        writer.add_scalar(tag='BC/BC Loss', scalar_value=bc_loss, global_step=curr_episode)
        writer.add_scalar(tag='BC/SAC Policy Loss', scalar_value=sac_policy_loss, global_step=curr_episode)
        policy_loss = total_policy_loss
    else:
        reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate, intrinsic_reward = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


if __name__ == "__main__":
    main()
