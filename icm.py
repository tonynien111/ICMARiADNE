import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntrinsicCuriosityModule(nn.Module):
    # ICM
    def __init__(self, node_input_dim, embedding_dim, action_dim, feature_dim=256):
        super(IntrinsicCuriosityModule, self).__init__()
        
        self.node_input_dim = node_input_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Feature network 
        self.feature_net = nn.Sequential(
            nn.Linear(node_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, feature_dim)
        )
        
        # Inverse model
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, action_dim)
        )
        
        # Forward model 
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, feature_dim)
        )
    
    def encode_state(self, node_features):
        batch_size, num_nodes, _ = node_features.shape
        
        # Encode each node
        node_encoded = self.feature_net(node_features.view(-1, self.node_input_dim))
        node_encoded = node_encoded.view(batch_size, num_nodes, self.feature_dim)
        
        # Aggregate node features 
        state_features = torch.mean(node_encoded, dim=1)
        
        return state_features
    
    def forward(self, current_state, next_state, action):
        # Encode states into feature representations
        current_state_features = self.encode_state(current_state)
        next_state_features = self.encode_state(next_state)
        
        # Inverse model: predict action from state transition
        inverse_input = torch.cat([current_state_features, next_state_features], dim=1)
        predicted_action = self.inverse_net(inverse_input)
        
        # Forward model: predict next state from current state and action
        forward_input = torch.cat([current_state_features, action], dim=1)
        predicted_next_state = self.forward_net(forward_input)
        
        return predicted_action, predicted_next_state, current_state_features, next_state_features
    
    def compute_intrinsic_reward(self, current_state, next_state, action):
        with torch.no_grad():
            current_state_features = self.encode_state(current_state)
            next_state_features = self.encode_state(next_state)
            
            # Forward model prediction
            forward_input = torch.cat([current_state_features, action], dim=1)
            predicted_next_state = self.forward_net(forward_input)
            
            # Intrinsic reward is the prediction error (L2 distance)
            prediction_error = F.mse_loss(predicted_next_state, next_state_features, reduction='none')
            intrinsic_reward = torch.mean(prediction_error, dim=1)  # Average over feature dimension
            
        return intrinsic_reward
    
    def compute_losses(self, current_state, next_state, action):
        predicted_action, predicted_next_state, current_state_features, next_state_features = self.forward(
            current_state, next_state, action)
        
        # loss for inverse model
        inverse_loss = F.mse_loss(predicted_action, action)
        
        # loss for forward model 
        forward_loss = F.mse_loss(predicted_next_state, next_state_features)
        
        return inverse_loss, forward_loss


class ICMAgent:
    def __init__(self, node_input_dim, embedding_dim, action_dim, device='cpu'):
        self.icm = IntrinsicCuriosityModule(node_input_dim, embedding_dim, action_dim).to(device)
        self.device = device
        
    def action_to_onehot(self, action_indices, action_dim):
        #handle different shape
        if len(action_indices.shape) == 0:
            # Scalar tensor - convert to batch of size 1
            action_indices = action_indices.unsqueeze(0)
        
        batch_size = action_indices.shape[0]
        one_hot = torch.zeros(batch_size, action_dim, device=self.device)
        
        # Ensure action_indices has proper shape for scatter_
        if len(action_indices.shape) == 1:
            action_indices = action_indices.unsqueeze(1)
        
        one_hot.scatter_(1, action_indices, 1)
        return one_hot
    
    def extract_node_features(self, observation):
        node_inputs, node_padding_mask = observation[0], observation[1]
        
        # Handle different input formats
        if isinstance(node_inputs, list):
            node_inputs = torch.stack(node_inputs, dim=0)
        
        if len(node_inputs.shape) == 2:
            node_inputs = node_inputs.unsqueeze(0)  #adding batch
            
        return node_inputs
    
    def compute_intrinsic_reward(self, current_obs, next_obs, action_indices, action_dim):
        #Compute intrinsic curiosity reward

        current_state = self.extract_node_features(current_obs)
        next_state = self.extract_node_features(next_obs)
        
        # Convert action indices to one-hot - handle various tensor shapes
        # Flatten to 1D if needed, but preserve batch dimension
        if len(action_indices.shape) > 1:
            action_indices = action_indices.squeeze()
        
        # Ensure we have at least 1D tensor
        if len(action_indices.shape) == 0:
            action_indices = action_indices.unsqueeze(0)
            
        action_onehot = self.action_to_onehot(action_indices, action_dim)
        
        intrinsic_reward = self.icm.compute_intrinsic_reward(current_state, next_state, action_onehot)
        return intrinsic_reward
    
    def compute_losses(self, current_obs, next_obs, action_indices, action_dim):

        current_state = self.extract_node_features(current_obs)
        next_state = self.extract_node_features(next_obs)
        
        # Convert action indices to one-hot - handle various tensor shapes
        # Flatten to 1D if needed, but preserve batch dimension
        if len(action_indices.shape) > 1:
            action_indices = action_indices.squeeze()
        
        # Ensure we have at least 1D tensor
        if len(action_indices.shape) == 0:
            action_indices = action_indices.unsqueeze(0)
            
        action_onehot = self.action_to_onehot(action_indices, action_dim)
        
        return self.icm.compute_losses(current_state, next_state, action_onehot)