import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import random
from sklearn.model_selection import train_test_split

from model import PolicyNet
from expert.expert_data_collector import ExpertDataCollector
from parameter import *


class BehaviorCloningTrainer:
    def __init__(self, policy_net, device='cpu', tensorboard_writer=None):
        self.policy_net = policy_net
        self.device = device
        self.optimizer = optim.Adam(policy_net.parameters(), lr=BC_LR)
        self.criterion = nn.CrossEntropyLoss()
        self.writer = tensorboard_writer
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_step(self, observations, actions):
        """Single training step"""
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        # Prepare batch data
        batch_size = len(observations)
        
        # Stack observations into tensors - remove extra batch dimension from individual observations
        node_inputs = torch.stack([obs[0].squeeze(0) for obs in observations]).to(self.device)
        node_padding_mask = torch.stack([obs[1].squeeze(0) for obs in observations]).to(self.device)
        edge_mask = torch.stack([obs[2].squeeze(0) for obs in observations]).to(self.device)
        current_index = torch.stack([obs[3].squeeze(0) for obs in observations]).to(self.device)
        current_edge = torch.stack([obs[4].squeeze(0) for obs in observations]).to(self.device)
        edge_padding_mask = torch.stack([obs[5].squeeze(0) for obs in observations]).to(self.device)
        
        # Stack actions - ensure proper shape for CrossEntropyLoss
        actions_tensor = torch.stack([action.flatten()[0] for action in actions]).to(self.device)
        
        # Forward pass
        logp = self.policy_net(node_inputs, node_padding_mask, edge_mask, 
                              current_index, current_edge, edge_padding_mask)
        
        # Compute loss
        loss = self.criterion(logp, actions_tensor)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Calculate accuracy
        predicted = torch.argmax(logp, dim=1)
        accuracy = (predicted == actions_tensor).float().mean()
        
        return loss.item(), accuracy.item()
    
    def validate_step(self, observations, actions):
        """Single validation step"""
        self.policy_net.eval()
        
        with torch.no_grad():
            # Prepare batch data - remove extra batch dimension from individual observations
            node_inputs = torch.stack([obs[0].squeeze(0) for obs in observations]).to(self.device)
            node_padding_mask = torch.stack([obs[1].squeeze(0) for obs in observations]).to(self.device)
            edge_mask = torch.stack([obs[2].squeeze(0) for obs in observations]).to(self.device)
            current_index = torch.stack([obs[3].squeeze(0) for obs in observations]).to(self.device)
            current_edge = torch.stack([obs[4].squeeze(0) for obs in observations]).to(self.device)
            edge_padding_mask = torch.stack([obs[5].squeeze(0) for obs in observations]).to(self.device)
            
            # Stack actions - ensure proper shape for CrossEntropyLoss
            actions_tensor = torch.stack([action.flatten()[0] for action in actions]).to(self.device)
            
            # Forward pass
            logp = self.policy_net(node_inputs, node_padding_mask, edge_mask, 
                                  current_index, current_edge, edge_padding_mask)
            
            # Compute loss and accuracy
            loss = self.criterion(logp, actions_tensor)
            predicted = torch.argmax(logp, dim=1)
            accuracy = (predicted == actions_tensor).float().mean()
            
        return loss.item(), accuracy.item()
    
    def create_batches(self, observations, actions, batch_size):
        """Create batches from observations and actions"""
        assert len(observations) == len(actions)
        
        # Shuffle data
        combined = list(zip(observations, actions))
        random.shuffle(combined)
        observations, actions = zip(*combined)
        
        # Create batches
        batches = []
        for i in range(0, len(observations), batch_size):
            batch_obs = observations[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batches.append((batch_obs, batch_actions))
        
        return batches
    
    def train(self, expert_demonstrations):
        """Main training loop for behavior cloning"""
        print(f"Starting behavior cloning with {len(expert_demonstrations)} demonstrations")
        
        # Extract observations and actions
        observations = []
        actions = []
        
        for demo in expert_demonstrations:
            observations.append(demo['observation'])
            actions.append(demo['action'])
        
        print(f"Extracted {len(observations)} observation-action pairs")
        
        # Split into train/validation
        train_obs, val_obs, train_actions, val_actions = train_test_split(
            observations, actions, test_size=BC_VALIDATION_SPLIT, random_state=42
        )
        
        print(f"Training set: {len(train_obs)} samples")
        print(f"Validation set: {len(val_obs)} samples")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(BC_EPOCHS):
            # Training
            train_batches = self.create_batches(train_obs, train_actions, BC_BATCH_SIZE)
            
            train_losses = []
            train_accuracies = []
            
            for batch_obs, batch_actions in train_batches:
                loss, accuracy = self.train_step(batch_obs, batch_actions)
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_accuracy = np.mean(train_accuracies)
            
            # Validation
            val_batches = self.create_batches(val_obs, val_actions, BC_BATCH_SIZE)
            
            val_losses = []
            val_accuracies = []
            
            for batch_obs, batch_actions in val_batches:
                loss, accuracy = self.validate_step(batch_obs, batch_actions)
                val_losses.append(loss)
                val_accuracies.append(accuracy)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            
            # Logging
            if self.writer:
                self.writer.add_scalar('BC/Train_Loss', avg_train_loss, epoch)
                self.writer.add_scalar('BC/Train_Accuracy', avg_train_accuracy, epoch)
                self.writer.add_scalar('BC/Val_Loss', avg_val_loss, epoch)
                self.writer.add_scalar('BC/Val_Accuracy', avg_val_accuracy, epoch)
            
            # Print progress
            if epoch % 10 == 0 or epoch == BC_EPOCHS - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {avg_train_accuracy:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch, avg_val_loss, 'best_bc_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= BC_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience: {BC_PATIENCE})")
                break
        
        print(f"Behavior cloning completed. Best validation loss: {best_val_loss:.4f}")
        
        # Load best model
        self.load_checkpoint('best_bc_model.pth')
        
        return best_val_loss
    
    def save_checkpoint(self, epoch, val_loss, filename):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load training checkpoint"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {filename}")
            return checkpoint.get('val_loss', float('inf'))
        else:
            print(f"Checkpoint {filename} not found")
            return float('inf')


def run_behavior_cloning(policy_net, device='cpu', tensorboard_writer=None):
    """Main function to run behavior cloning pretraining"""
    print("=" * 60)
    print("STARTING BEHAVIOR CLONING PRETRAINING")
    print("=" * 60)
    
    # Initialize data collector
    collector = ExpertDataCollector(device=device)
    
    # Load or collect expert demonstrations
    if os.path.exists(BC_DEMONSTRATIONS_PATH):
        print(f"Loading existing demonstrations from {BC_DEMONSTRATIONS_PATH}")
        expert_demonstrations = collector.load_demonstrations(BC_DEMONSTRATIONS_PATH)
    else:
        print("No existing demonstrations found. Collecting new ones...")
        expert_demonstrations = collector.collect_expert_demonstrations(
            num_episodes=EXPERT_EPISODES, 
            save_path=BC_DEMONSTRATIONS_PATH
        )
    
    if len(expert_demonstrations) == 0:
        print("ERROR: No expert demonstrations available!")
        return False
    
    # Print dataset statistics
    stats = collector.get_dataset_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize BC trainer
    bc_trainer = BehaviorCloningTrainer(policy_net, device, tensorboard_writer)
    
    # Run behavior cloning
    best_val_loss = bc_trainer.train(expert_demonstrations)
    
    print(f"Behavior cloning completed with validation loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    return True


def test_behavior_cloning():
    """Test behavior cloning implementation"""
    device = torch.device('cuda' if USE_GPU_GLOBAL and torch.cuda.is_available() else 'cpu')
    
    # Create policy network
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # Create tensorboard writer
    writer = SummaryWriter('test_bc_logs')
    
    # Run behavior cloning
    success = run_behavior_cloning(policy_net, device, writer)
    
    writer.close()
    
    if success:
        print("Behavior cloning test completed successfully!")
    else:
        print("Behavior cloning test failed!")
    
    return success


if __name__ == "__main__":
    test_behavior_cloning()