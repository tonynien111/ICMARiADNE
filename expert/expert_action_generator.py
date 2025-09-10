import numpy as np
import torch
from expert.expert_planner import ExpertPlanner
from parameter import *

class ExpertActionGenerator:
    """
    Generates expert actions using the expert planner for behavior cloning
    """
    
    def __init__(self, node_manager, device='cpu'):
        self.expert_planner = ExpertPlanner(node_manager)
        self.device = device
        self.current_expert_path = None
        self.path_index = 0
        self.last_robot_location = None
        
    def get_expert_action(self, observation, robot_location, node_manager):
        """
        Get expert action based on current observation and robot location
        
        Args:
            observation: Current observation tuple (same as agent observation)
            robot_location: Current robot location [x, y]
            node_manager: Current node manager with updated graph
            
        Returns:
            expert_action_index: Action index that expert would take
            is_expert_available: Whether expert action is available
        """
        try:
            # Update the expert planner's node manager
            self.expert_planner.node_manager = node_manager
            
            # Check if we need to replan (new location or no current path)
            if (self.current_expert_path is None or 
                self.path_index >= len(self.current_expert_path) or
                self.last_robot_location is None or
                np.linalg.norm(robot_location - self.last_robot_location) > NODE_RESOLUTION * 0.5):
                
                # Generate new expert path
                expert_paths = self.expert_planner.plan_coverage_paths([robot_location])
                
                if expert_paths and len(expert_paths[0]) > 1:
                    self.current_expert_path = expert_paths[0]
                    self.path_index = 0
                    self.last_robot_location = robot_location.copy()
                else:
                    return None, False
            
            # Get next waypoint from expert path
            if (self.current_expert_path is not None and 
                self.path_index < len(self.current_expert_path) - 1):
                
                next_expert_waypoint = self.current_expert_path[self.path_index + 1]
                expert_action_index = self._convert_waypoint_to_action(
                    next_expert_waypoint, observation, node_manager
                )
                
                if expert_action_index is not None:
                    self.path_index += 1
                    return expert_action_index, True
                    
        except Exception as e:
            print(f"Expert action generation failed: {e}")
            
        return None, False
    
    def _convert_waypoint_to_action(self, target_waypoint, observation, node_manager):
        """
        Convert expert waypoint to action index in the current action space
        
        Args:
            target_waypoint: Target waypoint coordinates [x, y]
            observation: Current observation
            node_manager: Current node manager
            
        Returns:
            action_index: Index of action leading to target waypoint
        """
        try:
            node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
            
            # Get current node coordinates
            current_node_idx = current_index[0, 0, 0].item()
            
            # Get all available next positions from current_edge
            available_edges = current_edge[0, :, 0]  # Shape: [K_SIZE]
            
            # Find the coordinates for each available edge
            best_action_idx = None
            min_distance = float('inf')
            
            for i, edge_node_idx in enumerate(available_edges):
                if edge_node_idx.item() == 0 and i > 0:  # Padding
                    break
                    
                # Get coordinates of this potential next node
                try:
                    edge_node_idx_val = edge_node_idx.item()
                    if edge_node_idx_val < len(node_manager.nodes_dict.all_points()):
                        node_coords = list(node_manager.nodes_dict.all_points())[edge_node_idx_val].data.coords
                        distance = np.linalg.norm(np.array(node_coords) - np.array(target_waypoint))
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_action_idx = i
                except (IndexError, AttributeError):
                    continue
            
            if best_action_idx is not None and min_distance < NODE_RESOLUTION * 1.5:
                return torch.tensor([[best_action_idx]]).to(self.device)
            
        except Exception as e:
            print(f"Waypoint to action conversion failed: {e}")
            
        return None
    
    def reset_expert_path(self):
        """Reset the current expert path (e.g., at episode start)"""
        self.current_expert_path = None
        self.path_index = 0
        self.last_robot_location = None
    
    def get_bc_weight(self, episode):
        """
        Get behavior cloning weight based on scheduling
        
        Args:
            episode: Current episode number
            
        Returns:
            bc_weight: Current BC weight
        """
        if not USE_BC:
            return 0.0
            
        if BC_SCHEDULE == "constant":
            return BC_WEIGHT
        elif BC_SCHEDULE == "linear_decay":
            decay = min(1.0, episode * BC_DECAY_RATE)
            return max(BC_MIN_WEIGHT, BC_WEIGHT * (1 - decay))
        elif BC_SCHEDULE == "exponential_decay":
            decay = np.exp(-episode * BC_DECAY_RATE)
            return max(BC_MIN_WEIGHT, BC_WEIGHT * decay)
        else:
            return BC_WEIGHT