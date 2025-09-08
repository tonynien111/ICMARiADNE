"""
Simple expert planner that doesn't require OR-Tools
Fallback implementation for environments where OR-Tools has dependency issues
"""

import numpy as np
from copy import deepcopy
from utils import *
import quads


class SimpleExpertPlanner:
    """Simple expert planner using greedy utility-based selection"""
    
    def __init__(self, node_manager):
        self.node_manager = node_manager
        
    def plan_coverage_paths(self, robot_locations):
        """Simple greedy planning approach"""
        all_node_coords = []
        utility = []
        
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            utility.append(node.data.utility)
            
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = np.array(utility)
        
        if len(all_node_coords) == 0:
            print("Warning: No nodes found in node manager")
            return [[robot_locations[0], robot_locations[0]]]
        
        # Find nodes with positive utility
        q_indices = np.where(utility > 0)[0]
        
        if len(q_indices) == 0:
            print("Warning: No nodes with positive utility found")
            return [[robot_locations[0], robot_locations[0]]]
        
        # Simple greedy approach: go to node with highest utility
        best_utility_idx = q_indices[np.argmax(utility[q_indices])]
        best_node = all_node_coords[best_utility_idx]
        
        # Check if reachable using Dijkstra
        try:
            dist_dict, _ = self.node_manager.Dijkstra(robot_locations[0])
            key = (best_node[0], best_node[1])
            if key in dist_dict and dist_dict[key] < 1e8:
                return [[robot_locations[0], best_node]]
        except Exception as e:
            print(f"Dijkstra failed: {e}")
        
        # Fallback: go to closest node with utility
        robot_pos = robot_locations[0]
        distances = [np.linalg.norm(robot_pos - all_node_coords[idx]) for idx in q_indices]
        closest_idx = q_indices[np.argmin(distances)]
        closest_node = all_node_coords[closest_idx]
        
        return [[robot_locations[0], closest_node]]


class SimpleTempNodeManager:
    def __init__(self):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)

    def add_node_to_dict(self, coords, observable_frontiers):
        key = (coords[0], coords[1])
        node = SimpleTempNode(coords, observable_frontiers)
        self.nodes_dict.insert(point=key, data=node)


class SimpleTempNode:
    def __init__(self, coords, observable_frontiers):
        self.coords = coords
        self.observable_frontiers = observable_frontiers
        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0

    def delete_observed_frontiers(self, observed_frontiers):
        self.observable_frontiers = self.observable_frontiers - observed_frontiers
        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0