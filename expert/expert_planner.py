from copy import deepcopy
import numpy as np

from utils import *
import quads

# Try to import OR-Tools solver, use fallback if not available
try:
    from expert.ortools_solver import solve_vrp
    ORTOOLS_AVAILABLE = True
except (ImportError, Exception) as e:
    ORTOOLS_AVAILABLE = False
    print(f"OR-Tools not available ({type(e).__name__}: {e}), using greedy fallback in ExpertPlanner")
    
    # Define dummy solve_vrp for consistency
    def solve_vrp(distance_matrix, robot_indices):
        raise NotImplementedError("OR-Tools not available")


class ExpertPlanner:
    def __init__(self, node_manager):
        self.max_iteration_step = 5
        self.node_manager = node_manager
        self.last_viewpoints = None

    def plan_coverage_paths(self, robot_locations):
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
        
        best_paths = None
        c_best = 1e10
        q_indices = np.where(utility > 0)[0]
        q_array = all_node_coords[q_indices]
        
        if len(q_indices) == 0:
            print("Warning: No nodes with positive utility found")
            return [[robot_locations[0], robot_locations[0]]]

        try:
            dist_dict, _ = self.node_manager.Dijkstra(robot_locations[0])
        except Exception as e:
            print(f"Warning: Dijkstra failed: {e}")
            return [[robot_locations[0], all_node_coords[q_indices[0]]]]

        if self.last_viewpoints:
            temp_node_manager = TempNodeManager()
            for node in self.node_manager.nodes_dict.__iter__():
                coords = node.data.coords
                observable_frontiers = deepcopy(node.data.observable_frontiers)
                temp_node_manager.add_node_to_dict(coords, observable_frontiers)

            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]

            for viewpoints in self.last_viewpoints:
                last_node = temp_node_manager.nodes_dict.find(viewpoints.tolist()).data
                if last_node.utility > 0:
                    v_list.append(last_node.coords)
                    observable_frontiers = last_node.observable_frontiers
                    index = np.argwhere(
                        last_node.coords[0] + last_node.coords[1] * 1j == q_array_prime[:, 0] + q_array_prime[:,
                                                                                                1] * 1j)[0][0]
                    q_array_prime = np.delete(q_array_prime, index, axis=0)
                    for coords in q_array_prime:
                        node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                        if node.utility > 0 and np.linalg.norm(coords - last_node.coords) < 2 * SENSOR_RANGE:
                            node.delete_observed_frontiers(observable_frontiers)

            q_utility = []
            for coords in q_array_prime:
                node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                q_utility.append(node.utility)
            q_utility = np.array(q_utility)

            iteration_count = 0
            max_iterations = min(20, len(q_array_prime))
            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0 and iteration_count < max_iterations:
                iteration_count += 1
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                node = temp_node_manager.nodes_dict.find(viewpoint_coords.tolist()).data
                if dist_dict[(node.coords[0], node.coords[1])] == 1e8:
                    observable_frontiers = set()
                else:
                    v_list.append(viewpoint_coords)
                    observable_frontiers = node.observable_frontiers
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                    if node.utility > 0 and np.linalg.norm(coords - viewpoint_coords) <= 2 * SENSOR_RANGE:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            best_paths = paths
            c_best = dist

        for i in range(min(self.max_iteration_step, 3)):
            temp_node_manager = TempNodeManager()
            for node in self.node_manager.nodes_dict.__iter__():
                coords = node.data.coords
                observable_frontiers = deepcopy(node.data.observable_frontiers)
                temp_node_manager.add_node_to_dict(coords, observable_frontiers)

            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]
            q_utility = utility[q_indices]
            
            iteration_count = 0
            max_iterations = min(20, len(q_array_prime))
            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0 and iteration_count < max_iterations:
                iteration_count += 1
                
                # Early termination if no significant improvement
                if iteration_count > 5 and q_utility.sum() < 0.1:
                    break
                    
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                node = temp_node_manager.nodes_dict.find(viewpoint_coords.tolist()).data
                if dist_dict[(node.coords[0], node.coords[1])] == 1e8:
                    observable_frontiers = set()
                else:
                    v_list.append(viewpoint_coords)
                    observable_frontiers = node.observable_frontiers
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                    if node.utility > 0:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            if dist < c_best:
                best_paths = paths
                c_best = dist
                self.last_viewpoints = v_list[len(robot_locations):]

        return best_paths

    def find_paths(self, viewpoints, robot_locations, all_node_coords, utility):
        size = len(viewpoints)
        path_matrix = []
        distance_matrix = np.ones((size, size), dtype=int) * 1000
        for i in range(size):
            path_matrix.append([])
            for j in range(size):
                path_matrix[i].append([])

        for i in range(size):
            dist_dict, prev_dict = self.node_manager.Dijkstra(viewpoints[i])
            for j in range(size):
                path, dist = self.node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict,
                                                                          viewpoints[j])
                assert dist != 1e8

                dist = dist.astype(int)
                distance_matrix[i][j] = dist
                path_matrix[i][j] = path

        robot_indices = [i for i in range(len(robot_locations))]
        for i in range(size):
            for j in robot_indices:
                distance_matrix[i][j] = 0

        # Use OR-Tools VRP solver with fallback
        if ORTOOLS_AVAILABLE:
            try:
                paths, max_dist = solve_vrp(distance_matrix, robot_indices)
            except Exception as e:
                print(f"VRP solver failed: {e}. Using fallback greedy approach.")
                paths, max_dist = self._greedy_fallback_solution(distance_matrix, robot_indices, size)
        else:
            print("OR-Tools not available, using greedy approach.")
            paths, max_dist = self._greedy_fallback_solution(distance_matrix, robot_indices, size)

        paths_coords = []
        for path, robot_location in zip(paths, robot_locations):
            path_coords = []
            for index1, index2 in zip(path[:-1], path[1:]):
                if index1 < len(viewpoints) and index2 < len(viewpoints):
                    start_pos = viewpoints[index1]
                    end_pos = viewpoints[index2]
                    # Simple straight line path for now
                    path_coords.extend([start_pos, end_pos])
            
            if len(path_coords) == 0:
                indices = np.argwhere(utility > 0).reshape(-1)
                node_coords = all_node_coords[indices]
                dist_dict, prev_dict = self.node_manager.Dijkstra(robot_location)
                nearest_utility_coords = robot_location
                nearest_dist = 1e8
                for coords in node_coords:
                    dist = dist_dict.get((coords[0], coords[1]), 1e8)
                    if 0 < dist < nearest_dist:
                        nearest_dist = dist
                        nearest_utility_coords = coords

                path_coords = [robot_location, nearest_utility_coords]

            paths_coords.append(path_coords)
        return paths_coords, max_dist

    def _greedy_fallback_solution(self, distance_matrix, robot_indices, size):
        """Simple greedy fallback when OR-Tools is not available"""
        paths = []
        max_dist = 0
        
        for robot_idx in robot_indices:
            path = [robot_idx]
            current_pos = robot_idx
            visited = set([robot_idx])
            
            while len(visited) < size:
                best_next = None
                best_dist = float('inf')
                
                for next_pos in range(size):
                    if next_pos not in visited and distance_matrix[current_pos][next_pos] < best_dist:
                        best_dist = distance_matrix[current_pos][next_pos]
                        best_next = next_pos
                
                if best_next is not None:
                    path.append(best_next)
                    visited.add(best_next)
                    current_pos = best_next
                else:
                    break
            
            paths.append(path)

            # Calculate route distance
            route_dist = 0
            for i in range(len(path) - 1):
                route_dist += distance_matrix[path[i]][path[i+1]]
            max_dist = max(max_dist, route_dist)
        
        return paths, max_dist


class TempNodeManager:
    def __init__(self):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)

    def add_node_to_dict(self, coords, observable_frontiers):
        key = (coords[0], coords[1])
        node = TempNode(coords, observable_frontiers)
        self.nodes_dict.insert(point=key, data=node)


class TempNode:
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