
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from typing import List, Tuple, Optional


def create_data_model(distance_matrix, robot_indices):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = len(robot_indices)
    data['starts'] = robot_indices
    data['ends'] = robot_indices
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))


def output_solution(manager, routing, solution, data):
    paths = []
    dists = []
    for vehicle_id in range(data['num_vehicles']):
        path = []
        index = routing.Start(vehicle_id)
        route_distance = 0
        path.append(index)
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            path.append(manager.IndexToNode(index))
        dists.append(route_distance)
        paths.append(path[:-1])

    return paths, max(dists)


def solve_vrp(distance_matrix, robot_indices):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(distance_matrix, robot_indices)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(50)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Add timeout to prevent hanging
    search_parameters.time_limit.FromSeconds(10)

    # Solve the problem.
    try:
        solution = routing.SolveWithParameters(search_parameters)

        if solution is not None:
            # Print solution on console.
            paths, dist = output_solution(manager, routing, solution, data)
            return paths, dist
        else:
            print("OR-Tools found no solution, using greedy fallback")
            return _greedy_fallback_solution(distance_matrix, robot_indices)
    except Exception as e:
        print(f"OR-Tools solver failed: {e}, using greedy fallback")
        return _greedy_fallback_solution(distance_matrix, robot_indices)


def _greedy_fallback_solution(distance_matrix, robot_indices):
    """Simple greedy fallback when OR-Tools fails"""
    n_nodes = distance_matrix.shape[0]
    n_robots = len(robot_indices)
    
    # Simple assignment: divide nodes among robots
    paths = []
    max_dist = 0
    
    # Get all non-robot nodes
    all_nodes = set(range(n_nodes))
    robot_set = set(robot_indices)
    target_nodes = list(all_nodes - robot_set)
    
    if not target_nodes:
        # No target nodes, return simple paths
        for robot_idx in robot_indices:
            paths.append([robot_idx])
        return paths, 0
    
    # Distribute nodes among robots
    nodes_per_robot = max(1, len(target_nodes) // n_robots)
    
    start_idx = 0
    for i, robot_idx in enumerate(robot_indices):
        path = [robot_idx]
        
        # Assign nodes to this robot
        end_idx = min(start_idx + nodes_per_robot, len(target_nodes))
        assigned_nodes = target_nodes[start_idx:end_idx]
        
        # Sort assigned nodes by distance from robot
        if assigned_nodes:
            assigned_nodes.sort(key=lambda x: distance_matrix[robot_idx][x])
            path.extend(assigned_nodes)
        
        paths.append(path)
        
        # Calculate route distance
        route_dist = 0
        for j in range(len(path) - 1):
            route_dist += distance_matrix[path[j]][path[j+1]]
        max_dist = max(max_dist, route_dist)
        
        start_idx = end_idx
        if start_idx >= len(target_nodes):
            break
    
    return paths, max_dist


# Legacy functions for backward compatibility
def solve_tsp(distance_matrix: np.ndarray, start_idx: int = 0) -> Tuple[List[int], int]:
    """Solve TSP using VRP with single vehicle"""
    routes, max_dist = solve_vrp(distance_matrix, [start_idx])
    if routes:
        return routes[0], max_dist
    else:
        return [start_idx], 0


def optimize_viewpoint_sequence(viewpoints: List[np.ndarray], robot_location: np.ndarray,
                               distance_func=None) -> Tuple[List[int], float]:
    """Optimize viewpoint sequence using TSP"""
    if distance_func is None:
        distance_func = lambda p1, p2: np.linalg.norm(p1 - p2)
    
    # Create distance matrix including robot position
    all_points = [robot_location] + list(viewpoints)
    n_points = len(all_points)
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                distance_matrix[i][j] = distance_func(all_points[i], all_points[j])
    
    # Solve TSP starting from robot position
    try:
        route, total_dist = solve_tsp(distance_matrix, start_idx=0)
        # Convert back to viewpoint indices
        viewpoint_sequence = [idx - 1 for idx in route[1:] if idx > 0]
        return viewpoint_sequence, total_dist
    except Exception as e:
        print(f"TSP optimization failed: {e}")
        return list(range(len(viewpoints))), 0


if __name__ == "__main__":
    # Test with sample data
    test_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 12, 18],
        [15, 12, 0, 8],
        [20, 18, 8, 0]
    ])
    robot_indices = [0]
    paths, dist = solve_vrp(test_matrix, robot_indices)
    print(f"Test result: {paths}, distance: {dist}")