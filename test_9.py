from z3 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import json

#1. Normal, no implied constraints and no symmetry breaking
#2. With implied constraints
#3. With symmetry breaking
#4. With both implied and symmetry breaking

def transform_distance_matrix(lines):
    # Loop to read the lines for the distance matrix
    # Starting from 4 until the end
    for l in range(4, len(lines)):
        lines[l] = lines[l].rstrip('\n').split()

    # Move the last element of each row to the front
    # The last element represents the distance to the depot
    # Moving the last element to the front could ensure that the depot is correctly represented as the first node (index 0).
    for l in range (4, len(lines)): 
        lines[l] = [lines[l][-1]] + lines[l]
        del lines[l][-1] 

    # convert the distance matrix to an array for easier work
    distances = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])

    # Take the last row and insert it as the first one
    last_row = distances[-1]
    other_rows = distances[:-1]
    reordered_matrix = np.vstack([last_row, other_rows])

    # Convert all elements to integers
    reordered_matrix = reordered_matrix.astype(int)
    return reordered_matrix

#function that reads the instance
def read_instance(number):
    if number < 10:
        file_path = "instances/inst0" + str(number) + ".dat"  # inserire nome del file
    else:
        file_path = "instances/inst" + str(number) + ".dat"  # inserire nome del file
    variables = []
    with open(file_path, 'r') as file:
        for line in file:
            variables.append(line)
    #read the number of couriers
    n_couriers = int(variables[0].rstrip('\n'))
    #read the number of items
    n_items = int(variables[1].rstrip('\n'))
    #read the max load for each courier
    max_loads = list(map(int, variables[2].rstrip('\n').split()))
    #read the size of each package
    sizes = list(map(int, variables[3].rstrip('\n').split()))
    #read the distances
    distances_matrix = transform_distance_matrix(variables)
    
    return n_couriers, n_items, max_loads, [0]+sizes, distances_matrix

#function that creates the graph
def createGraph(all_distances):
    all_dist_size = all_distances.shape[0]
    size_item = all_distances.shape[0] - 1
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(all_dist_size))

    # Add double connections between nodes
    for i in range(all_dist_size):
        for j in range(i + 1, size_item + 1):  # size item + 1 because we enclude also the depot in the graph
            G.add_edge(i, j)
            G.add_edge(j, i)

    # Assign edge lengths
    lengths = {(i, j): all_distances[i][j] for i, j in G.edges()}
    nx.set_edge_attributes(G, lengths, 'length')

    return G

#exactly one function
def exactly_one(vars):
    return And(
        Or(vars),  # At least one is true
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))]))
    )

#function that finds the model
def find_model(instance, config, remaining_time, upper_bound=None):
    # Check if the input is correct
    if instance < 1 or instance > 21:
        print(f"ERROR: Instance {instance} doesn't exist. Please insert a number between 1 and 21")
        return
    if config < 1 or config > 4:
        print(f"ERROR: Configuration {config} doesn't exist. Please insert a number between 1 and 4")
        return

    # Read the instance
    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

    # Define the solver and set the timeout
    max_time = 300
    s = Optimize()
    if remaining_time is None:
        remaining_time = max_time
    s.set("timeout", (int(remaining_time) * 1000))

    # Create the graph containing all possible paths
    G = createGraph(distances)

    # Decision variables
    courier_loads = [Int(f'Courier_load_for_{k}') for k in range(n_couriers)]
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in range(n_items + 1)]
    u = [Int(f"u_{j}") for j in G.nodes]
    nodes = [Int(f'n_{j}') for j in G.nodes]

    # Constraints -------------------------------------------------------------------------------------------------------------------------------
    # (Existing constraints...)
    
    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    # Every item must be delivered
    # (each 3-dimensional column must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            s.add(exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j]))

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for k in range(n_couriers):
        for i in G.nodes:
            s1 = Sum([x[i][j][k] for j in G.nodes if i != j])
            s2 = Sum([x[j][i][k] for j in G.nodes if i != j])
            s.add(s1 == s2)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for k in range(n_couriers):
        s.add(Sum([x[i][0][k] for i in G.nodes if i != 0]) == 1)
        s.add(Sum([x[0][j][k] for j in G.nodes if j != 0]) == 1)

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        s.add(courier_loads[k] == Sum([If(x[i][j][k], sizes[i], 0) for i, j in G.edges]))
        s.add(courier_loads[k] > 0)
        s.add(courier_loads[k] <= max_loads[k])

    # Objective Function
    total_distance = Sum([If(x[i][j][k], G.edges[i, j]['length'], 0)
                          for k in range(n_couriers)
                          for i in G.nodes
                          for j in G.nodes
                          if i != j])

    # Initialize variables for max and min distances
    max_distance = Real('max_distance')
    min_distance = Real('min_distance')

    # Initialize min and max distance values
    s.add(min_distance == total_distance)  # Initially, assume the total distance is the min distance
    s.add(max_distance == total_distance)  # Initially, assume the total distance is the max distance

    # Calculating the actual min and max distances
    for k in range(n_couriers):
        temp_distance = Sum([If(x[i][j][k], G.edges[i, j]['length'], 0) for i in G.nodes for j in G.nodes if i != j])
        s.add(min_distance <= temp_distance)
        s.add(max_distance >= temp_distance)

    # Adding upper and lower bounds logic for total_distance
    if upper_bound is not None:
        s.add(total_distance <= upper_bound)
    else:
        upper_bound = sum(sum(row) for row in distances)  # Sum of all distances as a loose upper bound

    lower_bound = 0  # Start with the lowest possible total distance
    step_size = (upper_bound - lower_bound) / 10  # Define the step size

    best_model = None
    best_min_distance = None
    best_max_distance = None
    best_total_distance = None
    best_paths = None
    best_time = None

    while lower_bound < upper_bound:
        # Define the optimization objectives
        s.minimize(min_distance)
        s.minimize(max_distance)
        s.minimize(total_distance)
        start_time = time.time()
        if s.check() == sat:
            model = s.model()
            min_distance_val = model.evaluate(min_distance).as_decimal(6)
            max_distance_val = model.evaluate(max_distance).as_decimal(6)
            total_distance_val = model.evaluate(total_distance).as_decimal(6)

            # Update best model if it is better
            if best_min_distance is None or float(min_distance_val) < float(best_min_distance):
                best_model = model
                best_min_distance = min_distance_val
                best_max_distance = max_distance_val
                best_total_distance = total_distance_val

                # Extract paths
                best_paths = []
                for courier in range(n_couriers):
                    tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
                    found = []
                    for (i, j) in tour_edges:
                        if i not in found and i != 0:
                            found.append(i)
                        if j not in found and j != 0:
                            found.append(j)
                    best_paths.append(found)

                best_time = time.time() - start_time

            # Adjust bounds based on the new solution
            upper_bound = float(best_total_distance)
        else:
            lower_bound += step_size

    if best_model:
        print(f'Min distance = {best_min_distance}')
        print(f'Max distance = {best_max_distance}')
        print(f'Total distance = {best_total_distance}')
        return best_time, best_min_distance, best_paths, best_total_distance, best_max_distance

    else:
        print("No solution found within bounds.")
        return None, None, None, None, None

    
def find_best(instance, config):
    best_obj, best_solution = -1, []
    run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist = find_model(instance, config, 300,None)
    remaining_time = 300 - run_time
    best_obj, best_solution, best_total_dist, best_max_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist

    while remaining_time > 0:
        run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist = find_model(instance, config, remaining_time, temp_obj)
        remaining_time = remaining_time - run_time
        if temp_obj == -1:
            if (300 - remaining_time) >= 299:
                return 300, False, str(best_obj), best_solution, best_total_dist, best_max_dist
            else:
                return int(300 - remaining_time), True, str(best_obj), best_solution, best_total_dist, best_max_dist
        else:
            best_obj, best_solution, best_total_dist, best_max_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist

    print("time limit exceeded")
    print("Remaining time: ", remaining_time)
    return 300, False, str(best_obj), best_solution, best_total_dist, best_max_dist
    
instance = 1  # Choose the instance number
config = 1    # Choose the configuration number
#elapsed_time, new_objective, tot_item, total_distance, max_distance = find_model(instance, config, None)
runtime, status, obj, solution, total_distance, max_dist = find_best(instance, config)

if total_distance is not None:
    print(f"Total distance: {total_distance}")
    print(f"Elapsed time: {runtime} seconds")
    print(f"Paths: {solution}")
    #print(f"Is optimal: {is_optimal}")
else:
    print("No solution was found.")