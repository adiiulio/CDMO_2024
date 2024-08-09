from z3 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import json

# 1. Normal, no implied constraints and no symmetry breaking
# 2. With implied constraints
# 3. With symmetry breaking
# 4. With both implied and symmetry breaking

# Function that reads the instance
def read_instance(number):
    if number < 10:
        file_path = "instances/inst0" + str(number) + ".dat"  # inserire nome del file
    else:
        file_path = "instances/inst" + str(number) + ".dat"  # inserire nome del file
    variables = []
    with open(file_path, 'r') as file:
        for line in file:
            variables.append(line)
    # Read the number of couriers
    n_couriers = int(variables[0].rstrip('\n'))
    # Read the number of items
    n_items = int(variables[1].rstrip('\n'))
    # Read the max load for each courier
    max_loads = list(map(int, variables[2].rstrip('\n').split()))
    # Read the size of each package
    sizes = list(map(int, variables[3].rstrip('\n').split()))
    # Read the distances
    distances_matrix = [list(map(int, line.rstrip('\n').split())) for line in variables[4:]]

    return n_couriers, n_items, max_loads, [0]+sizes, distances_matrix

# Function that creates the graph
def createGraph(all_distances):
    num_rows = len(all_distances)
    num_cols = len(all_distances[0]) if num_rows > 0 else 0
    all_dist_size = num_rows
    size_item = num_rows - 1
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

# Exactly one function
def exactly_one(vars):
    return And(
        Or(vars),  # At least one is true
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))]))
    )

# Calculate the lower bound
def find_lower_bound(G, sizes, max_loads):
    min_dist = min(G.edges[i, j]['length'] for i in G.nodes for j in G.nodes if i != j)
    total_size = sum(sizes)
    min_load = min(max_loads)
    return min_dist * (total_size // min_load)

# Calculate the upper bound using a simple heuristic
def find_upper_bound(G, sizes, max_loads, n_couriers):
    from itertools import combinations
    min_distance = sum(sorted([G.edges[i, j]['length'] for i, j in combinations(G.nodes, 2)])[:n_couriers])
    upper_bound = min_distance * 2  # Simple heuristic to estimate an upper bound
    return upper_bound

# Function that finds the model
def find_model(instance, config, upper_bound=None):

    # Check if the input is correct 
    if instance < 1 or instance > 21:
        print(f"ERROR: Instance {instance} doesn't exist. Please insert a number between 1 and 21")
        return
    if config < 1 or config > 4:
        print(f"ERROR: Configuration {config} doesn't exist. Please insert a number between 1 and 4")
        return

    # Read the instance
    n_couriers, n_items, max_load, sizes, distances = read_instance(instance)

    # Define the solver and the max time, and set the timeout
    s = Optimize()
    s.set("timeout", (int(300) * 1000))

    # Create the graph containing all possible paths
    G = createGraph(distances)

    objective = Int('objective')

    # Calculate and set the lower bound
    #lower_bound = find_lower_bound(G, sizes, max_loads)
    lower_bound = 0
    for i in G.nodes:
        if distances[0][i] + distances[i][0] > lower_bound: lower_bound = distances[0][i] + distances[i][0]
    s.add(Sum([If(Bool(f"x_{i}_{j}_{k}"), G.edges[i, j]['length'], 0)
               for k in range(n_couriers)
               for i in G.nodes
               for j in G.nodes if i != j]) >= lower_bound)

    # Calculate and set the upper bound
    #upper_bound = find_upper_bound(G, sizes, max_loads, n_couriers)
    #s.add(Sum([If(Bool(f"x_{i}_{j}_{k}"), G.edges[i, j]['length'], 0)
    #          for k in range(n_couriers)
    #          for i in G.nodes
    #          for j in G.nodes if i != j]) <= upper_bound)

    # Create the decision variables

    # Courier load for courier k
    courier_loads = [Int(f'Courier_load_for_{k}') for k in range(n_couriers)] 

    # x[i][j][k] = True means that the route from i to j is used by courier k
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
            range(n_items + 1)]

    u = [Int(f"u_{j}") for j in G.nodes]

    nodes = [Int(f'n_{j}') for j in G.nodes]

    # Constraints -------------------------------------------------------------------------------------------------------------------------------

    # The load must be smaller than the max_loads and greater than 0
    # for k in range(n_couriers):
    #     const_1 = cour_load[k] > 0
    #     const_2 = cour_load[k] < max_loads[k]
    #     s.add(const_1)
    #     s.add(const_2)

    # Each courier must start and end at origin point o
    # o = 0
    # for k in range(n_couriers):
    #     const_3 = Sum([x[o][j][k] for j in G.nodes if j != 0]) == 1
    #     const_4 = Sum([x[i][o][k] for i in G.nodes if i != 0]) == 1
    #     s.add(const_3)
    #     s.add(const_4)

    # # Each node is visited at most once by the same courier
    # for k in range(n_couriers):
    #     for j in G.nodes:
    #         const_5 = Sum([x[i][j][k] for i in G.nodes if i != j]) <= 1
    #         s.add(const_5)

    # # Each node must be entered and left once by the same courier
    # for k in range(n_couriers):
    #     for j in G.nodes:
    #         a = Sum([x[i][j][k] for i in G.nodes if i != j]) 
    #         b = Sum([x[j][i][k] for i in G.nodes if i != j]) 
    #         const_6 = (a == b)
    #         s.add(const_6)

    # # There is no route from a node to itself
    # for k in range(n_couriers):
    #     const_7 = Sum([x[i][i][k] for i in G.nodes]) == 0
    #     s.add(const_7)

    # # Every item needs to be delivered by whichever courier
    # for j in G.nodes:
    #         if j != o:  # the origin point o is not considered
    #             const_8 = exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j])
    #             s.add(const_8)

    # # Constraint of the upper bounds of items a courier can bring
    # for k in range(n_couriers):
    #     m = max_loads[k] / np.min(sizes)
    #     const_9 = cour_load[k] <= m
    #     s.add(const_9)

    #CONSTRAINTS FROM THE OTHER PROJECT 
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
        s.add(courier_loads[k] <= max_load[k])


    # SYMMETRY BREAKING CONSTRAINTS --------------------

    # for c in range(n_couriers-1):
    #     for j in G.nodes:
    #         if j != 0:
    #             const_10 = u[c] <= u[c + 1]
    #             if config == 3 or config == 4:
    #                 s.add(const_10)

    # # IMPLIED CONSTRAINTS ------------------
    # # The total load of all vehicles doesn't exceed the sum of vehicles capacities
    # const_11 = Sum([cour_load[k] for k in range(n_couriers)]) <= Sum(max_loads)
    # if config == 2 or config == 4:
    #     s.add(const_11)

    # # All nodes must be visited after depot
    # eps = 1
    # for k in range(n_couriers):
    #     for i in G.nodes:
    #         for j in G.nodes:
    #             if i != 0 and j != 0 and i != j:
    #                 const_12 = u[i] - u[j] + n_items * x[i][j][k] <= n_items - eps
    #                 if config == 2 or config == 4:
    #                     s.add(const_12)

    # Objective function
    total_distance = Sum([If(x[i][j][k], G.edges[i, j]['length'], 0)
                          for k in range(n_couriers)
                          for i in G.nodes
                          for j in G.nodes if i != j])
    
    # min_distance = Sum(
    #     [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])

    # max_distance = Sum(
    #     [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])
    
    # for k in range(n_couriers):
    #     temp = Sum(
    #         [If(x[i][j][k], int(distances[i][j]), 0) for i, j in G.edges])
    #     min_distance = If(temp < min_distance, temp, min_distance)
    #     max_distance = If(temp > max_distance, temp, max_distance)

    # if upper_bound is None:
    #     s.minimize(total_distance)
    # else:
    #     s.add(upper_bound > total_distance)
    #     s.minimize(Sum(total_distance, (max_distance - min_distance)))
    # s.add(total_distance >= lower_bound)

    #minimization = s.minimize(total_distance)

    max_distance = Sum(
        [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])
    min_distance = Sum(
        [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])
    
    #max_distance_1 = s.upper(minimization)

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(distances[i][j]), 0) for i, j in G.edges])
        max_distance = If(temp > max_distance, temp, max_distance)
        min_distance = If(temp < min_distance, temp, min_distance)
    

    if upper_bound is None:
            s.add(objective == max_distance)
    else:
        s.add(objective == max_distance)
        s.add(upper_bound > objective)
    s.add(max_distance >= lower_bound)

    start_time = time.time()

    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()
        print(f'Max distance = {model.evaluate(max_distance)}')
        print(f'Min distance = {model.evaluate(min_distance)}')

        paths = []
    
        for courier in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
            print(f'the path for courier {courier} is {tour_edges}')
            found = []
            for (i, j) in tour_edges:
                if i not in found and i != 0:
                    found.append(i)
                if j not in found and j != 0:
                    found.append(j)
            paths.append(found)

        new_objective = model.evaluate(objective)

        return elapsed_time, new_objective, paths, model.evaluate(total_distance), model.evaluate(max_distance)
    else:
        elapsed_time = time.time() - start_time
        return elapsed_time, -1, [], 0, 0

    # Check if satisfiable
    # if s.check() == sat:
    #     elapsed_time = time.time() - start_time
    #     model = s.model()
    #     total_distance_value = model.evaluate(total_distance)
    #     print(f'Max distance = {model.evaluate(max_distance)}')

    #     paths = []
    #     for courier in range(n_couriers):
    #         tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
    #         found = []
    #         for (i, j) in tour_edges:
    #             if i not in found and i != 0:
    #                 found.append(i)
    #             if j not in found and j != 0:
    #                 found.append(j)
    #         paths.append(found)

    #     lower_bound = s.lower(minimization)
    #     upper_bound = s.upper(minimization)
    #     print(upper_bound)
    #     is_optimal = 'false'
    #     if lower_bound == upper_bound:
    #         is_optimal = 'true'
        
    #     print(f'solution - {paths} - Distance = {total_distance_value}')
    #     return total_distance_value, elapsed_time, paths, is_optimal

    # else:
    #     print(f"No solution found with {n_couriers} couriers. Increasing the number of couriers...")

    # print("No solution found.")
    # return None, None, None, None
    




# Example usage:
instance = 1  # Choose the instance number
config = 1    # Choose the configuration number
elapsed_time, new_objective, tot_item, total_distance, max_distance = find_model(instance, config, None)

if total_distance is not None:
    print(f"Total distance: {total_distance}")
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"Paths: {tot_item}")
    #print(f"Is optimal: {is_optimal}")
else:
    print("No solution was found.")
