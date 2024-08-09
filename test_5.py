from z3 import *
import networkx as nx
import numpy as np
import time

def read_instance(number):
    if number < 10:
        file_path = "instances/inst0" + str(number) + ".dat"
    else:
        file_path = "instances/inst" + str(number) + ".dat"
    variables = []
    with open(file_path, 'r') as file:
        for line in file:
            variables.append(line)
    n_couriers = int(variables[0].rstrip('\n'))
    n_items = int(variables[1].rstrip('\n'))
    max_loads = list(map(int, variables[2].rstrip('\n').split()))
    sizes = list(map(int, variables[3].rstrip('\n').split()))
    distances_matrix = [list(map(int, line.rstrip('\n').split())) for line in variables[4:]]

    return n_couriers, n_items, max_loads, sizes, distances_matrix

def createGraph(all_distances):
    G = nx.DiGraph()
    num_nodes = len(all_distances)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, length=all_distances[i][j])
            G.add_edge(j, i, length=all_distances[j][i])
    return G

def exactly_one(vars):
    return And(
        Or(vars),
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i + 1, len(vars))]))
    )

def find_model(instance, config):

    if instance < 1 or instance > 21:
        print(f"ERROR: Instance {instance} doesn't exist. Please insert a number between 1 and 21")
        return
    if config < 1 or config > 4:
        print(f"ERROR: Configuration {config} doesn't exist. Please insert a number between 1 and 4")
        return

    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

    G = createGraph(distances)

    s = Optimize()
    s.set("timeout", 1000 * 30)  # 30 seconds timeout

    x = {(i, j, k): Bool(f"x_{i}_{j}_{k}") for i in G.nodes for j in G.nodes for k in range(n_couriers)}

    # Simplified Constraints
    o = 0
    for k in range(n_couriers):
        s.add(Sum([If(x[o, j, k], 1, 0) for j in G.nodes if j != o]) == 1)
        s.add(Sum([If(x[i, o, k], 1, 0) for i in G.nodes if i != o]) == 1)

    for k in range(n_couriers):
        for j in G.nodes:
            s.add(Sum([If(x[i, j, k], 1, 0) for i in G.nodes if i != j]) <= 1)
            s.add(Sum([If(x[j, i, k], 1, 0) for i in G.nodes if i != j]) <= 1)
            s.add(Sum([If(x[i, j, k], 1, 0) for i in G.nodes if i != j]) ==
                  Sum([If(x[j, i, k], 1, 0) for i in G.nodes if i != j]))

    for j in G.nodes:
        if j != o:
            s.add(exactly_one([x[i, j, k] for k in range(n_couriers) for i in G.nodes if i != j]))

    total_distance = Sum([If(x[i, j, k], G.edges[i, j]['length'], 0)
                          for k in range(n_couriers)
                          for i in G.nodes
                          for j in G.nodes if i != j])

    s.minimize(total_distance)

    start_time = time.time()

    # Iteratively relax the constraints
    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()
        total_distance_value = model.evaluate(total_distance).as_long()

        paths = []
        for k in range(n_couriers):
            path = []
            current_node = 0
            while True:
                found_next = False
                for j in G.nodes:
                    if j != current_node and model.eval(x[current_node, j, k]).is_true():
                        path.append(j)
                        current_node = j
                        found_next = True
                        break
                if not found_next:
                    break
            paths.append(path)

        print(f"Solution found: {paths} with total distance {total_distance_value}")
        return total_distance_value, elapsed_time, paths

    else:
        print(f"No solution found with {n_couriers} couriers.")

    return None, None, None


# Example usage:
instance = 7  # Choose the instance number
config = 1    # Choose the configuration number
total_distance_value, elapsed_time, paths = find_model(instance, config)

if total_distance_value is not None:
    print(f"Total distance: {total_distance_value}")
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"Paths: {paths}")
else:
    print("No solution was found.")
