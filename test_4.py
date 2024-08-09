from z3 import *
import networkx as nx
import time

def createGraph(distances):
    G = nx.Graph()
    for (i, j, length) in distances:
        G.add_edge(i, j, length=length)
    return G

def find_upper_bound(G, sizes, max_loads, n_couriers, initial_margin=0.5, step=0.1):
    # Find the distance between the depot (node 0) and each other node
    depot_distances = []
    for j in G.nodes:
        if j != 0 and G.has_edge(0, j):  # Check if edge exists
            depot_distances.append(G.edges[0, j]['length'])

    if not depot_distances:
        raise ValueError("No valid edges found between depot and other nodes.")

    # Sort distances and select the smallest ones corresponding to the number of couriers
    sorted_distances = sorted(depot_distances)[:n_couriers]

    # Estimate the initial upper bound by summing the smallest distances (assuming a round trip)
    upper_bound = sum(sorted_distances) * 2
    upper_bound = int(upper_bound * (1 + initial_margin))

    return upper_bound

def extract_paths(G, model, n_couriers):
    paths = []
    for k in range(n_couriers):
        path = []
        current_node = 0
        while True:
            found_next = False
            for j in G.nodes:
                if j != current_node and model.eval(Bool(f"x_{current_node}_{j}_{k}")).is_true():
                    path.append((current_node, j))
                    current_node = j
                    found_next = True
                    break
            if not found_next:
                break
        paths.append(path)
    return paths


def find_model(instance, config, initial_upper_bound=None):
    distances, sizes, max_loads, n_couriers = instance

    G = createGraph(distances)
    s = Optimize()
    s.set("timeout", 300 * 1000)  # Set timeout to 300 seconds (adjustable)

    # Create decision variables
    x = {(i, j, k): Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers) for i in G.nodes for j in G.nodes if i != j}

    # Add constraints
    for i in G.nodes:
        if i != 0:  # Exclude the depot
            s.add(Sum([x[i, j, k] for j in G.nodes if i != j for k in range(n_couriers)]) == 1)

    for k in range(n_couriers):
        s.add(Sum([If(x[i, j, k], sizes[j], 0) for i in G.nodes if i != 0 for j in G.nodes if i != j and j != 0]) <= max_loads[k])

    objective = Int('objective')
    s.add(objective == Sum([If(x[i, j, k], G.edges[i, j]['length'], 0)
                            for k in range(n_couriers)
                            for i in G.nodes
                            for j in G.nodes if i != j]))

    # Dynamic upper bound adjustment
    if initial_upper_bound is None:
        upper_bound = find_upper_bound(G, sizes, max_loads, n_couriers)
    else:
        upper_bound = initial_upper_bound

    while True:
        s.push()
        s.add(objective <= upper_bound)
        s.minimize(objective)

        start_time = time.time()

        if s.check() == sat:
            elapsed_time = time.time() - start_time
            model = s.model()
            total_distance_value = model.evaluate(objective)
            paths = extract_paths(G, model, n_couriers)
            print(f'Solution found: Distance = {total_distance_value}')
            return total_distance_value, elapsed_time, paths, upper_bound

        else:
            s.pop()  # Restore the previous state before increasing the upper bound
            print("No solution found with the current upper bound. Increasing the upper bound...")
            upper_bound = int(upper_bound * 1.1)  # Increase the upper bound by 10%

# Example instance (replace with actual data)
distances = [(0, 1, 10), (0, 2, 20), (1, 2, 15), (1, 3, 10), (2, 3, 25), (0, 3, 30)]
sizes = {1: 5, 2: 10, 3: 8}  # Sizes of the items at each node
max_loads = [15, 20]  # Maximum load each courier can carry
n_couriers = 2  # Number of couriers

instance = (distances, sizes, max_loads, n_couriers)
config = {}  # Placeholder for any additional configuration

# Call the function
total_distance, elapsed_time, paths, final_upper_bound = find_model(instance, config)

print(f"Total distance: {total_distance}")
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Paths: {paths}")
print(f"Final upper bound: {final_upper_bound}")
