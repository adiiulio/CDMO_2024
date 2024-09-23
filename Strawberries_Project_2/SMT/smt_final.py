import sys
from z3 import *
import networkx as nx
import numpy as np
import time
import json
import os
import sys

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
    for l in range(4, len(lines)):
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
        file_path = f"data/instances/inst0{number}.dat"
    else:
        file_path = f"data/instances/inst{number}.dat"
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
    distances_matrix = transform_distance_matrix(variables)
    #read the distances
    return n_couriers, n_items, max_loads, [0]+sizes, distances_matrix

#function that creates the graph
def createGraph(distances):
    dist_size = distances.shape[0]
    size_item = distances.shape[0] - 1
    G = nx.DiGraph()
    # Add nodes to the graph
    G.add_nodes_from(range(dist_size))

    # Add double connections between nodes
    for i in range(dist_size):
        for j in range(i + 1, size_item + 1):
            G.add_edge(i, j)
            G.add_edge(j, i)

    # Assign edge lengths
    lengths = {(i, j): distances[i][j] for i, j in G.edges()}
    nx.set_edge_attributes(G, lengths, 'length')
    return G

#exactly one function
def exactly_one(vars):
    return And(
        Or(vars),
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))]))
    )

def calculate_lower_bound(G, all_distances):
    lower_bound = 0
    # Calculate the round trip distance from depot (node 0) to node i and back
    for i in G.nodes:
        round_trip_distance = all_distances[0, i] + all_distances[i, 0]
        if round_trip_distance > lower_bound:
            lower_bound = round_trip_distance
    return lower_bound

def find_model(instance, config, remaining_time, upper_bound = None):
    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

     #define the solver and the max time, and set the timeout
    max_time = 300
    s = Solver()
    if remaining_time is None:
        remaining_time = max_time
    s.set("timeout", int(remaining_time * 1000))

    #create the graph
    G = createGraph(distances)

    #create the decision variables

    #courier load for courier k
    courier_loads = [Int(f'Courier_load_for_{k}') for k in range(n_couriers)]

    #x = True if courier k takes the route from i to j
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in range(n_items + 1)]
    #gives the position or weight of each node
    u = [Int(f"u_{j}") for j in G.nodes]
    #represents un unknown value that the solver will attempt to resolve
    objective = Int('objective')

    lower_bound = calculate_lower_bound(G, distances)

    # there cannot be a route from a node to itself (x from i to i of each courier should be false)
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    # Every item must be delivered meaning that each 3-dimensional column must contain only 1 true value
    for j in G.nodes:
        if j != 0:
            s.add(exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j]))

    # Every node should be entered and left once and by the same vehicle
    for k in range(n_couriers):
        for i in G.nodes:
            leaves = Sum([x[i][j][k] for j in G.nodes if i != j])
            enters = Sum([x[j][i][k] for j in G.nodes if i != j])
            s.add(enters == leaves)

    # each courier leaves and enters exactly once in the depot
    for k in range(n_couriers):
        #Predecessors of the depot sum is 1
        s.add(Sum([x[i][0][k] for i in G.nodes if i != 0]) == 1)
        #Successors of the depot sum is 1
        s.add(Sum([x[0][j][k] for j in G.nodes if j != 0]) == 1)

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        # ensures that the sum of the package sizes assigned to a courier equals the total load that courier carries.
        s.add(courier_loads[k] == Sum([If(x[i][j][k], sizes[i], 0) for i, j in G.edges]))
        #courier load is not negative
        s.add(courier_loads[k] > 0)
        #courier load is less than or equal to its max load 
        s.add(courier_loads[k] <= max_loads[k])

    #NO SUBTOUR PROBLEM------------------------

    #MZT BASED APPROACH

    s.add(u[0] == 1)
    for i in G.nodes:
        if i != 0:
            s.add(u[i] >= 2)

    for k in range(n_couriers):
        for i in G.nodes:
            for j in G.nodes:
                if i != j and i != 0 and j != 0:
                    #the weight of the successor is more than or equal to the weight of the predecessor +1
                    s.add(x[i][j][k] * u[j] >= x[i][j][k] * (u[i] + 1))


    #SYMMETRY BREAKING CONSTRAINTS--------------------

    #if courier A takes route 1 and courier B takes route 2, the solution where these routes are swapped is not explored to reduce the search space
    if config == 3 or config == 4:
        for c in range(n_couriers-1):
            for j in G.nodes:
                if j != 0:
                    s.add(u[c] <= u[c + 1])

    #------IMPLIED CONSTRAINTS------------------
    #the total load of all vehicles doesn't exceed the sum of vehicles capacities
    if config == 2 or config == 4:
        s.add((Sum([courier_loads[k] for k in range(n_couriers)]) <= Sum(max_loads)))

    #all nodes must be visited after depot
    #used as a comparison threshold
    eps = 0.0005
    for j in G.nodes:
        if j != 0:
            #for every node aside from the depot the u value needs to be more than 0
            if config == 2 or config == 4:
                s.add((u[j] > eps))

    # OBJECTIVE FUNCTION

    #total distance travelled

    total_distance = Sum(
        [If(x[i][j][k], int(distances[i][j]), 0) 
            for k in range(n_couriers) 
            for i, j in G.edges])
    
    #represents the total distance traveled by a courier (or vehicle) along its route, assuming only one courier (k=0) is being considered.
    #we initialize this as a first guess for the max distance, then we updare the value
    max_distance = Sum(
        [If(x[i][j][0], int(distances[i][j]), 0) 
            for i, j in G.edges])

    #this finds the actual max distance, in z3 there is no max function
    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(distances[i][j]), 0) for i, j in G.edges])
        max_distance = If(temp > max_distance, temp, max_distance)

    s.add(max_distance >= lower_bound)

    if upper_bound is None:
        s.add(objective == max_distance)
    else:
        s.add(objective == max_distance)
        s.add(upper_bound > objective)

    start_time = time.time()
    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()

        paths = []
    
        for courier in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
            found = []
            for (i, j) in tour_edges:
                if i not in found and i != 0:
                    found.append(i)
                if j not in found and j != 0:
                    found.append(j)
            paths.append(found)

        new_objective = int(model.evaluate(objective).as_long())
        total_distance_value = int(model.evaluate(total_distance).as_long())
        max_distance_value = int(model.evaluate(max_distance).as_long())

        return elapsed_time, new_objective, paths, total_distance_value, max_distance_value

    else:
        elapsed_time = time.time() - start_time
        return elapsed_time, -1, [], 0, 0
    
def find_optimal(instance, config):
    #default values
    opt_obj, opt_solution, opt_total_dist, opt_max_dist = -1, [], None, None
    #first try to find a solution
    run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist = find_model(instance, config, 10, None)
    if temp_obj == -1:
        return 300, False, -1, [], None, None
    remaining_time = 300 - run_time
    opt_obj, opt_solution, opt_total_dist, opt_max_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist

    #while this carries on we try to find all the next solutions
    while run_time < 300:
        run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist = find_model(instance, config, remaining_time, temp_obj)
        remaining_time = remaining_time - run_time
        if temp_obj == -1:
            if (300 - remaining_time) >= 9:
                return 300, False, opt_obj, opt_solution, opt_total_dist, opt_max_dist
            else:
                return int(300 - remaining_time), True, opt_obj, opt_solution, opt_total_dist, opt_max_dist
        else:
            opt_obj, opt_solution, opt_total_dist, opt_max_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist

    return 300, False, -1, [], 0, 0



def initialize_results_file(instance_number, configurations, output_path):
    """ Initialize or reset the JSON results file with default entries for all configurations. """
    results = {}
    for config in configurations:
        results[str(config)] = {
            'time': 300,  # Assuming the max timeout of 300 seconds
            'objective': -1,
            'distance': 0,
            'solution': [],
            'optimal': False
        }
    # Write the initialized data to the file
    with open(output_path, "w") as file:
        json.dump(results, file, indent=3)

# def update_results_file(output_path, config, result):
#     """ Update the JSON results file with the given result for a specific configuration. """
#     try:
#         with open(output_path, "r") as file:
#             results = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         results = {}

#     results[str(config)] = result

#     with open(output_path, "w") as file:
#         json.dump(results, file, indent=3)

def update_results_file(output_path, config, result):
    """ Update the JSON results file with the given result for a specific configuration. """
    try:
        with open(output_path, "r") as file:
            results = json.load(file)
            #print(f"Loaded existing results: {results}")  # Debugging line
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Creating new results file at {output_path}")  # Debugging line
        results = {}

    # Update or add the result for the given configuration
    results[str(config)] = result
    #print(f"Updated results: {results}")  # Debugging line

    # Write back the updated results to the JSON file
    with open(output_path, "w") as file:
        json.dump(results, file, indent=3)
    print(f"Results written to {output_path}")  # Debugging line


def main():
    try:
        instance_number = int(sys.argv[1])
        config = int(sys.argv[2])

        output_path = f"data/results/results_smt/{instance_number}.JSON"
        configurations = range(1, 5)  # Assuming configurations are from 1 to 4, adjust as needed

        # Initialize the JSON file with default entries
        if not os.path.exists(output_path):
            initialize_results_file(instance_number, configurations, output_path)

        # Attempt to find the optimal solution
        runtime, status, obj, solution, total_distance, max_dist = find_optimal(instance_number, config)

        print(f'Instance {instance_number} Configuration {config}')
        print(f"Total distance: {total_distance}")
        print(f"Elapsed time: {runtime} seconds")
        print(f"Paths: {solution}")
        print(f'Max Distance: {max_dist}')
        print(f"Is optimal: {status}")

        # Prepare the result to be written in JSON
        result = {
            'time': runtime,
            'objective': obj,
            'distance': total_distance,
            'solution': solution,
            'optimal': status
        }

        # Update the results file with the new result
        update_results_file(output_path, config, result)

        print(f"Results for Instance {instance_number}, Configuration {config} written to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

