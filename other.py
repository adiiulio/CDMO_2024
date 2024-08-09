# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
import math

from matplotlib import cm, pyplot as plt
from z3 import *
import numpy as np
import networkx as nx
import time
import json

# - - - - - - - - - - - - - - - - - - - - - CONFIGURATIONS - - - - - - - - - - - - - - - - - - - - - #
DEFAULT_MODEL = "maxDistanceObjModel"
DEFAULT_IMPLIED_CONS = "impliedMaxDistanceObjModel"
DEFAULT_SYMM_BREAK_CONS = "symmBreakDMaxDistanceObjModel"
DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS = "impliedAndSymmBreakMaxDistanceObjModel"

SECOND_OBJ_MODEL = "secondObjectiveModel"
SECOND_OBJ_IMPLIED_CONS = "impliedSecondObjectiveModel"
SECOND_OBJ_SYMM_BREAK_CONS = "symmBreakSecondObjectiveModel"
SECOND_OBJ_IMPLIED_AND_SYMM_BREAK_CONS = "impliedAndSymmBreakSecondObjectiveModel"


configurations = [DEFAULT_MODEL, DEFAULT_IMPLIED_CONS, DEFAULT_SYMM_BREAK_CONS, DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS,
                  SECOND_OBJ_MODEL, SECOND_OBJ_IMPLIED_CONS, SECOND_OBJ_SYMM_BREAK_CONS, SECOND_OBJ_IMPLIED_AND_SYMM_BREAK_CONS]


# - - - - - - - - - - - - - - - - - - - - - FUNCTIONS - - - - - - - - - - - - - - - - - - - - - #
def exactly_one(vars):
    return And(
        Or(vars),  # At least one is true
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))])))


def inputFile(num):
    # Instantiate variables from file
    if num < 10:
        instances_path = "instances/inst0" + str(num) + ".dat"  # inserire nome del file
    else:
        instances_path = "instances/inst" + str(num) + ".dat"  # inserire nome del file

    data_file = open(instances_path)
    lines = []
    for line in data_file:
        lines.append(line)
    data_file.close()
    n_couriers = int(lines[0].rstrip('\n'))
    n_items = int(lines[1].rstrip('\n'))
    max_load = list(map(int, lines[2].rstrip('\n').split()))
    size_item = list(map(int, lines[3].rstrip('\n').split()))
    for i in range(4, len(lines)):
        lines[i] = lines[i].rstrip('\n').split()

    for i in range(4, len(lines)):
        lines[i] = [lines[i][-1]] + lines[i]
        del lines[i][-1]

    dist = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])

    last_row = dist[-1]
    dist = np.insert(dist, 0, last_row, axis=0)
    dist = np.delete(dist, -1, 0)

    dist = dist.astype(int)

    return n_couriers, n_items, max_load, [0] + size_item, dist


def find_routes(routes, current_node, remaining_edges, current_route):
    if current_node == 0 and len(current_route) > 1:
        routes.append(list(current_route))
    else:
        for i in range(len(remaining_edges)):
            if remaining_edges[i][0] == current_node:
                next_node = remaining_edges[i][1]
                current_route.append(remaining_edges[i])
                find_routes(routes, next_node, remaining_edges[:i] + remaining_edges[i + 1:], current_route)
                current_route.pop()

    solution_route = []
    for i in range(len(routes)):
        temp_route = []
        for s in routes[i]:
            temp_route.append(s[0])
        temp_route = temp_route[1:]
        solution_route.append(temp_route)

    return solution_route


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


def print_loads(model, print_routes, max_l, loads, s_item):
    print("\n- - Print Loads - -")
    print("Size Items: ", s_item)
    for k in range(len(max_l)):
        print(f"{k} - Max Load: {max_l[k]}")
        print(print_routes[k])
        print(f"Total Load: {model.evaluate(loads[k])}\n")


# - - - - - - - - - - - - - - - - - - - - - MAIN - - - - - - - - - - - - - - - - - - - - - #

def find_model(instance_num, configuration, remaining_time=300, upper_bound=None):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)
    s = Solver()
    s.set("timeout", (int(remaining_time) * 1000))

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    # decision variables
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    courier_loads = [Int(f"courier_loads_{k}") for k in range(n_couriers)]

    u = [Int(f"u_{j}") for j in G.nodes]

    objective = Int('objective')
    lower_bound = 0
    for i in G.nodes:
        if all_distances[0, i] + all_distances[i, 0] > lower_bound: lower_bound = all_distances[0, i] + all_distances[i, 0]
    print(f'THE LOWER BOUND IS {lower_bound}')

    # - - - - - - - - - - - - - - - - CONSTRAINTS - - - - - - - - - - - - - - - - #

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
        s.add(courier_loads[k] == Sum([If(x[i][j][k], size_item[i], 0) for i, j in G.edges]))
        s.add(courier_loads[k] > 0)
        s.add(courier_loads[k] <= max_load[k])

    # - - - - - - - - - - - - - - - - IMPLIED CONSTRINTS & SIMMETRY BREAKING - - - - - - - - - - - - - - - - #

    # If (i, j) == True than --> for all the other k (i, j) != True
    if (configuration == DEFAULT_IMPLIED_CONS or configuration == DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS
            or configuration == SECOND_OBJ_IMPLIED_CONS or configuration == SECOND_OBJ_IMPLIED_AND_SYMM_BREAK_CONS):
        for i in range(n_items + 1):
            for j in range(n_items + 1):
                for k in range(n_couriers):
                    other_couriers = [k_prime for k_prime in range(n_couriers) if k_prime != k]
                    s.add(Implies(x[i][j][k], And([Not(x[i][j][k_prime]) for k_prime in other_couriers])))

        # For every courier, each row contains only one True
        for i in range(n_items + 1):
            for k in range(n_couriers):
                for j in range(n_items + 1):
                    other_destinations = [j_prime for j_prime in range(n_items + 1) if j_prime != j]
                    s.add(Implies(x[i][j][k], And([Not(x[i][j_prime][k]) for j_prime in other_destinations])))

    if (configuration == DEFAULT_SYMM_BREAK_CONS or configuration == DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS
            or configuration == SECOND_OBJ_SYMM_BREAK_CONS or configuration == SECOND_OBJ_IMPLIED_AND_SYMM_BREAK_CONS):
        for k1 in range(n_couriers):
            for k2 in range(n_couriers):
                if k1 != k2:
                    load_k1 = Sum([If(x[i][j][k1], size_item[i], 0) for i, j in G.edges])
                    load_k2 = Sum([If(x[i][j][k2], size_item[i], 0) for i, j in G.edges])
                    s.add(Implies(max_load[k1] < max_load[k2], load_k1 <= load_k2))

    # - - - - - - - - - - - - - - - - NO SUBTOUR PROBLEM - - - - - - - - - - - - - - - - #

    s.add(u[0] == 1)

    # all the other points must be visited after the depot
    for i in G.nodes:
        if i != 0:  # excluding the depot
            s.add(u[i] >= 2)

    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                s.add(x[i][j][z] * u[j] >= x[i][j][z] * (u[i] + 1))

    # - - - - - - - - - - - - - - - - SOLVING - - - - - - - - - - - - - - - - #

    total_distance = Sum(
        [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i, j in G.edges])

    min_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i, j in G.edges])

    max_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i, j in G.edges])

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(all_distances[i][j]), 0) for i, j in G.edges])
        min_distance = If(temp < min_distance, temp, min_distance)
        max_distance = If(temp > max_distance, temp, max_distance)

    if (configuration == SECOND_OBJ_MODEL or configuration == SECOND_OBJ_IMPLIED_CONS or configuration == SECOND_OBJ_SYMM_BREAK_CONS
            or configuration == SECOND_OBJ_IMPLIED_AND_SYMM_BREAK_CONS):
        # OBJECTIVE 1
        if upper_bound is None:
            s.add(objective == Sum(total_distance, (max_distance - min_distance)))
        else:
            s.add(objective == Sum(total_distance, (max_distance - min_distance)))
            s.add(upper_bound > objective)
    else:
        # OBJECTIVE 2
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

        tot_item = []
        for z in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][z])]
            items = []
            current = 0
            while len(tour_edges) > 0:
                for i, j in tour_edges:
                    if i == current:
                        items.append(j)
                        current = j
                        tour_edges.remove((i, j))
            tot_item.append([i for i in items if i != 0])

        new_objective = model.evaluate(objective)

        return elapsed_time, new_objective, tot_item, model.evaluate(total_distance), model.evaluate(max_distance), model.evaluate(min_distance)
    else:
        elapsed_time = time.time() - start_time
        return elapsed_time, -1, [], 0, 0, 0


def find_best(instance, config):
    best_obj, best_solution = -1, []
    run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist = find_model(instance, config, 300,None)
    remaining_time = 300 - run_time
    best_obj, best_solution, best_total_dist, best_max_dist, best_min_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist

    while remaining_time > 0:
        run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist = find_model(instance, config, remaining_time, temp_obj)
        remaining_time = remaining_time - run_time
        if temp_obj == -1:
            if (300 - remaining_time) >= 299:
                return 300, False, str(best_obj), best_solution, best_total_dist, best_max_dist, best_min_dist
            else:
                return int(300 - remaining_time), True, str(best_obj), best_solution, best_total_dist, best_max_dist, best_min_dist
        else:
            best_obj, best_solution, best_total_dist, best_max_dist, best_min_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist

    print("time limit exceeded")
    print("Remaining time: ", remaining_time)
    return 300, False, str(best_obj), best_solution, best_total_dist, best_max_dist, best_min_dist

inst = 0
configuration = 0

while inst < 1 or inst > 21:
    inst = int(input("Instance number: "))
    if inst < 1 or inst > 21:
        print(f"ERROR WITH FIRST ARGUMENT: Instance {inst} does not exist, please insert a number between 1 and 21")

while configuration < 1 or configuration > len(configurations):
    configuration = int(input("Configuration number: "))
    if configuration < 1 or configuration > len(configurations):
        print(f"ERROR WITH SECOND ARGUMENT: Configuration number {configuration} does not exist, please insert a number between 1 and {len(configurations)}")

configuration = configurations[configuration-1]

runtime, status, obj, solution, total_dist, max_dist, min_dist = find_best(inst, configuration)

print("\n#####################    OUTPUT   ######################")
print("Configuration: ", configuration)
if len(solution) == 0:
    print("Time taken: 300")
    print("Objective value: inf")
    print("Optimal solution not found")
    print("Solution: []")
else:
    print(f"Time taken: {runtime}")
    print(f"Objective value: {obj}")
    print("status: ", status)
    print(f"Solution: {solution}")

    print("------- Additional Information -------")
    print("Min path travelled: ", min_dist)
    print("Max path travelled: ", max_dist)
    print("Total path travelled: ", total_dist)

