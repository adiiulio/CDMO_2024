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

#function that reads the instance
def read_instance(number):
    if number < 10:
        file_path = "instances/inst0" + str(number) + ".dat"  # inserire nome del file
    else:
        file_path = "instances/inst" + str(number) + ".dat"  # inserire nome del file
    
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
    #read the number of couriers
    n_couriers = int(lines[0].rstrip('\n'))
    #read the number of items
    n_items = int(lines[1].rstrip('\n'))
    #read the max load for each courier
    max_loads = list(map(int, lines[2].rstrip('\n').split()))
    #read the size of each package
    sizes = list(map(int, lines[3].rstrip('\n').split()))
    #read the distances
    
    for i in range(4, len(lines)):
        lines[i] = lines[i].rstrip('\n').split()

    for i in range(4, len(lines)):
        lines[i] = [lines[i][-1]] + lines[i]
        del lines[i][-1]

    distances_matrix = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])
    last_row = distances_matrix[-1]
    distances_matrix = np.insert(distances_matrix, 0, last_row, axis=0)
    distances_matrix = np.delete(distances_matrix, -1, 0)

    distances_matrix = distances_matrix.astype(int)
    return n_couriers, n_items, max_loads, sizes, distances_matrix

#function that creates the graph
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

#exactly one function
def exactly_one(vars):
    return And(
        Or(vars),  # At least one is true
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))])))

#function that finds the model

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

def find_model(instance, config):

    #check if what you input is correct 
    

    #read the instance
    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

    #define the solver and the max time, and set the timeout
    max_time = 300
    s = Solver()
    s.set("timeout", (int(max_time) * 1000))

    #create the graph containing all possible paths
    G = createGraph(distances)

    #create the decision variables

    #courier load for courier k
    cour_load = [Int(f'Courier_load_for_{k}') for k in range(n_couriers)] 

    #x[i][j][k] = True means that the route from i to j is used by courier k
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
            range(n_items + 1)]

    u = [Int(f"u_{j}") for j in G.nodes]

    objective = Int('objective')
    lower_bound = 0
    for i in G.nodes:
        if distances[0][i] + distances[0][i] > lower_bound: lower_bound = distances[0][i] + distances[0][i]



    #Constraints -------------------------------------------------------------------------------------------------------------------------------

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    #the load must be smaller than the max_loads and greater than 0
    for k in range(n_couriers):
        const_1 = cour_load[k]>0
        const_2 = cour_load[k]<max_loads[k]
        s.add(const_1)
        s.add(const_2)

    #each courier must start and end at origin point o
    o = 0
    for k in range(n_couriers):
        #I used sum because the amount of x[o][j][k] (or the other one) that needs to be True (aka 1) is exactly one
        #therefore if I sum all of those, the sum needs to be 1, as they are all 0 and only one is 1
        const_3 = Sum([x[o][j][k] for j in G.nodes if j != 0]) == 1
        const_4 = Sum([x[i][o][k] for i in G.nodes if i != 0]) == 1
        s.add(const_3)
        s.add(const_4)

    #each node is visited at most once by the same courier
    for k in range(n_couriers):
        #I consider both the "normal" points and the origin
        for j in G.nodes:
            const_5 = Sum([x[i][j][k] for i in G.nodes if i != j]) <= 1
            s.add(const_5)

    #each node must be entered and left once by the same courier
    #meaning that the amount of times a courier enters the node is the exact same amount as it leaves the node
    for k in range(n_couriers):
        #I consider both the "normal" points and the origin
        for j in G.nodes:
            a = Sum([x[i][j][k] for i in G.nodes if i != j]) 
            b = Sum([x[j][i][k] for i in G.nodes if i != j]) 
            const_6 = (a==b)
            s.add(const_6)

    #there is no route from a node to itself
    for k in range(n_couriers):
        const_7 = Sum([x[i][i][k] for i in G.nodes]) == 0
        s.add(const_7)

    #every item needs to be delivered by whichever courier
    for j in G.nodes:
            if j != o:  # the origin point o is not considered
                #this means that every 3d column of the output must have only one truth value = True
                const_8 = exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j])
                s.add(const_8)

    #constraint of the upper bounds of items a courier can bring
    for k in range(n_couriers):
        m = max_loads[k]/np.min(sizes)
        const_9 = cour_load[k]<=m
        s.add(const_9)

    #SYMMETRY BREAKING CONSTRAINTS--------------------

    for c in range(n_couriers-1):
        for j in G.nodes:
            if j != 0:
                const_10 = u[c] <= u[c + 1]
                if config == 3 or config == 4:
                    s.add(const_10)



    #------IMPLIED CONSTRAINTS------------------
    #the total load of all vehicles doesn't exceed the sum of vehicles capacities
    const_11 = Sum([cour_load[k] for k in range(n_couriers)]) <= Sum(max_loads)
    if config == 2 or config == 4:
        s.add(const_11)

    #all nodes must be visited after depot
    eps = 0.0005
    for j in G.nodes:
        if j != 0:
            const_12 = u[j] > eps
            if config == 2 or config == 4:
                s.add(const_12)

    # - - - - - - - - - - - - - - - - NO SUBTOUR PROBLEM - - - - - - - - - - - - - - - - #

    #s.add(u[0] == 1)

    # all the other points must be visited after the depot
    for i in G.nodes:
        if i != 0:  # excluding the depot
            s.add(u[i] >= 2)

    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                s.add(x[i][j][z] * u[j] >= x[i][j][z] * (u[i] + 1))

    # OBJECTIVE FUNCTION

    #total distance travelled
    total_distance = Sum(
        [If(x[i][j][k], int(distances[i][j]), 0) for k in range(n_couriers) for i, j in G.edges])
    min_distance = Sum(
        [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])

    max_distance = Sum(
        [If(x[i][j][0], int(distances[i][j]), 0) for i, j in G.edges])

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(distances[i][j]), 0) for i, j in G.edges])
        min_distance = If(temp < min_distance, temp, min_distance)
        max_distance = If(temp > max_distance, temp, max_distance)

    s.add(objective == Sum(total_distance, (max_distance - min_distance)))

    #minimize the total distance
    #minimization = s.minimize(total_distance)

    start_time = time.time()
    #check if satisfiable
    
    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()
        total_distance_value = model.evaluate(total_distance)

        paths = []
        tot_item = []
        for courier in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
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
        print("No solution found.")
        return None, None, None, None

def find_best(instance, config):
    best_obj, best_solution = -1, []
    run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist = find_model(instance, config)
    remaining_time = 300 - run_time
    best_obj, best_solution, best_total_dist, best_max_dist, best_min_dist = temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist
    while remaining_time > 0:
        run_time, temp_obj, temp_solution, temp_total_dist, temp_max_dist, temp_min_dist = find_model(instance, config)
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



#---------------main----------------
for instance in range(1, 22):
    inst = {}
    count = 1
    for config in range(1, 5):
        runtime, status, obj, solution, total_dist, max_dist, min_dist = find_best(instance, config)
        result = {}
        result['Time'] = runtime
        result['Distance'] = int(total_dist.as_long())
        result['Solution'] = solution
        #result['IsOptimal'] = is_optimal
        

        inst[config] = result
        count += 1
    with open(f"results_folder/{instance}.JSON", "w") as file:
        file.write(json.dumps(inst, indent=3))
    print(f'JSON {instance} updated')
