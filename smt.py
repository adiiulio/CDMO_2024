from z3 import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import json

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
    distances_matrix = [list(map(int, line.rstrip('\n').split())) for line in variables[4:]]

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
        Not(Or([And(vars[i], vars[j]) for i in range(len(vars)) for j in range(i+1, len(vars))]))
    )

#function that finds the model
def find_model(instance, config):

    #check if what you input is correct 
    if instance < 1 or instance > 21:
        print(f"ERROR: Instance {instance} doesn't exist. Please insert a number between 1 and 21")
        return
    if config<1 or config>4:
        print(f"ERROR: Configuration {config} doesn't exist. Please insert a number between 1 and 4")
        return

    #read the instance
    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

    #define the solver and the max time, and set the timeout
    max_time = 300
    s = Optimize()
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

    nodes = [Int(f'n_{j}') for j in G.nodes]

    #Constraints -------------------------------------------------------------------------------------------------------------------------------

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

    #FIXME THE SYMMETRY BREAKING CONSTRAINTS AND THE IMPLIED CONSTRAINTS DO NOT WORK!!!!!!

    #SYMMETRY BREAKING CONSTRAINTS--------------------

    #lexicographical order for couriers
    #for c in range(n_couriers-1):
    #    for j in G.nodes:
    #        if j != 0:
    #            const_10 = u[k, j] <= u[k + 1, j]
    #            s.add(const_10)



    #------IMPLIED CONSTRAINTS------------------
    #the total load of all vehicles doesn't exceed the sum of vehicles capacities
    #const_11 = Sum(Sum(sizes[j] * x[k][i, j] for i, j in G.edges) for k in range(n_couriers)) <= Sum(max_loads)
    #s.add(const_11)

    #all nodes must be visited after depot
    #eps=0.0005
    #for j in G.nodes:
    #    if j != 0:
    #        const_12 = u[k, 0] + eps <= u[k, j]
    #1
    #         s.add(const_12)

    # OBJECTIVE FUNCTION

    #total distance travelled
    total_distance = Sum([If(x[i][j][k], G.edges[i, j]['length'], 0)
                        for k in range(n_couriers)
                        for i in G.nodes
                        for j in G.nodes
                        if i != j])

    #minimize the total distance
    minimization = s.minimize(total_distance)

    start_time = time.time()
    #check if satisfiable
    result = {}
    if s.check() == sat:
        elapsed_time = time.time() - start_time
        print(f'The total time elapsed is {elapsed_time}')
        model = s.model()
        total_distance_value = model.evaluate(total_distance)
        print("Total distance of the path found:", total_distance_value)

        paths = []
    
        for courier in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][courier])]
            print(f'the path for courier {courier} is {tour_edges}')
            found = []
            for (i, j) in tour_edges:
                if i not in found:
                    found.append(i)
                if j not in found:
                    found.append(j)
            paths.append(found)
        print(f'The solution found is {paths}')

        lower_bound = s.lower(minimization)
        upper_bound = s.upper(minimization)
        is_optimal = 'false'
        if lower_bound == upper_bound:
            is_optimal = 'true'
        print(f'The solution is optimal? {is_optimal}')

        inst = {}
        config = {}
        config['Time'] = elapsed_time
        config['Distance'] = int(total_distance_value.as_long())
        config['Solution'] = paths
        config['IsOptimal'] = is_optimal
        

        inst[1] = config

        with open(f"results_folder/{instance}.JSON", "w") as file:
            file.write(json.dumps(inst, indent=3))

        return total_distance, elapsed_time, tour_edges

    else:
        print("No solution found.")

    with open(f"results_folder/{instance}.JSON", "w") as file:
        file.write(json.dumps(result, indent=3))

# Main function call
find_model(5, 1)


#---------------main----------------
for inst in range(1, 21):
    find_model(inst, 1)