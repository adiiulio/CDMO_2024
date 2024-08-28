import json
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

##################################    POSSIBLE CONFIGURATIONS OF THE MODEL      ##################################
#1. Normal, no implied constraints and no symmetry breaking
#2. With implied constraints
#3. With symmetry breaking
#4. With both implied and symmetry breaking
#5. flow based subtour elimination instead of the normal one

configurations=["1","2","3","4","5"]
##################################################################################################################

def transform_distance_matrix(lines):
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
    return dist

#function that reads the instance
def inputFile(num):
    # Instantiate variables from file
    if num < 10:
        instances_path = "CDMO_2024/Strawberries_Project/instances/inst0" + str(num) + ".dat" 
    else:
        instances_path = "CDMO_2024/Strawberries_Project/instances/instances/inst" + str(num) + ".dat"

    data_file = open(instances_path)
    lines = []
    for line in data_file:
        lines.append(line)
    data_file.close()
    #read the number of couriers
    n_couriers = int(lines[0].rstrip('\n'))
    #read the number of items
    n_items = int(lines[1].rstrip('\n'))
    #read the max load for each courier
    max_load = list(map(int, lines[2].rstrip('\n').split()))
    #read the size of each package
    size_item = list(map(int, lines[3].rstrip('\n').split()))
    #read the distances
    dist=transform_distance_matrix(lines)

    # Print general informations about the problem instance
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("")

    return n_couriers, n_items, max_load, [0] + size_item, dist

#function that creates the graph
def createGraph(all_distances):
    all_dist_size = all_distances.shape[0]
    size_item = all_distances.shape[0] - 1
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(all_dist_size))

    # Add double connections between nodes
    for i in range(all_dist_size):
        for j in range(i + 1, size_item + 1):  # size item + 1 because we include also the depot in the graph
            G.add_edge(i, j)
            G.add_edge(j, i)

    # Assign edge lengths
    lengths = {(i, j): all_distances[i][j] for i, j in G.edges()}
    nx.set_edge_attributes(G, lengths, 'length')

    return G


def find_model(num,configuration,json_bool:bool):
    #boolean variable: if true run on a sigle instance, if false run on all instances
    if not json_bool:
        num = input("Instance number: ")
        configuration = input("Configuration number: ")

    num=int(num)
    configuration=int(configuration)

    if num < 1 or num > 21:
        print(f"ERROR WITH FIRST ARGUMENT: Instance {num} doesn't exist, please insert a number between 1 and 21")
        return

    if configuration < 1 or configuration > len(configurations):
        print(f"ERROR WITH SECOND ARGUMENT: Configuration number {configuration} doesn't exist, please insert a number between 1 and {len(configurations)}")
        return
    else:
        configuration = configurations[configuration-1]

    # read input file
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(num)

    # model: set initial parameters
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('TimeLimit', 300)

    #create the graph containing all possible paths
    G = createGraph(all_distances)
    print(G)

    # decision variables
    #x is a list of n_couriers variables, each representing a binary decision variable for each edge in the graph G. Specifically, x[i][e] represents whether edge e is used by courier i or not, 1 if yes, 0 if not.
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]
    #u is a variable that represents the number of items delivered to each node in the graph. In other words, u determines how many items are delivered to each node by the couriers. ub=n_items sets an upper bound of n_items on the number of items that can be delivered to each node.
    u = model.addVars(G.nodes, vtype=GRB.INTEGER, ub=n_items)

    # objective function
    maxTravelled = model.addVar(vtype=GRB.INTEGER, name="maxTravelled")

    if configuration in ["1", "2", "3", "4","5"]:
        lower_bound = 0
        for i in G.nodes:
            if all_distances[0, i] + all_distances[i, 0] > lower_bound: lower_bound = all_distances[0, i] + all_distances[i, 0]
        for k in range(n_couriers):
            model.addConstr(quicksum(all_distances[i, j] * x[k][i, j] for i, j in G.edges) <= maxTravelled)
        model.setObjective(maxTravelled, GRB.MINIMIZE)
        model.addConstr(maxTravelled >= lower_bound)


    ################################## CONSTRAINTS ####################################

    ### IMPLIED CONSTRAINTS
    if configuration in ["2","4"]:
        # all the other points must be visited after the depot
        for i in G.nodes:
            if i != 0:  # excluding the depot
                model.addConstr(u[i] >= 2)
        
        #the total load of all vehicles doesn't exceed the sum of vehicles capacities
        total_load = quicksum(size_item[j] * x[k][i, j] for k in range(n_couriers) for i, j in G.edges)
        max_total_load = sum(max_load)
        model.addConstr(total_load <= max_total_load)

    ### SYMMETRY BREAKING CONSTRAINTS
    if configuration in ["3","4"]:
        # Compare the total distance traveled by courier k and courier k+1
        for k in range(n_couriers - 1):
            model.addConstr(
                quicksum(all_distances[i, j] * x[k][i, j] for i, j in G.edges) <= quicksum(all_distances[i, j] * x[k+1][i, j] for i, j in G.edges)
            )

    ### DEFAULT CONSTRAINTS
    # Every item must be delivered
    # (each 3-dimensional column must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            model.addConstr(quicksum(x[k][i, j] for k in range(n_couriers) for i in G.nodes if i != j) == 1)

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for k in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[k][i, j] - x[k][j, i] for j in G.nodes if i != j) == 0)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for k in range(n_couriers):
        model.addConstr(quicksum(x[k][i, 0] for i in G.nodes if i != 0) == 1)
        model.addConstr(quicksum(x[k][0, j] for j in G.nodes if j != 0) == 1)

    # each courier does not exceed its max_load
    # sum of size_items must be minor than max_load for each courier
    for k in range(n_couriers):
        model.addConstr(quicksum(size_item[j] * x[k][i, j] for i, j in G.edges) <= max_load[k])

    # sub-tour elimination constraints
    # the depot is always the first point visited

    if(configuration == "5"):
        f = [model.addVars(G.edges, vtype=GRB.CONTINUOUS, name=f"flow_{k}") for k in range(n_couriers)]
        # Flow conservation constraints: Ensure that the flow is conserved at each node. The total flow into a node minus the flow out should equal the demand at that node.
        for k in range(n_couriers):
            for i in G.nodes:
                if i != 0:
                    model.addConstr(quicksum(f[k][j, i] for j in G.nodes if (j, i) in G.edges) - quicksum(f[k][i, j] for j in G.nodes if (i, j) in G.edges) == size_item[i] * quicksum(x[k][i, j] for j in G.nodes if (i, j) in G.edges))

        # Depot flow constraints: Set the flow at the depot (node 0) to be equal to the total size of items that the courier must carry.
        for k in range(n_couriers):
            model.addConstr(quicksum(f[k][0, j] for j in G.nodes if (0, j) in G.edges) == quicksum(size_item[j] * x[k][i, j] for i, j in G.edges))

        # Capacity constraints for flow: Ensure that the flow along each edge does not exceed the capacity of that edge, which is bound by whether the edge is used or not.
        for k in range(n_couriers):
            for i, j in G.edges:
                model.addConstr(f[k][i, j] <= max_load[k] * x[k][i, j])
    else:
        # MTZ approach core
        model.addConstr(u[0] == 1)
        for k in range(n_couriers):
            for i, j in G.edges:
                if i != 0 and j != 0 and i != j:  # excluding the depot
                    model.addConstr(x[k][i, j] * u[j] >= x[k][i, j] * (u[i] + 1))

    
    model.update()
    print("The NUMBER OF CONSTRAINTS **** IS {}".format(model.NumConstrs))
    model.optimize()

    if model.SolCount == 0:
        return 300, False, "Inf", []

    # print information about solving process
    print("\n#####################    OUTPUT   ######################")
    print("Configuration: ", configuration)
    if model.SolCount == 0:
        print("Time taken: 300")
        print("Objective value: inf")
        print("Optimal solution not found")
        print("Solution: []")

    else:
        objectiveVal = max([sum(all_distances[i, j] * x[k][i, j].x for i, j in G.edges) for k in range(n_couriers)])

        print("Runtime: ", model.Runtime)
        print("Objective value: ", objectiveVal)
        if (model.status == GRB.OPTIMAL):
            print("Status: Optimal solution found")
        else:
            print("Status: Optimal solution not found")

        tot_item = []
        for k in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if x[k][i, j].x >= 1]
            items = []
            current = 0
            while len(tour_edges) > 0:
                for i, j in tour_edges:
                    if i == current:
                        items.append(j)
                        current = j
                        tour_edges.remove((i, j))
            tot_item.append([i for i in items if i != 0])
        print("Solution: ", tot_item)

        print("\n-------- Additional Information --------")
        print("Min path travelled: ",
              min([sum(all_distances[i, j] * x[k][i, j].x for i, j in G.edges) for k in range(n_couriers)]))
        print("Max path travelled: ",
              max([sum(all_distances[i, j] * x[k][i, j].x for i, j in G.edges) for k in range(n_couriers)]))
        print("Total path travelled: ",
              sum([sum(all_distances[i, j] * x[k][i, j].x for i, j in G.edges) for k in range(n_couriers)]))

    print("############################################################################### \n")
    return int(model.Runtime), model.status == GRB.OPTIMAL, objectiveVal, tot_item

def main(json_bool:bool=True):
    # number of instances over which iterate
    if(json_bool):
        n_istances = 21

        for instance in range(n_istances):
            inst = {}
            count = 1
            for configuration in configurations:
                print(f"\n\n\n###################    Instance {instance + 1}/{n_istances}, Configuration {count} out of {len(configurations)} -> {configuration}    ####################")
                runTime, status, obj, solution = find_model(instance + 1, configuration, True)

                # JSON
                config = {}
                config["time"] = runTime
                config["optimal"] = status
                config["obj"] = obj
                config["solution"] = solution

                inst[configuration] = config
                count += 1

            with open(f"CDMO_2024/Strawberries_Project/res/{instance + 1}.JSON", "w") as file:
                file.write(json.dumps(inst, indent=3))
    else:
        find_model(0,0,json_bool)


main(json_bool=True)