import json
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx
import matplotlib.pyplot as plt

#1. Normal, no implied constraints and no symmetry breaking
#2. With implied constraints
#3. With symmetry breaking
#4. With both implied and symmetry breaking
#5. flow based subtour elimination instead of the normal one

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
        file_path = "CDMO_2024/Strawberries_Project/instances/inst0" + str(number) + ".dat"  # inserire nome del file
    else:
        file_path = "CDMO_2024/Strawberries_Project/instances/inst" + str(number) + ".dat"  # inserire nome del file
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

def calculate_lower_bound(G, all_distances):
    """
    Calculate the lower bound for the minimum distance a courier needs to travel.
    
    Parameters:
    G (nx.DiGraph): The graph representing the delivery network.
    all_distances (np.ndarray): The distance matrix where the value at [i, j] represents the distance from node i to node j.

    Returns:
    int: The calculated lower bound.
    """
    lower_bound = 0
    for i in G.nodes:
        # Calculate the round trip distance from depot (node 0) to node i and back
        round_trip_distance = all_distances[0, i] + all_distances[i, 0]
        # Update lower_bound if this round trip distance is greater than the current lower_bound
        if round_trip_distance > lower_bound:
            lower_bound = round_trip_distance

    return lower_bound

def find_model(instance, config, remaining_time, upper_bound=None):

    #check if what you input is correct 
    if instance < 1 or instance > 21:
        print(f"ERROR: Instance {instance} doesn't exist. Please insert a number between 1 and 21")
        return
    if config<1 or config>4:
        print(f"ERROR: Configuration {config} doesn't exist. Please insert a number between 1 and 4")
        return
    
    #read the instance
    n_couriers, n_items, max_loads, sizes, distances = read_instance(instance)

    #define the model and the timeout
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('TimeLimit', 300)

    G=createGraph(distances)

    #create the decision variables

    #courier load for courier k
    courier_loads = [model.addVar(vtype=GRB.INTEGER, name=f'Courier_load_for_{k}') for k in range(n_couriers)]

    # x[k][e] means that the the route e (edge (i,j)) is used by courier k
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]

    # this will be used for the MZT approach
    u = [model.addVar(vtype=GRB.INTEGER, name=f'u_{j}') for j in G.nodes]

    lower_bound=calculate_lower_bound(G, distances)

    #Constraints -------------------------------------------------------------------------------------------------------------------------------

    #No routes from any node to itself
    for k in range(n_couriers):
        model.addConstr((x[k][i, i] == 0 for i in G.nodes))
    
    #Every item must be delivered
    for j in G.nodes:
        if j != 0:  # no depot
            model.addConstr(quicksum(x[z][i, j] for z in range(n_couriers) for i in G.nodes if i != j) == 1)

    # Every node should be entered and left once and by the same vehicle.








#################Prove di funzionamento#################

n_couriers, n_items, max_loads, sizes, distances = read_instance(3)
print(n_couriers)
print('\n')
print(n_items)
print('\n')
print(max_loads)
print('\n')
print(sizes)
print('\n')
print(distances)
print('\n')
G = createGraph(distances)

pos = nx.spring_layout(G)

# Disegnare i nodi
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

# Disegnare le etichette dei nodi
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
plt.title("Graph Representation of Courier Routes")
plt.show()
