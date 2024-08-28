import json
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx
import matplotlib.pyplot as plt

############## CONFIG ##############
DEFAULT_MODEL = "second_Model"
DEFAULT_IMPLIED_CONS = "second_Model_implied_Cons"
DEFAULT_SYMM_BREAK_CONS = "second_Model_symmBreak"
DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS = "second_Model_implied_And_SymBreak"

configurations=[DEFAULT_MODEL,DEFAULT_IMPLIED_CONS,DEFAULT_SYMM_BREAK_CONS,DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS]
impliedConfiguration = [DEFAULT_IMPLIED_CONS, DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS]
symmBreakConfiguration = [DEFAULT_SYMM_BREAK_CONS, DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS]
####################################



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
    distances_matrix = [list(map(int, line.rstrip('\n').split())) for line in variables[4:]]

    return n_couriers, n_items, max_loads, [0] + sizes, distances_matrix


#this is the create graph from the github code, copied and pasted just to try tu understand
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

def drawGraph(G, all_distances, x, n_couriers):
    pos = nx.spring_layout(G)
    
    # Disegnare i nodi
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Disegnare le etichette dei nodi
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Disegnare gli archi con diverse colorazioni per ciascun corriere
    edge_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for z in range(n_couriers):
        edges = [(i, j) for i, j in G.edges if x[z][i, j].x >= 1]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[z % len(edge_colors)], width=2.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): all_distances[i, j] for (i, j) in edges}, font_color='red')

    # Disegnare gli archi rimanenti in grigio
    remaining_edges = [(i, j) for i, j in G.edges if not any(x[z][i, j].x >= 1 for z in range(n_couriers))]
    nx.draw_networkx_edges(G, pos, edgelist=remaining_edges, edge_color='gray', style='dashed')

    plt.title("Graph Representation of Courier Routes")
    plt.show()



def main():
    inst_number=int(input("Instance number: "))
    conf_number=int(input("Configuration number: "))

    #Check correctness of the input
    if inst_number < 1 or inst_number > 21:
        print(f"ERROR: Instance {inst_number} doesn't exist. Please insert a number between 1 and 21")
        return 
    
    if conf_number < 1 or conf_number > len(configurations):
        print(f"ERROR: Configuration {conf_number} doesn't exist. Please insert a number between 1 and {len(configurations)}")
        return
    else:
        configuration=configurations[conf_number-1]

    #Reading from instance file
    n_couriers, n_items, max_loads , sizes, distances = read_instance(inst_number)

    #Setting initial parameters of the model
    env=gp.Env(empty=True)
    env.setParam('OutputFlag',0)
    env.start()
    model=gp.Model(env=env)
    model.setParam('TimeLimit',300)

    #Creating the graph 
    G=createGraph(distances)
    print(G)
    #x is a list of n_couriers variables, each representing a binary decision variable for each edge in the graph G. Specifically, x[i][e] represents whether edge e is used by courier i or not, 1 if yes, 0 if not.
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]
    #u is a variable that represents the number of items delivered to each node in the graph. In other words, u determines how many items are delivered to each node by the couriers. ub=n_items sets an upper bound of n_items on the number of items that can be delivered to each node.
    #u = model.addVars(G.nodes, vtype=GRB.INTEGER, ub=n_items)
    # u is a variable that represents the order in which nodes are visited in the tour of each courier.
    u = model.addVars(n_couriers, G.nodes, vtype=GRB.INTEGER, lb=0, ub=n_items)

    ######### DEFAULT CONSTRAINTS #########

    #Each courier leaves and enters the depot exactly once
    o=0
    for k in range(n_couriers):
        model.addConstr(quicksum(x[k] [o,j] for j in G.nodes if j != 0) == 1)
        model.addConstr(quicksum(x[k] [i,o] for i in G.nodes if i != 0) == 1)

    #Each courier load must not exceed its max_load
    for k in range(n_couriers):
        model.addConstr(quicksum(sizes[j] * x[k][i, j] for i, j in G.edges) <= max_loads[k])

    #Each node is visited at most once by the same couries #MAYBE REDUNDANT WITH THE NEXT CONSTRAINT
    for k in range(n_couriers):
        for j in G.nodes:
            model.addConstr(quicksum(x[k][i, j] for i in G.nodes if i != j) <= 1)
    
    # Every node should be entered and left once and by the same vehicle FLOW CORRECTNESS
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for k in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[k][i, j] - x[k][j, i] for j in G.nodes if i != j) == 0)
   
    #No route from a node to itself. THERE IS NO (i,i) ARC IN THE GRAPH
    '''
    for k in range(n_couriers):
        model.addConstr(quicksum(x[k][i,i] for i in G.nodes if i != 0)==0)
    '''
    #Every item must be delivered
    for j in G.nodes:
        if j != 0:  # no depot
            model.addConstr(quicksum(x[k][i, j] for k in range(n_couriers) for i in G.nodes if i != j) == 1)
   
    # Vincoli di eliminazione dei sotto-cicli DFJ: TOO MANY CONSTRAINTS (example: instance 10 82k constraints vs 343 MTZ)
    '''
    for subset_size in range(2, n_items):
        for S in combinations(G.nodes, subset_size):
            if o not in S:  # Ensure depot is not in the subset
                for k in range(n_couriers):
                    model.addConstr(quicksum(x[k][i, j] for i in S for j in G.nodes if j not in S) >= 2)
    '''
    '''
    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                model.addConstr(x[z][i, j] * u[j] >= x[z][i, j] * (u[i] + 1))
    '''
    #MTZ sub-tour elimination constraints NOT SURE ABOUT THE CORRECTNESS
    #I don't understand why there is this explosion in constraints with this MZT implementation. +1.5k constraints with instance 10
    for k in range(n_couriers):
        for i, j in G.edges:
            if i != j and i != 0 and j != 0:
                model.addConstr(u[k, i] - u[k, j] + (n_items * x[k][i, j]) <= n_items - 1)
    
    #Depot is always the starting point for each courier
    for k in range(n_couriers):
        model.addConstr(u[k, 0] == 0)

    #Depot is always the ending point for each courier (Warning: it doesn't impose to stop there, just to come back there)
    for k in range(n_couriers):
        model.addConstr(quicksum(x[k][i, o] for i in G.nodes if i != 0) == 1)

    ######### SYMMETRY BREAKING CONSTRAINTS #########
    if configuration in symmBreakConfiguration:
        #Lexicographical Order Constraints (symmetry breaking)
        for k in range(n_couriers - 1):
            for j in G.nodes:
                if j != 0:
                    model.addConstr(u[k, j] <= u[k + 1, j])

    ######### IMPLIED CONSTRAINTS #########
    if configuration in impliedConfiguration:
        #Ensure that the total load of all vehicles does not exceed the sum of the vehicles' capacities.
        model.addConstr(sum(quicksum(sizes[j] * x[k][i, j] for i, j in G.edges) for k in range(n_couriers)) <= sum(max_loads))

        #All the other points come after the depot (implied?)
        eps=0.0005
        for j in G.nodes:
            if j != 0:
                model.addConstr(u[k, 0] + eps <= u[k, j])
            
    model.update()
    print(model.NumConstrs)
    ######## OBJECTIVE ########


main()