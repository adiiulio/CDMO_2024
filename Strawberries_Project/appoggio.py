    if configuration in ["10"]:
        minTravelled = model.addVar(vtype=GRB.INTEGER, name="minTravelled")
        for z in range(n_couriers):
            model.addConstr(quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges) <= maxTravelled)
            model.addConstr(quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges) >= minTravelled)
        sumOfAllPaths = gp.LinExpr(
            quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i, j in G.edges))
        model.setObjective(sumOfAllPaths + (maxTravelled - minTravelled), GRB.MINIMIZE)

    if configuration in ["1", "2", "3", "4","5"]:
        lower_bound = 0
        upper_bound = 0
        max_dist = 0
        
        # Calcolo del lower bound
        for i in G.nodes:
            if all_distances[0, i] + all_distances[i, 0] > lower_bound: 
                lower_bound = all_distances[0, i] + all_distances[i, 0]
        
        # Calcolo del max_dist tra tutti i nodi
        for i, j in G.edges:
            if all_distances[i, j] > max_dist:
                max_dist = all_distances[i, j]
        
        # Calcolo del upper bound
        upper_bound = max_dist + max(all_distances[0, i] for i in G.nodes)
        
        # Vincoli
        for z in range(n_couriers):
            model.addConstr(quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges) <= maxTravelled)
        
        model.setObjective(maxTravelled, GRB.MINIMIZE)
        model.addConstr(maxTravelled >= lower_bound)
        model.addConstr(maxTravelled <= upper_bound)



    #NO SUBTOUR PROBLEM------------------------

    # sub-tour elimination constraints
    # the depot is always the first point visited
    if configuration == "5":
        # Definizione delle variabili di flusso
        f = [[[model.addVar(vtype=GRB.INTEGER, name=f'f_{k}_{i}_{j}') for j in G.nodes] for i in G.nodes] for k in range(n_couriers)]

        # Vincoli sul flusso
        for k in range(n_couriers):
            for i in G.nodes:
                for j in G.nodes:
                    if i != j:
                        model.addConstr(f[k][i][j] <= max_load[k] * x[k][i, j])

        # Conservazione del flusso
        for k in range(n_couriers):
            for j in G.nodes:
                if j != 0:
                    model.addConstr(
                        quicksum(f[k][i][j] for i in G.nodes if i != j) == quicksum(f[k][j][i] for i in G.nodes if i != j)
                    )

        # Vincoli sul deposito
        for k in range(n_couriers):
            model.addConstr(
                quicksum(f[k][0][j] for j in G.nodes if j != 0) == quicksum(size_item[j] * x[k][0, j] for j in G.nodes if j != 0)
            )
            model.addConstr(
                quicksum(f[k][j][0] for j in G.nodes if j != 0) == quicksum(size_item[j] * x[k][j, 0] for j in G.nodes if j != 0)
            )
        else:
            # MTZ approach core
            for z in range(n_couriers):
                for i, j in G.edges:
                    if i != 0 and j != 0 and i != j:  # excluding the depot
                        model.addConstr(x[z][i, j] * u[j] >= x[z][i, j] * (u[i] + 1))


    #------IMPLIED CONSTRAINTS------------------
    if configuration in ["2","4"]:
        #the total load of all vehicles doesn't exceed the sum of vehicles capacities
        total_load = quicksum(size_item[j] * x[z][i, j] for z in range(n_couriers) for i, j in G.edges)
        max_total_load = sum(max_load)
        model.addConstr(total_load <= max_total_load)

        # all the other points must be visited after the depot
        for i in G.nodes:
            if i != 0:  # excluding the depot
                model.addConstr(u[i] >= 2)
  
    #------SYMMETRY BREAKING CONSTRAINTS-----------
    # symmetry breaking couriers (added only if the configuration allows it)
    if configuration in ["3","4"]:
        #couriers with lower max_load must bring less weight
        for z1 in range(n_couriers):
            for z2 in range(n_couriers):
                if max_load[z1] > max_load[z2]:
                    model.addConstr(quicksum(size_item[j] * x[z1][i, j] for i, j in G.edges) >= quicksum(size_item[j] * x[z2][i, j] for i, j in G.edges))

    # symmetry breaking couriers based on the length of the paths GPTTT
    if configuration in ["3","4"]:
        for z in range(n_couriers - 1):
            model.addConstr(quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges) <= quicksum(all_distances[i, j] * x[z + 1][i, j] for i, j in G.edges))

    # symmetry breaking couriers based on max load capacity GPTTT
    if configuration in ["3","4"]:
        for z in range(n_couriers - 1):
            model.addConstr(max_load[z] >= max_load[z + 1], name=f"symmBreak_maxLoad_{z}")

    #------DEFAULT CONSTRAINTS-----------
    # Every item must be delivered
    # (each 3-dimensional column must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            model.addConstr(quicksum(x[z][i, j] for z in range(n_couriers) for i in G.nodes if i != j) == 1)

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for z in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[z][i, j] - x[z][j, i] for j in G.nodes if i != j) == 0)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for z in range(n_couriers):
        model.addConstr(quicksum(x[z][i, 0] for i in G.nodes if i != 0) == 1)
        model.addConstr(quicksum(x[z][0, j] for j in G.nodes if j != 0) == 1)

    # each courier does not exceed its max_load
    # sum of size_items must be minor than max_load for each courier
    for z in range(n_couriers):
        model.addConstr(quicksum(size_item[j] * x[z][i, j] for i, j in G.edges) <= max_load[z])