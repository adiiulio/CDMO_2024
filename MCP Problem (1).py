import numpy as np
import datetime
import json
import os
from minizinc import Instance, Model, Solver
    
def read_instance(filename):
    with open(filename, "r") as file:
        n_couriers = int(file.readline().strip())
        items_count = int(file.readline().strip())
        max_load = list(map(int, file.readline().strip().split()))
        size_item = list(map(int, file.readline().strip().split()))
        all_distances_matrix = []
        
        # Flatten the 2D array
        for _ in range(items_count + 1):
            all_distances_matrix.extend(list(map(int, file.readline().strip().split())))

        # Reshape the flat list into a 2D array
        matrix_size = int(len(all_distances_matrix) ** 0.5)  # Calculate the size of the matrix
        all_distances_matrix = np.reshape(all_distances_matrix, (matrix_size, matrix_size)).tolist()

        # Flatten the 2D list again for MiniZinc array2d format
        all_distances = [item for sublist in all_distances_matrix for item in sublist]
        print(all_distances)

    return n_couriers, items_count, max_load, size_item, matrix_size, all_distances

   

def solve_mcp(n_couriers, items_count, max_load, size_item, matrix_size, all_distances):
    # Load MiniZinc model
    mcp_model = Model("C:\MCP\MCP_Example_test_2.mzn")
    solver = Solver.lookup("gecode")

    # Create an instance of the model
    instance = Instance(solver, mcp_model)

    # Assign data to model parameters
    instance["n_couriers"] = n_couriers  # m in the original code
    instance["items_count"] = items_count  # n in the original code
    instance["max_load"] = max_load  # l in the original code
    instance["size_item"] = size_item  # s in the original code
    # Specify the matrix size for MiniZinc to interpret the flattened list correctly
    instance["matrix_size"] = matrix_size    
    instance["all_distances"] = all_distances  # D in the original code

    # Debug output: Print the data being passed to MiniZinc
    print("n_couriers:", n_couriers)
    print("items_count:", items_count)
    print("max_load:", max_load)
    print("size_item:", size_item)
    print("matrix_size:", matrix_size)    
    print("all_distances:", all_distances)


    # Solve the model
    result = instance.solve()

    return result

    
def main():
    filename = "C:\MCP\instances\instances\inst01.dat"
    n_couriers, items_count, max_load, size_item, matrix_size, all_distances = read_instance(filename)
    result = solve_mcp(n_couriers, items_count, max_load, size_item, matrix_size, all_distances)
    print(result)

if __name__ == "__main__":
    main()
