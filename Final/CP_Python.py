import numpy as np
import os
import subprocess
import json
import time
import re
import tkinter as tk
from tkinter import filedialog

# Function to read the instance data from a file
def read_instance(filename):
    with open(filename, "r") as file:
        m_couriers_count = int(file.readline().strip()) # Read and parse the 1st line as the num of couriers(m)
        n_items_count = int(file.readline().strip()) # Read and parse the 2nd line as the num of items (n)
        l_max_loads = list(map(int, file.readline().strip().split())) # Read and parse the 3rd line as the max load for each courier
        s_load_sizes = list(map(int, file.readline().strip().split())) # Read and parse the 4th line as the sizes of the items

        # Check if the num of item sizes matches the num of items
        if len(s_load_sizes) != n_items_count:
            raise ValueError("Length of s_load_sizes does not match n_items_count")

        # Initialize an empty list for the distance matrix
        D_distances_matrix = []
        # Read the distance matrix from a file
        for _ in range(n_items_count + 1):
            D_distances_matrix.extend(list(map(int, file.readline().strip().split())))

        # Calculate the size of the matrix and reshape into 2D matrix
        matrix_size = int(len(D_distances_matrix) ** 0.5)
        D_distances_matrix = np.reshape(D_distances_matrix, (matrix_size, matrix_size)).tolist()

    return m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix

# Write the data to .dzn file
def write_dzn_file(filename, m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix):
    assert len(s_load_sizes) == n_items_count, "Length of s_load_sizes must match n_items_count"
    
    with open(filename, 'w') as f:
        f.write(f"m_couriers_count = {m_couriers_count};\n")
        f.write(f"n_items_count = {n_items_count};\n")
        f.write(f"l_max_loads = [{', '.join(map(str, l_max_loads))}];\n")
        f.write(f"s_load_sizes = [{', '.join(map(str, s_load_sizes))}];\n")
        f.write("D_distances_matrix = [| 0, ")

        # Iterate over each row of the distance matrix
        for i, row in enumerate(D_distances_matrix):
            if i == 0:
                f.write(f"{', '.join(map(str, row[1:]))},\n") # Write the 1st row except the 1st element
            elif i < len(D_distances_matrix) - 1:
                f.write(f"\t\t\t | {', '.join(map(str, row))},\n") # Write the intermediate rows
            else:
                f.write(f"\t\t\t | {', '.join(map(str, row))} |];\n") # Write the last row

# Function to solve the problem using the MiniZinc CLI
def solve_mcp_cli(dzn_filename, model_filename, solver_name, timeout_s=300):
    # Use a platform-independent way to find the MiniZinc executable
    minizinc_path = "minizinc" # define the path to the Minizinc executable
    start_time = time.time()
    
    # Start the subprocess to run Minizinc
    process = subprocess.Popen(
        [minizinc_path, "--solver", solver_name, model_filename, dzn_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        while process.poll() is None: # While the process is running
            time.sleep(1) # wait for 1 sec
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_s: # if timeout is exceeded terminate
                process.kill() 
                print(f"Timeout flag: {solver_name} solver timed out after {timeout_s} seconds.")
                return None, timeout_s

        stdout, stderr = process.communicate()
        elapsed_time = round((time.time() - start_time)) # Calculate total elapsed time

        if process.returncode == 0:
            print(f"Output for {solver_name}:\n{stdout}")
            return stdout, elapsed_time
        else:
            print(f"Error with {solver_name} solver:\n{stderr}")
            return None, elapsed_time

    except Exception as e:
        process.kill()
        print(f"Error: {solver_name} solver encountered an exception: {e}")
        return None, timeout_s

# Function to parse the output from Minizinc
def parse_minizinc_output(output):
    if not output: # If no output was received return default values
        return {"sol": None, "obj": None, "optimal": False}

    obj_match = re.search(r"maximum_distance = (\d+);", output) # Search for the objective value
    obj = int(obj_match.group(1)) if obj_match else None # Extract the objective value

    # Search for the delivery order
    order_of_delivery_section = re.search(r"order_of_delivery = \[(.*?)\];", output, re.DOTALL)
    order_of_delivery = []
    if order_of_delivery_section:
        rows = order_of_delivery_section.group(1).strip().splitlines() # Split the delivery order into lines
        for row in rows:
            cleaned_row = row.strip().replace('[', '').replace(']', '').strip()
            if cleaned_row:
                order_of_delivery.append([int(num) for num in cleaned_row.split(',') if num.strip()]) # Parse each row

    # Check if the solution is optimal
    optimal = "==========" in output

    return {
        "sol": order_of_delivery,
        "obj": obj,
        "optimal": optimal
    }

# Function to save results as a JSON file
def save_results_as_json(results, filepath):
    class CompactJSONEncoder(json.JSONEncoder): # Custom JSON encoder to handle compact lsits
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.indent = 4 # Set the indent level

        def encode(self, obj):
            if isinstance(obj, list):
                return '[{}]'.format(', '.join(self.encode(el) for el in obj)) # Handle lists
            return super().encode(obj) # Use default encoding for other types

    formatted_results = { 
        solver: {
            "time": results[solver]["time"],
            "sol": results[solver]["sol"] if results[solver]["sol"] is not None else [],
            "obj": results[solver]["obj"] if results[solver]["obj"] is not None else "null",
            "optimal": results[solver]["optimal"]
        }
        for solver in results # For each solver
    }

    with open(filepath, 'w') as result_file:
        result_file.write('{\n')
        for solver, data in formatted_results.items():
            result_file.write(f'    "{solver}": {{\n')
            result_file.write(f'        "time": {data["time"]},\n')
            result_file.write(f'        "sol": [\n')
            for route in data["sol"]:
                result_file.write(f'            {route},\n')
            result_file.write('        ],\n')
            result_file.write(f'        "obj": {data["obj"]},\n')
            result_file.write(f'        "optimal": {str(data["optimal"]).lower()}\n')
            result_file.write('    },\n' if solver == "gecode" else '    }\n')
        result_file.write('}\n')


# Function to process all .dat files in a directory
def process_all_dat_files(dat_directory, dzn_directory, result_directory, model_filename):
    if not os.path.exists(dzn_directory): # Check if the .dzn directory exista
        os.makedirs(dzn_directory) # Create new if not

    if not os.path.exists(result_directory): # Check if the result directory exist
        os.makedirs(result_directory) # Create a new one if not

    for dat_file in os.listdir(dat_directory): # Iterate over the files in the .dat directory
        if dat_file.endswith(".dat"):
            dat_filepath = os.path.join(dat_directory, dat_file) # Get the full path of the file
            print(f"Processing {dat_filepath}...")

            # Read the instance data from the .dat file
            m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix = read_instance(dat_filepath)

            # Create a .dzn filename from the .dat filename
            dzn_filename = os.path.splitext(dat_file)[0] + ".dzn"
            dzn_filepath = os.path.join(dzn_directory, dzn_filename) # Get the full path

            write_dzn_file(dzn_filepath, m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix)

            # Solve the problem using Gecode and Chuffed solvers and record the results
            result_gecode, time_gecode = solve_mcp_cli(dzn_filepath, model_filename, "gecode")
            result_chuffed, time_chuffed = solve_mcp_cli(dzn_filepath, model_filename, "chuffed")

            # Parse the ouput from Gecode
            gecode_result = parse_minizinc_output(result_gecode)
            gecode_result["time"] = time_gecode # time taken by Gecode

            # Parse the output from the Chuffed solver
            chuffed_result = parse_minizinc_output(result_chuffed)
            chuffed_result["time"] = time_chuffed

            results = {
                "gecode": {
                    "time": gecode_result["time"],
                    "sol": gecode_result["sol"],
                    "obj": gecode_result["obj"],
                    "optimal": gecode_result["optimal"],
                },
                "chuffed": {
                    "time": chuffed_result["time"],
                    "sol": chuffed_result["sol"],
                    "obj": chuffed_result["obj"],
                    "optimal": chuffed_result["optimal"],
                }
            }

            result_filename = os.path.splitext(dat_file)[0] + "_results.json" # Filename for the results
            result_filepath = os.path.join(result_directory, result_filename) # Get full path

            # Save the results to a JSon file
            save_results_as_json(results, result_filepath)

            print(f"Results saved to {result_filepath}")


def main():
    root = tk.Tk() # Create the Tkinter root window
    root.withdraw()  # Hide the main window

    # Ask user to select the necessary directories and files
    dat_directory = filedialog.askdirectory(title="Select the directory containing .dat files")
    dzn_directory = filedialog.askdirectory(title="Select the directory to save .dzn files")
    result_directory = filedialog.askdirectory(title="Select the directory to save results")
    model_filename = filedialog.askopenfilename(title="Select the MiniZinc model file", filetypes=[("MiniZinc files", "*.mzn")])

    # Check if all selections were made
    if not dat_directory or not dzn_directory or not result_directory or not model_filename:
        print("All selections are required. Exiting.")
        return

    # Process all .dat files using the selected directories and model file
    process_all_dat_files(dat_directory, dzn_directory, result_directory, model_filename)


if __name__ == "__main__":
    main()
