import numpy as np
import os
import subprocess
import json
import time
import re

# Function to read the instance data from a file
def read_instance(filename):
    with open(filename, "r") as file:
        m_couriers_count = int(file.readline().strip()) # Read and parse the 1st line as the num of couriers(m)
        n_items_count = int(file.readline().strip()) # Read and parse the 2nd line as the num of items (n)
        l_max_loads = list(map(int, file.readline().strip().split())) # Read and parse the 3rd line as the max load for each courier
        s_load_sizes = list(map(int, file.readline().strip().split())) # Read and parse the 4th line as the sizes of the items


        # Check if s_load_sizes length matches n_items_count
        if len(s_load_sizes) != n_items_count:
            raise ValueError("Length of s_load_sizes does not match n_items_count")

        D_distances_matrix = []

        # Flatten the 2D array
        for _ in range(n_items_count + 1):
            D_distances_matrix.extend(list(map(int, file.readline().strip().split())))

        # Reshape the flat list into a 2D array
        matrix_size = int(len(D_distances_matrix) ** 0.5)  # Calculate the size of the matrix
        D_distances_matrix = np.reshape(D_distances_matrix, (matrix_size, matrix_size)).tolist()

    return m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix

# Write the data to .dzn file
def write_dzn_file(filename, m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix):
    # Ensure that s_load_sizes has the correct length
    assert len(s_load_sizes) == n_items_count, "Length of s_load_sizes must match n_items_count"
    
    with open(filename, 'w') as f:
        f.write(f"m_couriers_count = {m_couriers_count};\n")
        f.write(f"n_items_count = {n_items_count};\n")
        
        # Manually format l_max_loads and s_load_sizes as comma-delimited lists within square brackets
        f.write(f"l_max_loads = [{', '.join(map(str, l_max_loads))}];\n")
        f.write(f"s_load_sizes = [{', '.join(map(str, s_load_sizes))}];\n")
        
        # Start writing the matrix with the correct 2D format and exact placement
        f.write("D_distances_matrix = [| 0, ")

        # Loop through each row in the matrix
        for i, row in enumerate(D_distances_matrix):
            if i == 0:
                # Add the rest of the numbers in the first row
                f.write(f"{', '.join(map(str, row[1:]))},\n")
            elif i < len(D_distances_matrix) - 1:
                # Format the subsequent rows with tabs and correct placement of vertical bars
                f.write(f"\t\t\t | {', '.join(map(str, row))},\n")
            else:
                # For the last row, ensure correct placement without an extra comma or vertical bar
                f.write(f"\t\t\t | {', '.join(map(str, row))} |];\n")

# Function to solve the problem using the MiniZinc CLI
def solve_mcp_cli(dzn_filename, model_filename, solver_name, timeout_s=300):
    minizinc_path = r"minizinc" # define the path to the Minizinc executable
    start_time = time.time()
    
    # Start the subprocess to run Minizinc
    process = subprocess.Popen(
        [minizinc_path, "--solver", solver_name, model_filename, dzn_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Poll the process regularly to enforce the timeout
        while process.poll() is None:
            time.sleep(1)  # Sleep for a short while to avoid busy-waiting
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_s:
                process.kill()  # Kill the process if it exceeds the timeout
                print(f"Timeout flag: {solver_name} solver timed out after {timeout_s} seconds.")
                return None, timeout_s  # Return timeout in seconds

        # Capture the output and return it
        stdout, stderr = process.communicate()
        elapsed_time = round((time.time() - start_time))  # Time in seconds

        if process.returncode == 0:
            print(f"Output for {solver_name}:\n{stdout}")  # Print the full output
            return stdout, elapsed_time
        else:
            print(f"Error with {solver_name} solver:\n{stderr}")
            return None, elapsed_time

    except Exception as e:
        process.kill()
        print(f"Error: {solver_name} solver encountered an exception: {e}")
        return None, timeout_s

# Function to parse the output from Minizinc
def parse_minizinc_output(output, depot_number=None):
    if not output:
        return {"sol": None, "obj": None, "optimal": False}

    # Extract the maximum_distance (objective function value)
    obj_match = re.search(r"maximum_distance = (\d+);", output) # Search for the objective value
    obj = int(obj_match.group(1)) if obj_match else None # Extract the objective value

    # Extract the order_of_delivery matrix
    order_of_delivery_section = re.search(r"order_of_delivery = \[(.*?)\];", output, re.DOTALL)
    order_of_delivery = []

    if depot_number is None:
        # Automatically detect the depot number if not provided
        # Assuming the depot number is the one that appears first and is repeated in the output
        depot_match = re.search(r"order_of_delivery = \[\s*\[(\d+),", output)
        depot_number = int(depot_match.group(1)) if depot_match else None

    if order_of_delivery_section and depot_number is not None:
        rows = order_of_delivery_section.group(1).strip().splitlines()
        for row in rows:
            # Clean up the row, remove spaces, brackets, and split by comma
            cleaned_row = row.strip().replace('[', '').replace(']', '').strip()
            if cleaned_row:  # Ensure the row is not empty
                # Convert to list of integers
                path = [int(num) for num in cleaned_row.split(',') if num.strip()]
                
                # Remove depot number from the beginning and end of the path
                if path and path[0] == depot_number:
                    path = path[1:]  # Remove depot from the start
                if path and path[-1] == depot_number:
                    path = path[:-1]  # Remove depot from the end

                # Remove any depot numbers that might appear in the middle of the path
                path = [num for num in path if num != depot_number]

                order_of_delivery.append(path)

    # Determine if the solution is optimal (by checking if MiniZinc finished normally)
    optimal = "==========" in output

    return {
        "sol": order_of_delivery,  # Solution paths for each courier
        "obj": obj,                # Objective function value (maximum_distance)
        "optimal": optimal         # Whether the solution is optimal or not
    }



def save_results_as_json(results, filepath):
    # Custom encoder to control the compact array format in JSON
    class CompactJSONEncoder(json.JSONEncoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.indent = 4

        def encode(self, obj):
            if isinstance(obj, list):
                # Ensure that only the nested lists under 'sol' are compacted
                return '[{}]'.format(', '.join(self.encode(el) for el in obj))
            return super().encode(obj)
    # Prepare the results dictionary in the correct format
    formatted_results = {
        solver: {
            "time": results[solver]["time"],
            "sol": results[solver]["sol"] if results[solver]["sol"] is not None else [],
            "obj": results[solver]["obj"] if results[solver]["obj"] is not None else "null",
            "optimal": results[solver]["optimal"]
        }
        for solver in results
    }

    # Write the results to a file
    with open(filepath, 'w') as result_file:
        result_file.write('{\n')
        solvers = list(formatted_results.keys())
        
        for i, (solver, data) in enumerate(formatted_results.items()):
            result_file.write(f'    "{solver}": {{\n')
            result_file.write(f'        "time": {data["time"]},\n')
            result_file.write(f'        "sol": [\n')
            
            for j, route in enumerate(data["sol"]):
                result_file.write(f'            {route}')
                if j < len(data["sol"]) - 1:  # Check if it's not the last route
                    result_file.write(',\n')
                else:
                    result_file.write('\n')  # No comma after the last route
            
            result_file.write('        ],\n')
            result_file.write(f'        "obj": {data["obj"]},\n')
            result_file.write(f'        "optimal": {str(data["optimal"]).lower()}\n')
            
            # Add a comma after the solver block if it's not the last one
            if i < len(solvers) - 1:
                result_file.write('    },\n')
            else:
                result_file.write('    }\n')
        
        result_file.write('}\n')        


# Function to process all .dat files in a directory
def process_all_dat_files(dat_directory, dzn_directory, result_directory, model_filename):
    if not os.path.exists(dzn_directory):
        os.makedirs(dzn_directory)

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    for dat_file in os.listdir(dat_directory):
        if dat_file.endswith(".dat"):
            dat_filepath = os.path.join(dat_directory, dat_file)
            print(f"Processing {dat_filepath}...")

            # Read the instance data
            m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix = read_instance(dat_filepath)

            # Create a corresponding .dzn filename
            dzn_filename = os.path.splitext(dat_file)[0] + ".dzn"
            dzn_filepath = os.path.join(dzn_directory, dzn_filename)

            # Write the instance data to a .dzn file
            write_dzn_file(dzn_filepath, m_couriers_count, n_items_count, l_max_loads, s_load_sizes, D_distances_matrix)

            # Solve the problem using MiniZinc with both solvers
            result_gecode, time_gecode = solve_mcp_cli(dzn_filepath, model_filename, "gecode")
            result_chuffed, time_chuffed = solve_mcp_cli(dzn_filepath, model_filename, "chuffed")

            # Parse results
            gecode_result = parse_minizinc_output(result_gecode)
            gecode_result["time"] = time_gecode

            chuffed_result = parse_minizinc_output(result_chuffed)
            chuffed_result["time"] = time_chuffed

            # Prepare the final results dictionary
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

            # Save the results to the result directory in JSON format
            result_filename = os.path.splitext(dat_file)[0] + "_results.json"
            result_filepath = os.path.join(result_directory, result_filename)

            save_results_as_json(results, result_filepath)

            print(f"Results saved to {result_filepath}")


def main():
    dat_directory = "instances/instances"
    dzn_directory = "instances/instancesDZN"
    result_directory = "instances/results" # If you want to save the results through docker, you can save here instances/reesults_docker
    model_filename = "MCP_Model_2.mzn"

    process_all_dat_files(dat_directory, dzn_directory, result_directory, model_filename)


if __name__ == "__main__":
    main()