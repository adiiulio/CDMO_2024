import subprocess

def run_instance_with_timeout(instance_number, config, timeout_seconds):
    try:
        # Construct the command to run the instance with the given config
        command = f"python smt_final.py {instance_number} {config}"
        
        # Run the command as a subprocess with a timeout
        result = subprocess.run(command, timeout=timeout_seconds, shell=True)

        if result.returncode == 0:
            print(f"Instance {instance_number}, Configuration {config}: Completed Successfully")
        else:
            print(f"Instance {instance_number}, Configuration {config}: Failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print(f"Instance {instance_number}, Configuration {config}: Timed out after {timeout_seconds} seconds")

# Loop through the instances and configurations
if __name__ == "__main__":
    for instance in range(1, 22):  # Adjust the range as needed
        for config in range(1, 5):
            run_instance_with_timeout(instance, config, 300)

