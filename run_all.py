import os
import subprocess

# Define the sizes and number of runs
sizes = [40000,45000,50000,55000,60000,65000,70000]
num_runs = 1
interfaces = ["mpi", "mpi_gather", "mpi+omp", "mpi+omp_gather"]
Iterations = 20

# Path to the driver script
driver_script = "driver.py"

# Loop through each size and execute the driver script
for size in sizes:
    for i in range(Iterations):
        cmd = [
            "python3",
            driver_script,
            "--kernels",
            "gemver",
            "--num-runs",
            str(num_runs),
            "--size",
            str(size),
            "--no-compile",
        ] 
        subprocess.run(cmd)
