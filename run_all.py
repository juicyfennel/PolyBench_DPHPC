import os
import subprocess

# Define the sizes and number of runs
# sizes = [40000,45000,50000,55000,60000,65000,70000] #2 -> 40000, 4 -> 42000, 8->46000, 12->50000, 16->54000, 24 -> 62000, 32 -> 70000
# size = 65000
sizes = [[40000,0],[45000,1],[50000,2],[55000,3],[60000,4],[65000,5],[70000,6],[42000,1],[46000,2],[50000,3],[54000,4],[62000,5]]
# sizes = [[40000,0],[42000,1],[46000,2],[50000,3],[54000,4],[62000,5],[70000,6]]
num_runs = 1
# interfaces = ["mpi", "mpi_gather", "mpi+omp", "mpi+omp_gather"]
# interfaces = ["omp","omp_blocked","mpi","mpi+omp"]
interfaces = ["omp"]
# interfaces = ["std", "std_blocked"]
Iterations = 50

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
            str(size[0]),
            "--idxProcesses",
            str(size[1]),
            "--no-compile",
            "--interfaces",
        ] + interfaces
        subprocess.run(cmd)
