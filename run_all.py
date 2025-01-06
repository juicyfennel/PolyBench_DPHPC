import os
import subprocess

# Define the sizes and number of runs
# sizes = [40000,45000,50000,55000,60000,65000,70000] #2 -> 40000, 4 -> 42000, 8->46000, 12->50000, 16->54000, 24 -> 62000, 32 -> 70000
# size = 65000
# sizes = [[40000,0],[45000,1],[50000,2],[55000,3],[60000,4],[65000,5],[70000,6],[42000,1],[46000,2],[50000,3],[54000,4],[62000,5]]
# sizes = [
#     [20000,2,["omp_fastest"],200],
#     [28284,4,["omp_fastest"],200],
#     [40000,8,["omp_fastest"],200],
#     [56568,16,["omp_fastest"],200],
#     [80000,32,["omp_fastest"],200],
#     ]
# sizes = [
#     [40000,2,["std","std_fastest"],10],

# ]
# sizes = [[40000,2,["omp_fastest"],200]]

# sizes = [
#     [20000,2,["mpi_fastest"],200],
#     [28284,4,["mpi_fastest"],200],
#     [40000,8,["mpi_fastest"],200],
#     [56568,16,["mpi_fastest"],200],
#     [80000,32,["mpi_fastest"],200],
#     ]
num_runs = 1
# interfaces = ["mpi", "mpi_gather", "mpi+omp", "mpi+omp_gather"]
# interfaces = ["omp", "omp_blocked", "mpi", "mpi+omp"]
# interfaces = ["omp"]
# interfaces = ["std", "std_blocked"]
# Iterations = 50

# weak scaling jacobi
# sizes = [
#     [14142,4,["omp"],100],
#     [21213,9,["omp"],100],
#     [28284,16,["omp"],100],
#     [35355,25,["omp"],100],
#     [42426,36,["omp"],100],
#     ]
sizes = [
    [14142,4,["mpi"],100],
    [21213,9,["mpi"],100],
    [28284,16,["mpi"],100],
    [35355,25,["mpi"],100],
    [42426,36,["mpi"],100],
    ]
# sizes = [
#     [20000,8,["mpi+omp"],100],
#     [28284,16,["mpi+omp"],100],
#     [30000,18,["mpi+omp"],100],
#     [42426,36,["mpi+omp"],100],
#     ]

# strong scaling jacobi
# sizes = [28284, 1, ["std"], 100]
# sizes = [
#     [28284,4,["omp"],100],
#     [28284,9,["omp"],100],
#     [28284,16,["omp"],100],
#     [28284,25,["omp"],100],
#     [28284,36,["omp"],100],
#     ]
# sizes = [
#     [28284,4,["mpi"],100],
#     [28284,9,["mpi"],100],
#     [28284,16,["mpi"],100],
#     [28284,25,["mpi"],100],
#     [28284,36,["mpi"],100],
#     ]
# sizes = [
#     [28284,8,["mpi+omp"],100],
#     [28284,16,["mpi+omp"],100],
#     [28284,18,["mpi+omp"],100],
#     [28284,36,["mpi+omp"],100],
#     ]

# Path to the driver script
driver_script = "driver.py"

# Loop through each size and execute the driver script
for size in sizes:
    for i in range(size[-1]):
        cmd = [
            "python3",
            driver_script,
            "--kernels",
            "jacobi-2d",
            "--num-runs",
            str(num_runs),
            "--size",
            str(size[0]),
            "--processes",
            str(size[1]),
            "--no-compile",
            "--interfaces",
        ] + size[2]
        subprocess.run(cmd)
