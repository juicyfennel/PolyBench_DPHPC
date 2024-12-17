import os
import subprocess

# Define the sizes and number of runs
sizes = [10000, 25000, 40000]
num_runs = 10

# Path to the driver script
driver_script = "driver.py"

# Loop through each size and execute the driver script
for size in sizes:
    print(f"Running for size: {size}")
    cmd = [
        "python3",
        driver_script,
        "--kernels",
        "gemver",
        "--num-runs",
        str(num_runs),
        "--size",
        str(size),
    ]
    subprocess.run(cmd)
