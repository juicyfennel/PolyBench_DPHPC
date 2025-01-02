import argparse
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from k_means import get_fast_group

# Argument parser
parser = argparse.ArgumentParser(description="Process runtime outputs into CSV files")
parser.add_argument(
    "--dir",
    default=None,
    help="Path to the specific directory containing benchmark outputs (e.g., ./outputs/2024_12_15__14-30-45). Defaults to the latest folder in ./outputs.",
)

args = parser.parse_args()
output_base = "./outputs"

# Determine the directory to process
if args.dir:
    output_dir = args.dir
    if not os.path.exists(output_dir):
        print(f"Error: The directory {output_dir} does not exist.")
        exit(1)
else:
    # Default to the latest folder in ./outputs
    if not os.path.exists(output_base):
        print(f"Error: The directory {output_base} does not exist.")
        exit(1)
    benchmark_outputs = [
        f for f in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, f))
    ]
    if not benchmark_outputs:
        print(f"Error: No benchmark folders found in {output_base}.")
        exit(1)
    latest_folder = max(
        benchmark_outputs, key=lambda f: datetime.strptime(f, "%Y_%m_%d__%H-%M-%S")
    )
    output_dir = os.path.join(output_base, latest_folder)

print(f"Processing directory: {output_dir}")

rows = []
time_pattern = re.compile(r"Time:\s*([\d.]+)")

# Process the provided or determined benchmark folder
dirs = [
    f
    for f in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, f))
]

for dir in dirs:
    match = re.match(r"^(?P<kernel>[A-Za-z0-9-]+)_N_(?P<size>\d+)_np_(?P<processes>\d+)_(?P<type>[\w+]+)$", dir)
    if not match:
        continue

    kernel = match.group("kernel")
    size = int(match.group("size"))
    num_processes = int(match.group("processes"))
    num_processes_original = num_processes
    run_type = match.group("type")

    out_dir = os.path.join(output_dir, dir)
    out_files = [f for f in os.listdir(out_dir) if f.endswith(".out")]

    for file in out_files:
        with open(os.path.join(out_dir, file), "r") as f:
            lines = f.readlines()
        if run_type == "mpi+omp" or run_type == "mpi+omp_gather":   
            num_processes = 0
        flag = False
        valid_lines = []
        for line in lines:
            match = time_pattern.search(line)
            if run_type == "mpi+omp" or run_type == "mpi+omp_gather":
                if "=" in line:
                    flag = True
                elif not flag:
                    num_processes += 1
            if match:
                try:
                    runtime = float(match.group(1))
                    valid_lines.append(runtime)
                except ValueError:
                    continue
            else:
                try:
                    runtime = float(line)
                    valid_lines.append(runtime)
                except ValueError:
                    continue

        if run_type.startswith("mpi"):
            runs = [
                valid_lines[i:i + num_processes]
                for i in range(0, len(valid_lines), num_processes)
            ]
            max_runtimes = []
            for run in runs:
                if len(run) == num_processes:
                    max_runtime = max(run)
                    max_runtimes.append(max_runtime)
            # max_runtimes = get_fast_group(max_runtimes)
            mean_runtime = np.mean(max_runtimes)
            variability = np.std(max_runtimes)
            rows.append({
                "Kernel": kernel,
                "Size": size,
                "Processes": num_processes_original,
                "Type": run_type,
                "Mean Runtime": mean_runtime,
                "STD": variability,
                "num-runs": len(max_runtimes)
            })
        elif run_type == "omp":
            if valid_lines:
                # valid_lines = get_fast_group(valid_lines)
                mean_runtime = np.mean(valid_lines)
                variability = np.std(valid_lines)
                rows.append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": num_processes,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines)
                })
        elif run_type == "std":
            if valid_lines:
                # valid_lines = get_fast_group(valid_lines)
                mean_runtime = np.mean(valid_lines)
                variability = np.std(valid_lines)
                rows.append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": 1,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines)  
                })

# Create a DataFrame
df = pd.DataFrame(rows)

# Create a new runtime_analysis directory with the same date_time as the source
analysis_dir = os.path.join("./runtime_analysis", os.path.basename(output_dir))
os.makedirs(analysis_dir, exist_ok=True)

# Save a single CSV file for the processed data     
output_file = os.path.join(analysis_dir, "runtime_analysis.csv")
df.to_csv(output_file, index=False)
print(f"Runtime analysis saved to {output_file}")
