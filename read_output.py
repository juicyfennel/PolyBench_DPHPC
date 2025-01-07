import argparse
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from k_means import get_fast_group
import json

# Argument parser
parser = argparse.ArgumentParser(description="Process runtime outputs into CSV and JSON files")
parser.add_argument(
    "--dir",
    help="Path to the specific directory containing benchmark outputs (e.g., ./outputs/2024_12_15__14-30-45).",
)

parser.add_argument(
    "--clusters",
    type=int,
    default=1,
    help="Number of clusters to use for K-means clustering. Defaults to 1.",
)

args = parser.parse_args()

clusters = args.clusters

# Base directory
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

rows = {}
json_data = {}
time_pattern = re.compile(r"Time(?: for Kernel calculation)?:\s*([\d.]+)")

# Process the provided or determined benchmark folder
dirs = [
    f
    for f in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, f))
]
date = output_dir.split("/")[-1]

for dir in dirs:
    match = re.match(r"^(?P<kernel>[A-Za-z0-9-]+)_N_(?P<size>\d+)_np_(?P<processes>\d+)_(?P<type>[\w+]+)$", dir)
    if not match:
        continue

    kernel = match.group("kernel")
    size = int(match.group("size"))
    if size not in rows:
        rows[size] =[]
    if size not in json_data:
        json_data[size] = []
    num_processes = int(match.group("processes"))
    num_processes_original = num_processes
    run_type = match.group("type")

    out_dir = os.path.join(output_dir, dir)
    out_files = [f for f in os.listdir(out_dir) if f.endswith(".out")]

    for file in out_files:
        with open(os.path.join(out_dir, file), "r") as f:
            lines = f.readlines()
        if run_type.startswith("mpi+omp"):   
            num_processes = 0
        flag = False
        valid_lines = []
        for line in lines:
            match = time_pattern.search(line)
            if run_type.startswith("mpi+omp"):
                if "=" in line:
                    flag = True
            if match:
                try:
                    runtime = float(match.group(1))
                    if run_type.startswith("mpi+omp") and not flag:
                        num_processes += 1
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
            if len(max_runtimes) < clusters:
                continue
            max_runtimes = get_fast_group(max_runtimes,date,dir,clusters)
            mean_runtime = np.mean(max_runtimes)
            variability = np.std(max_runtimes)
            rows[size].append({
                "Kernel": kernel,
                "Size": size,
                "Processes": num_processes_original,
                "Type": run_type,
                "Mean Runtime": mean_runtime,
                "STD": variability,
                "num-runs": len(max_runtimes)
            })
            json_data[size].append({
                "Kernel": kernel,
                "Size": size,
                "Processes": num_processes_original,
                "Type": run_type,
                "Mean Runtime": mean_runtime,
                "STD": variability,
                "num-runs": len(max_runtimes),
                "Data Points": max_runtimes
            })

        elif run_type in {"omp", "omp_blocked", "omp_fastest", "omp_fastest2"}:
            if valid_lines and len(valid_lines) >= clusters:
                valid_lines = get_fast_group(valid_lines,date,dir,clusters)
                mean_runtime = np.mean(valid_lines)
                variability = np.std(valid_lines)
                rows[size].append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": num_processes,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines)
                })
                json_data[size].append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": num_processes,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines),
                    "Data Points": valid_lines
                })

        elif (run_type == "std" or run_type == "std_blocked" or run_type == "std_fastest"):
            if valid_lines and len(valid_lines) >= clusters:
                valid_lines = get_fast_group(valid_lines,date,dir,clusters)
                mean_runtime = np.mean(valid_lines)
                variability = np.std(valid_lines)
                rows[size].append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": 1,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines)  
                })
                json_data[size].append({
                    "Kernel": kernel,
                    "Size": size,
                    "Processes": 1,
                    "Type": run_type,
                    "Mean Runtime": mean_runtime,
                    "STD": variability,
                    "num-runs": len(valid_lines),
                    "Data Points": valid_lines
                })

# Create a new runtime_analysis directory with the same date_time as the source
analysis_dir = os.path.join("./runtime_analysis", os.path.basename(output_dir))
os.makedirs(analysis_dir, exist_ok=True)

# Save individual CSV and JSON files for each size
for size in rows:
    output_file_csv = os.path.join(analysis_dir, f"runtime_analysis_{size}.csv")
    output_file_json = os.path.join(analysis_dir, f"runtime_analysis_{size}.json")
    df = pd.DataFrame(rows[size])
    df.to_csv(output_file_csv, index=False)
    with open(output_file_json, "w") as json_file:
        json.dump(json_data[size], json_file, indent=4)
    print(f"Runtime analysis for size {size} saved to {output_file_csv} and {output_file_json}")

# Combine all rows into a single DataFrame
all_data = pd.concat([pd.DataFrame(rows[size]) for size in rows])

# Define selection conditions for the final CSV file
conditions = [
    (20000, 2), (28284, 4), (40000, 8),
    (56568, 16), (80000, 32)
]

# Filter and save weak_scaling_data.csv and weak_scaling_data.json
weak_scaling_data = all_data[
    all_data.apply(
        lambda x: (x["Size"], x["Processes"]) in conditions and
                  x["Type"] in {"omp", "omp_fastest", "mpi", "mpi_fastest","mpi_fastest_128B", "mpi+omp", "mpi+omp_fastest"},
        axis=1
    )
]
weak_scaling_csv_path = os.path.join(analysis_dir, "weak_scaling_data.csv")
weak_scaling_json_path = os.path.join(analysis_dir, "weak_scaling_data.json")
weak_scaling_data.to_csv(weak_scaling_csv_path, index=False)

# Add data points to the weak_scaling_data JSON
weak_scaling_json = []
for _, row in weak_scaling_data.iterrows():
    size = row['Size']
    matching_entry = next((entry for entry in json_data[size] if 
                           entry['Processes'] == row['Processes'] and 
                           entry['Type'] == row['Type']), None)
    if matching_entry:
        weak_scaling_json.append(matching_entry)

with open(weak_scaling_json_path, "w") as json_file:
    json.dump(weak_scaling_json, json_file, indent=4)

print(f"Weak scaling data saved to {weak_scaling_csv_path} and {weak_scaling_json_path}")
