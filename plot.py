import argparse
import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Argument parser to take the kernel as input
parser = argparse.ArgumentParser(description="Plot execution times for a given kernel")
parser.add_argument("--all", action="store_true", help="Plot all measurements from ./outputs (default: latest only)")
parser.add_argument("--remake", action="store_true", help="Remake existing plots")
args = parser.parse_args()

# Construct the directory path for the given kernel
output_dir = "./outputs"

# Check if the directory exists
if not os.path.exists(output_dir):
    print(f"Error: The directory ./outputs does not exist.")
    exit(1)
    
# List containing all folders in ./outputs
benchmark_outputs = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]

# If --all flag is not provided, plot only latest measurements
if not args.all:
    # Get the latest folder based on the timestamp in the folder name
    latest_folder = max(benchmark_outputs, key=lambda f: datetime.strptime(f, "%Y_%m_%d__%H-%M-%S"))
    benchmark_outputs = [latest_folder]

# List to store the rows of the DataFrame
rows = {}

for bm in benchmark_outputs:
    # skip if plot already exists
    if (not args.remake) and os.path.exists(f"./outputs/{bm}/plots"):
        continue
    
    mpi_config = {}
    # if mpi.json exists, read mpi configuration from it
    if os.path.exists(f"./outputs/{bm}/mpi.json"):
        with open(f"./outputs/{bm}/mpi.json", "r") as f:
            mpi_config = json.load(f)
    
    dirs = [f for f in os.listdir(f"./outputs/{bm}") if os.path.isdir(os.path.join(f"./outputs/{bm}", f))]
    
    for dir in dirs:
        # Retrieve the kernel, datasets, and interface from the directory name
        pattern = r"^(?P<kernel>[A-Za-z0-9-]+)_((?P<datasets>(?:\w+_\w+_)*)?)(?P<interface>\w+)$"
        match = re.match(pattern, dir)

        if not match:
            continue

        kernel = match.group("kernel")
        if kernel not in rows:
            rows[kernel] = []
        dataset_str = match.group("datasets")
        dataset = ""
        if dataset_str:
            # Split into individual key-value strings and form dictionary
            key_val_list = dataset_str.strip('_').split('_')
            dataset = "; ".join(
            f"{key}: {val}" for key, val in zip(key_val_list[::2], key_val_list[1::2])
            )
        interface = match.group("interface")

        err_dir = f"./outputs/{bm}/{dir}/err"
        err_files = [f for f in os.listdir(err_dir) if f.endswith(".err")]
        out_dir = f"./outputs/{bm}/{dir}/out"
        out_files = [f for f in os.listdir(out_dir) if f.endswith(".out")]

        for file in err_files:
            with open(f"{err_dir}/{file}", "r") as f:
                if not f.read() == "":
                    print(f"Error occured during benchmarking: See {err_dir}/{file}")
                    exit(1)
        
        
        for file in out_files:
            res = []
            with open(f"{out_dir}/{file}", "r") as f:
                for line in f:
                    res.append(float(line.strip()))
            rows[kernel].append({"Kernel": kernel, "Interface": interface, "Dataset": dataset, "Execution Time": max(res)})

    dfs = {}
    # Create a DataFrame from the rows
    for kernel, data in rows.items():
        dfs[kernel] = pd.DataFrame(data)

    for kernel, df in dfs.items():
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        # Line plot with confidence interval for the median
        sns.lineplot(
            x="Dataset",
            y="Execution Time",
            hue="Interface",
            data=df,
            estimator=np.median,  # dk if median or mean is better
            errorbar=("ci", 95),  # 95% confidence interval
            n_boot=1000  # Number of bootstrap samples
        )
        # Set the axis labels and title
        plt.xlabel("Input Size", fontsize=12)
        plt.ylabel("Execution Time [s]", fontsize=12)
        plt.title(f"Execution Times for {kernel} Across Different Parallelization Paradigms and Input Sizes", fontsize=14)
        plt.legend(title="Paradigm", fontsize=10)

        plt.tight_layout()

        # Save the plot to a file
        if not os.path.exists(f"./outputs/{bm}/plots"):
            os.makedirs(f"./outputs/{bm}/plots")

        plt.savefig(f"./outputs/{bm}/plots/{kernel}.png")

    exit(0)
