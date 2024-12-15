import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# Argument parser to take the kernel as input
parser = argparse.ArgumentParser(description="Store output for a given kernel as csv")
parser.add_argument(
    "--all",
    action="store_true",
    help="Store all measurements from ./outputs (default: latest only)",
)
args = parser.parse_args()

# Construct the directory path for the given kernel
output_dir = "./outputs"

# Check if the directory exists
if not os.path.exists(output_dir):
    print("Error: The directory ./outputs does not exist.")
    exit(1)

# List containing all folders in ./outputs
benchmark_outputs = [
    f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))
]

# If --all flag is not provided, store only latest measurements
if not args.all:
    # Get the latest folder based on the timestamp in the folder name
    latest_folder = max(
        benchmark_outputs, key=lambda f: datetime.strptime(f, "%Y_%m_%d__%H-%M-%S")
    )
    benchmark_outputs = [latest_folder]

# List to store the rows of the DataFrame
rows = {}

for bm in benchmark_outputs:
    dirs = [
        f
        for f in os.listdir(f"./outputs/{bm}")
        if os.path.isdir(os.path.join(f"./outputs/{bm}", f))
    ]

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
            key_val_list = dataset_str.strip("_").split("_")
            key_val_pairs = dict(zip(key_val_list[::2], key_val_list[1::2]))
            dataset = "; ".join(
                [f"{k}={v}" for k, v in key_val_pairs.items() if k != "np"]
            )

        interface = match.group("interface")

        err_dir = f"./outputs/{bm}/{dir}/err"
        err_files = [f for f in os.listdir(err_dir) if f.endswith(".err")]
        out_dir = f"./outputs/{bm}/{dir}/out"
        out_files = [f for f in os.listdir(out_dir) if f.endswith(".out")]

        # for file in err_files:
        #     with open(f"{err_dir}/{file}", "r") as f:
        #         if not f.read() == "":
        #             print(f"Error occured during benchmarking: See {err_dir}/{file}")
        #             exit(1)

        for file in out_files:
            res = []
            with open(f"{out_dir}/{file}", "r") as f:
                for line in f:
                    res.append(float(line.strip()))
            rows[kernel].append(
                {
                    "Kernel": kernel,
                    "Interface": interface,
                    **key_val_pairs,
                    "Dataset": dataset,
                    "Execution Time": res,
                }
            )

    # Create a DataFrame from the rows
    for kernel, data in rows.items():
        df = pd.DataFrame(data)
        df["Max Execution Time"] = df["Execution Time"].apply(
            lambda x: max(x) if isinstance(x, list) else x
        )
        df["Min Execution Time"] = df["Execution Time"].apply(
            lambda x: min(x) if isinstance(x, list) else x
        )
        df["Median Execution Time"] = df["Execution Time"].apply(
            lambda x: np.median(x) if isinstance(x, list) else x
        )
        df["Mean Execution Time"] = df["Execution Time"].apply(
            lambda x: np.mean(x) if isinstance(x, list) else x
        )
        df["Execution Time STD"] = df["Execution Time"].apply(
            lambda x: np.std(x) if isinstance(x, list) else x
        )
        if not os.path.exists(f"./outputs/{bm}/data"):
            os.makedirs(f"./outputs/{bm}/data")
        df.to_csv(f"./outputs/{bm}/data/{kernel}.csv", index=False)
