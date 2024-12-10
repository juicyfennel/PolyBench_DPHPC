import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Argument parser to take the kernel as input
parser = argparse.ArgumentParser(description="Plot execution times for a given kernel")
parser.add_argument("--all", action="store_true", help="Plot all measurements from ./outputs (default: latest only)")
parser.add_argument("--type", type=str, default="all", help="Type of plot to generate (default: np)")
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


def np_plot(bm, kernel, df):
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    # Line plot with confidence interval for the median
    sns.lineplot(
        x="np",
        y="Max Execution Time",
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
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(title="Paradigm", fontsize=10)

    plt.tight_layout()

    # Save the plot to a file
    if not os.path.exists(f"./outputs/{bm}/plots"):
        os.makedirs(f"./outputs/{bm}/plots")

    plt.savefig(f"./outputs/{bm}/plots/{kernel}_np.png")
    plt.savefig(f"./outputs/{bm}/plots/{kernel}_np.svg")


def dataset_plot(bm, kernel, df):
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    # Line plot with confidence interval for the median
    sns.lineplot(
        x="Dataset",
        y="Max Execution Time",
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
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(title="Paradigm", fontsize=10)

    plt.tight_layout()

    # Save the plot to a file
    if not os.path.exists(f"./outputs/{bm}/plots"):
        os.makedirs(f"./outputs/{bm}/plots")

    plt.savefig(f"./outputs/{bm}/plots/{kernel}_datasets.png")
    plt.savefig(f"./outputs/{bm}/plots/{kernel}_datasets.svg")


for bm in benchmark_outputs:
    dfs = {}
    csv_files = [f for f in os.listdir(f"./outputs/{bm}/data") if f.endswith(".csv")]

    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f"./outputs/{bm}/data/{csv_file}")
        # Extract the kernel name from the CSV file name
        kernel = csv_file.split(".")[0]
        # Add the DataFrame to the dictionary
        dfs[kernel] = df


    for kernel, df in dfs.items():
        if args.type == "np" or args.type == "all":
            np_plot(bm, kernel, df)

        if args.type == "dataset" or args.type == "all":
            dataset_plot(bm, kernel, df)

    exit(0)
