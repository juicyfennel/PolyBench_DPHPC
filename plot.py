import argparse
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Argument parser to take the kernel as input
parser = argparse.ArgumentParser(description="Plot execution times for a given kernel")
parser.add_argument("kernel", type=str, help="Base name of the kernel (e.g., 'gemver')")
args = parser.parse_args()

# Construct the directory path for the given kernel
kernel_dir = os.path.join("measurements", args.kernel)

# Check if the directory exists
if not os.path.exists(kernel_dir):
    print(f"Error: The directory {kernel_dir} does not exist.")
    exit(1)

# Get the list of JSON files in the directory
json_files = [f for f in os.listdir(kernel_dir) if f.endswith(".json")]
if not json_files:
    print(f"Error: No JSON files found in {kernel_dir}")
    exit(1)

# Find the latest JSON file based on the timestamp in the filename
latest_file = max(json_files, key=lambda f: datetime.strptime(f.split(".")[0], "%Y_%m_%d__%H:%M:%S"))
latest_file_path = os.path.join(kernel_dir, latest_file)

# Load the data from the latest JSON file
with open(latest_file_path, "r") as file:
    data = json.load(file)

# Convert JSON data to a DataFrame for plotting
rows = []
for dataset, kernels in data.items():
    for version, times in kernels.items():
        for time in times:
            rows.append({"Dataset": dataset, "Kernel Version": version, "Execution Time": time})

df = pd.DataFrame(rows)

# Create the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Line plot with confidence interval for the median
sns.lineplot(
    x="Dataset",
    y="Execution Time",
    hue="Kernel Version",
    data=df,
    estimator=np.median,  # dk if median or mean is better
    ci=95,  # 95% confidence interval
    n_boot=1000  # Number of bootstrap samples
)

# Set the axis labels and title
plt.xlabel("Dataset Size", fontsize=12)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.title(f"Execution Time Comparison for {args.kernel} Across Different Versions and Dataset Sizes", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title="Kernel Version", fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
