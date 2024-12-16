import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot runtime analysis from a CSV file")
parser.add_argument(
    "--file", required=True, help="Path to the runtime_analysis.csv file"
)
args = parser.parse_args()

# Validate the input file
if not os.path.exists(args.file):
    print(f"Error: File {args.file} does not exist.")
    exit(1)

# Load the CSV data
df = pd.read_csv(args.file)

# Extract the problem size
if "Size" not in df.columns:
    print("Error: 'Size' column not found in the CSV.")
    exit(1)
problem_size = df["Size"].iloc[0]  # Assumes all rows have the same size
print(f"Problem Size: {problem_size}")

# Group by Kernel, Size, Processes, and Type to compute averages
df_grouped = (
    df.groupby(["Kernel", "Size", "Processes", "Type"])
    .agg(
        {
            "Max Runtime": "mean",  # Average runtime
            "STD": "mean",  # Average standard deviation
        }
    )
    .reset_index()
)

# Calculate Speedup and Efficiency
baseline_runtime = df_grouped[
    (df_grouped["Processes"] == 1) & (df_grouped["Type"] == "std")
]["Max Runtime"].mean()
if pd.isna(baseline_runtime):
    print("Error: Could not find baseline runtime for standard (std) runs.")
    exit(1)

df_grouped["Speedup"] = baseline_runtime / df_grouped["Max Runtime"]
df_grouped["Efficiency"] = df_grouped["Speedup"] / df_grouped["Processes"]

# Plot Runtime vs Number of Processes
plt.figure(figsize=(10, 6))
for t in df_grouped["Type"].unique():
    subset = df_grouped[df_grouped["Type"] == t].sort_values(by="Processes")
    plt.errorbar(
        subset["Processes"],
        subset["Max Runtime"],
        yerr=subset["STD"],
        label=t,
        marker="o",
    )
plt.xlabel("Number of Processes")
plt.ylabel("Runtime (s)")
plt.title(f"Runtime vs Number of Processes (Size={problem_size})")
plt.legend(title="Type")
plt.grid()
plt.show()

# Plot Speedup vs Number of Processes
plt.figure(figsize=(10, 6))
for t in df_grouped["Type"].unique():
    subset = df_grouped[df_grouped["Type"] == t].sort_values(by="Processes")
    plt.plot(subset["Processes"], subset["Speedup"], label=t, marker="o")
plt.plot(
    df_grouped["Processes"].unique(),
    df_grouped["Processes"].unique(),
    "k--",
    label="Ideal Speedup",
)
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title(f"Speedup vs Number of Processes (Size={problem_size})")
plt.legend(title="Type")
plt.grid()
plt.show()

# Plot Efficiency vs Number of Processes
plt.figure(figsize=(10, 6))
for t in df_grouped["Type"].unique():
    subset = df_grouped[df_grouped["Type"] == t].sort_values(by="Processes")
    plt.plot(subset["Processes"], subset["Efficiency"], label=t, marker="o")
plt.xlabel("Number of Processes")
plt.ylabel("Efficiency")
plt.title(f"Efficiency vs Number of Processes (Size={problem_size})")
plt.legend(title="Type")
plt.grid()
plt.show()
