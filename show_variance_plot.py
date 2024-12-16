#python show_variance_plot.py  local_data.txt euler_data_std.txt euler_data_separate_node.txt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def load_data(file_path):
    """Loads the data from a CSV-like file and returns a DataFrame."""
    columns = ["kernel", "size", "processes", "interface", "time", "variance"]
    data = pd.read_csv(file_path, header=None, names=columns, sep=r'[\s,]+', engine='python')
    return data

def load_third_file(file_path):
    """Loads the third file with raw numbers and calculates statistics."""
    # Load numbers into a numpy array
    values = np.loadtxt(file_path)
    # Assume fixed sizes and group into chunks of 10
    sizes = [10000, 25000, 40000]
    grouped_stats = []
    for i, size in enumerate(sizes):
        chunk = values[i * 10: (i + 1) * 10]  # Slice for each size
        mean = np.mean(chunk)
        std = np.std(chunk)
        grouped_stats.append((size, mean, std))
    # Convert to a DataFrame for consistency
    return pd.DataFrame(grouped_stats, columns=["size", "mean", "std"])

def plot_variability(file1, file2, file3, output_file="runtime_variability.png"):
    """Plots runtime variability using shaded regions for datasets in file1, file2, and file3."""
    # Load data
    data1 = load_data(file1)
    data2 = load_data(file2)
    stats3 = load_third_file(file3)

    # Group by size to calculate mean and std for file1 and file2
    stats1 = data1.groupby("size")["time"].agg(['mean', 'std']).reset_index()
    stats2 = data2.groupby("size")["time"].agg(['mean', 'std']).reset_index()

    # Plot
    plt.close('all')  # Close all previous figures

    plt.figure(figsize=(10, 6))
    
    # Plot Local (file1)
    plt.plot(stats1["size"], stats1["mean"], label="Local (mean)", marker='o', linestyle='-')
    plt.fill_between(stats1["size"], 
                     stats1["mean"] - stats1["std"], 
                     stats1["mean"] + stats1["std"], 
                     color='blue', alpha=0.2, label="Local (±std)")

    # Plot Euler (file2)
    plt.plot(stats2["size"], stats2["mean"], label="Euler (mean)", marker='s', linestyle='--')
    plt.fill_between(stats2["size"], 
                     stats2["mean"] - stats2["std"], 
                     stats2["mean"] + stats2["std"], 
                     color='red', alpha=0.2, label="Euler (±std)")

    # Plot Third File (file3)
    plt.plot(stats3["size"], stats3["mean"], label="Euler- separate nodes (mean)", marker='^', linestyle='-.')
    plt.fill_between(stats3["size"], 
                     stats3["mean"] - stats3["std"], 
                     stats3["mean"] + stats3["std"], 
                     color='green', alpha=0.2, label="Euler- separate nodes (±std)")

    plt.xlabel("Data Size (N)", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Data Size with Variability", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save plot to a file
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot runtime variability for three datasets.")
    parser.add_argument("file1", help="Path to the first dataset file (e.g., Local)")
    parser.add_argument("file2", help="Path to the second dataset file (e.g., Euler)")
    parser.add_argument("file3", help="Path to the third dataset file (unformatted numbers)")
    parser.add_argument("--output", default="runtime_variability.png", help="Output file for the plot")
    args = parser.parse_args()

    plot_variability(args.file1, args.file2, args.file3, args.output)
