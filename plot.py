import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory.")
    # Sort files by modified time and get the latest one
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)

def plot_kernel_results(kernel):
    # Define the path to the kernel's measurements directory
    kernel_dir = os.path.join("measurements", kernel)
    if not os.path.exists(kernel_dir):
        print(f"Directory {kernel_dir} does not exist.")
        return

    # Load the latest JSON file
    latest_file = get_latest_file(kernel_dir)
    with open(latest_file, "r") as f:
        data = json.load(f)

    # Calculate averages for each dataset size
    dataset_sizes = list(data.keys())
    short_labels = [size.split('_')[0] for size in dataset_sizes]  # Use short labels like MINI, SMALL, etc.
    labels = [f"{kernel}", f"{kernel}_omp", f"{kernel}_mpi"]  # Updated to match the keys in your JSON file
    averages = {label: [] for label in labels}

    for size in dataset_sizes:
        for label in labels:
            if label in data[size]:
                times = data[size][label]
                avg_time = np.mean(times)
                averages[label].append(avg_time)
            else:
                averages[label].append(None)  # Append None if the label is not in the data

    # Plotting
    x = short_labels
    for label in labels:
        plt.plot(x, averages[label], marker='o', label=label)

    plt.xlabel("Dataset Size")
    plt.ylabel("Average Execution Time (s)")
    plt.title(f"Average Execution Time for {kernel}")
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py <kernel_name>")
        sys.exit(1)

    kernel_name = sys.argv[1]
    plot_kernel_results(kernel_name)
