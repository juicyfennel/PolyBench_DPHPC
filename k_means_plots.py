import os
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Regex to extract times from the .out files
time_regex = re.compile(r"Time:\s+([\d.]+)")

def extract_times_from_files(directory):
    """Extract times from all .out files in the given directory and its subdirectories."""
    times_by_dir = {}  # Store times for each directory
    for subdir, _, files in os.walk(directory):
        subdir_times = []
        for file in files:
            if file.endswith(".out"):
                with open(os.path.join(subdir, file), "r") as f:
                    content = f.read()
                    # Extract all times from the file
                    matches = time_regex.findall(content)
                    subdir_times.extend([float(time) for time in matches])
        if subdir_times:
            times_by_dir[subdir] = subdir_times
    return times_by_dir

def cluster_and_plot(times, subdir_name, output_dir):
    """Cluster execution times into 4 groups and save the results as a plot."""
    # Convert times to a numpy array
    times = np.array(times).reshape(-1, 1)

    # Apply K-means clustering with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=0)
    labels = kmeans.fit_predict(times)

    # Separate times into clusters
    clusters = [times[labels == i].flatten() for i in range(4)]

    # Find the fastest cluster (cluster with the lowest mean time)
    fastest_cluster_index = np.argmin([np.mean(cluster) for cluster in clusters])
    fastest_cluster = clusters[fastest_cluster_index]

    # Print results
    print(f"\nResults for {subdir_name}:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} (Mean Time = {np.mean(cluster):.2f}):", sorted(cluster))
    print("Fastest Cluster:", sorted(fastest_cluster))

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red']
    for i, cluster in enumerate(clusters):
        plt.hist(cluster, bins=15, alpha=0.7, label=f"Cluster {i + 1} (Mean = {np.mean(cluster):.2f})", color=colors[i])
        plt.axvline(np.mean(cluster), color=colors[i], linestyle="dashed", linewidth=1)

    plt.xlabel("Execution Time")
    plt.ylabel("Frequency")
    plt.title(f"Clustering of Execution Times for {subdir_name}")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{subdir_name.replace(os.sep, '_')}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

# Main script
if __name__ == "__main__":
    directory = input("Enter the root directory containing the .out files: ").strip()
    output_dir = "main_plot/clusters"  # Directory to save plots
    all_times = extract_times_from_files(directory)

    for subdir, times in all_times.items():
        relative_subdir = os.path.relpath(subdir, directory)
        cluster_and_plot(times, relative_subdir, output_dir)
