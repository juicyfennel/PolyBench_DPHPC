import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_fast_group(times, plot_dir="mainplots/clusters", plot_name="cluster_plot.png"):
    """
    Perform clustering on execution times to separate fast and slow groups.

    Parameters:
        times (list or np.array): A list of execution times.
        plot_dir (str): Directory to save the plot.
        plot_name (str): Name of the plot file.

    Returns:
        list: The fast group of execution times.
    """
    # Convert times to a numpy array
    times = np.array(times).reshape(-1, 1)

    # Apply K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(times)

    # Separate times into "fast" and "slow" clusters
    fast_cluster = times[labels == 0].flatten()
    slow_cluster = times[labels == 1].flatten()

    # Determine which cluster is "fast" and which is "slow"
    if np.mean(fast_cluster) > np.mean(slow_cluster):
        fast_cluster, slow_cluster = slow_cluster, fast_cluster

    # Plot the clusters
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(fast_cluster, bins=15, alpha=0.7, label="Fast Cluster", color="blue")
    plt.hist(slow_cluster, bins=15, alpha=0.7, label="Slow Cluster", color="orange")
    plt.axvline(np.mean(fast_cluster), color="blue", linestyle="dashed", linewidth=1, label="Fast Mean")
    plt.axvline(np.mean(slow_cluster), color="orange", linestyle="dashed", linewidth=1, label="Slow Mean")
    plt.xlabel("Execution Time")
    plt.ylabel("Frequency")
    plt.title("Clustering of Execution Times")
    plt.legend()
    plt.tight_layout()


    plt.close()
    # Return the fast group as a list
    return list(fast_cluster)

# Example usage
if __name__ == "__main__":
    # Sample times for testing
    sample_times = [
        18.753472, 18.525560, 18.690504, 18.394148, 18.075039, 17.989079, 17.498969, 16.978550, 17.102204, 16.843035,
        17.511884, 17.544597, 17.490991, 17.590224, 17.450545, 17.543477, 17.512372, 16.596638, 15.467317, 15.779492
    ]

    # Call the function
    fast_group = get_fast_group(sample_times, plot_name="example_plot.png")
    print("Fast Group:", fast_group)
