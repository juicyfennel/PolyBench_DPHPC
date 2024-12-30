import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_fast_group(times, plot_dir="mainplots/clusters", plot_name="cluster_plot.png"):
    """
    Perform clustering on execution times to separate them into 4 groups and return the fastest group.

    Parameters:
        times (list or np.array): A list of execution times.
        plot_dir (str): Directory to save the plot.
        plot_name (str): Name of the plot file.

    Returns:
        list: The fastest group of execution times.
    """
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

    return list(fastest_cluster)

# Example usage
if __name__ == "__main__":
    # Sample times for testing
    sample_times = [
        18.753472, 18.525560, 18.690504, 18.394148, 18.075039, 17.989079, 17.498969, 16.978550, 17.102204, 16.843035,
        17.511884, 17.544597, 17.490991, 17.590224, 17.450545, 17.543477, 17.512372, 16.596638, 15.467317, 15.779492
    ]

    # Call the function
    fastest_group = get_fast_group(sample_times, plot_name="example_plot.png")
    print("Fastest Group:", fastest_group)
