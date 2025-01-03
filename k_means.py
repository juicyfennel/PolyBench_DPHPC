import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_fast_group(times, plot_dir="clusters", plot_name="cluster_plot.png", n_clusters=4):
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
    # Apply K-means clustering with n_clusters clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(times)

    # Separate times into clusters
    clusters = [times[labels == i].flatten() for i in range(n_clusters)]

    # Find the fastest cluster (cluster with the lowest mean time)
    fastest_cluster_index = np.argmin([np.mean(cluster) for cluster in clusters])
    fastest_cluster = clusters[fastest_cluster_index]

     # Plot all clusters
    plot_dir = "mainplots/clusters/" + plot_dir
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red']
    for i, cluster in enumerate(clusters):
        plt.hist(cluster, bins=15, alpha=0.7, label=f"Cluster {i+1} (Mean = {np.mean(cluster):.2f})", color=colors[i])
        plt.axvline(np.mean(cluster), color=colors[i], linestyle="dashed", linewidth=1)

    plt.xlabel("Execution Time")
    plt.ylabel("Frequency")
    plt.title("Clustering of Execution Times into 4 Groups")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plot_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()
    # print(f"Plot saved to {plot_path}")

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
