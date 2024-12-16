import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

def load_data(file_path):
    """Loads data from a file and returns it as a list of floats."""
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data

def plot_distribution(data_file, output_file="completion_time_distribution.png"):
    """Plots the histogram and distribution of completion times with non-overlapping annotations."""
    # Load data
    data = load_data(data_file)

    # Compute statistical metrics
    min_value = np.min(data)
    max_value = np.max(data)
    median = np.median(data)
    mean = np.mean(data)
    quantile_95 = np.percentile(data, 95)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, color="white", edgecolor="black", density=True, alpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add a kernel density estimate (smooth curve)
    sns.kdeplot(data, color="blue", linewidth=1.5, label="Density")

    # Annotations for statistics with adjusted vertical positions
    offset = 0.05  # Vertical offset to prevent overlap
    plt.axvline(min_value, color="green", linestyle="--", linewidth=1, label="Min")
    plt.text(min_value - 0.5, 0.5 + offset, f"Min\n{min_value:.2f}", color="green", ha="center")

    plt.axvline(median, color="blue", linestyle="-", linewidth=2, label="Median")
    plt.text(median + 0.5, 0.4 + 2 * offset, f"Median\n{median:.2f}", color="blue", ha="center")

    plt.axvline(mean, color="purple", linestyle="-.", linewidth=2, label="Mean")
    plt.text(mean - 0.5, 0.3 + 3 * offset, f"Mean\n{mean:.2f}", color="purple", ha="center")

    plt.axvline(quantile_95, color="red", linestyle=":", linewidth=2, label="95% Quantile")
    plt.text(quantile_95 - 0.5, 0.2 + 4 * offset, f"95% Quantile\n{quantile_95:.2f}", color="red", ha="center")

    plt.axvline(max_value, color="orange", linestyle="--", linewidth=1, label="Max")
    plt.text(max_value + 0.5, 0.1 + 5 * offset, f"Max\n{max_value:.2f}", color="orange", ha="center")

    # Titles and labels
    plt.title("Distribution of Completion Times")
    plt.xlabel("Completion Time (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot completion time distribution with annotations.")
    parser.add_argument("data_file", help="Path to the file containing completion times.")
    parser.add_argument("--output", default="completion_time_distribution.png", help="Output file for the plot.")
    args = parser.parse_args()

    plot_distribution(args.data_file, args.output)
