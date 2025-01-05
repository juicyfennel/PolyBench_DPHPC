import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Parse .err files and plot cache misses.")
parser.add_argument("directory", type=str, help="Path to the directory containing subdirectories with .err files.")
args = parser.parse_args()

# Path to the folder containing the subdirectories
base_path = args.directory

# Subdirectories to process
subdirs = [
    subdir for subdir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, subdir))
]

# Adjusted regex to match cache miss lines
l1_regex = re.compile(r"^\s*([\d,]+)\s+L1-dcache-load-misses")
l3_regex = re.compile(r"^\s*([\d,]+)\s+cache-misses")

# Dictionary to store parsed results
results = {}

for subdir in subdirs:
    subdir_path = os.path.join(base_path, subdir)
    err_files = [
        f for f in os.listdir(subdir_path) if f.endswith(".err")
    ]  # Get all .err files

    l1_misses = []
    l3_misses = []

    for err_file in err_files:
        with open(os.path.join(subdir_path, err_file), "r") as file:
            content = file.readlines()  # Read line by line

            # Extract L1 and L3 cache misses
            for line in content:
                l1_match = l1_regex.match(line)
                l3_match = l3_regex.match(line)

                if l1_match:
                    l1_misses.append(int(l1_match.group(1).replace(",", "")))
                if l3_match:
                    l3_misses.append(int(l3_match.group(1).replace(",", "")))

    # Store the averages
    results[subdir] = {
        "L1": np.mean(l1_misses) if l1_misses else 0,
        "L3": np.mean(l3_misses) if l3_misses else 0,
    }

# Plot the results
labels = ["L1 Cache Misses", "L3 Cache Misses"]
std_values = [results[subdirs[0]]["L1"], results[subdirs[0]]["L3"]]
blocked_values = [results[subdirs[1]]["L1"], results[subdirs[1]]["L3"]]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots()
bar1 = ax.bar(x - width / 2, std_values, width, label="Standard")
bar2 = ax.bar(x + width / 2, blocked_values, width, label="Blocked")

# Add text for labels, title, and axes
ax.set_xlabel("Cache Levels")
ax.set_ylabel("Cache Misses (Average)")

ax.set_title("Average Cache Misses by Cache Level and Implementation")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
for bar in bar1 + bar2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{int(yval):,}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
