import csv
import os
import re

# Directory containing subdirectories with .err files
input_directory = "../myRuns/euler"
output_filename = "perf_stats.csv"

# Pattern to match performance metrics and runtime
pattern_map = {
    "task-clock": r"([\d,]+\.\d+) msec task-clock",
    "context-switches": r"([\d,]+) +context-switches",
    "cpu-migrations": r"([\d,]+) +cpu-migrations",
    "page-faults": r"([\d,]+) +page-faults",
    "cycles": r"([\d,]+) +cycles",
    "instructions": r"([\d,]+) +instructions",
    "branches": r"([\d,]+) +branches",
    "branch-misses": r"([\d,]+) +branch-misses",
    "stalled-cycles-frontend": r"([\d,]+) +stalled-cycles-frontend",
    "stalled-cycles-backend": r"([\d,]+) +stalled-cycles-backend",
    "cache-references": r"([\d,]+) +cache-references",
    "cache-misses": r"([\d,]+) +cache-misses",
    "time-elapsed": r"([\d,]+\.\d+) seconds time elapsed",
}

# Extract column names from pattern map and add filename for file tracking
columns = ["filename"] + list(pattern_map.keys())

# Store all rows of extracted data
all_data = []

# Use os.walk to traverse all subdirectories and their subdirectories
for root, _, files in os.walk(input_directory):
    for filename in files:
        if filename.endswith(".err"):
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, input_directory)
            print(f"Processing file: {relative_path}")

            # Read the contents of the current .err file
            with open(file_path, "r") as f:
                content = f.read()

            # Split the content into individual runs by "==============="
            runs = content.split("===============\n")

            # Extract metrics for each run
            for run in runs:
                if not run.strip():
                    continue  # Skip any empty segments

                # Store metrics for the current run
                run_metrics = {
                    "filename": relative_path
                }  # Add filename to track source file

                # Extract each metric using the corresponding regex
                for metric_name, pattern in pattern_map.items():
                    match = re.search(pattern, run)
                    if match:
                        value = match.group(1).replace(
                            ",", ""
                        )  # Remove commas from numbers
                        run_metrics[metric_name] = (
                            float(value) if "." in value else int(value)
                        )
                    else:
                        run_metrics[metric_name] = (
                            None  # If a metric is missing, store None
                        )

                if run_metrics:
                    all_data.append(run_metrics)

# Write all the extracted data to a CSV file
with open(output_filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()  # Write the header row
    writer.writerows(all_data)  # Write all the rows of extracted data

print(
    f"CSV file '{output_filename}' successfully created with {len(all_data)} total rows."
)
