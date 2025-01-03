import os
import shutil
base_dir = "outputs/euler"
# Function to create the combined directory structure and concatenate files
def combine_out_files(parent_dirs, combined_dir):
    # Create the combined parent directory if it doesn't exist
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Get the union of all subdirectories across parent directories
    all_subdirs = set()
    for parent in parent_dirs:
        for subdir in os.listdir(parent):
            full_path = os.path.join(parent, subdir)
            if os.path.isdir(full_path):
                all_subdirs.add(subdir)

    # Create an error log file
    error_log_path = os.path.join(combined_dir, "missing_files.err")
    with open(error_log_path, "w") as error_log:
        # Process each subdirectory
        for subdir in all_subdirs:
            combined_subdir_path = os.path.join(combined_dir, subdir)
            os.makedirs(combined_subdir_path, exist_ok=True)

            # Collect all *.out files from this subdirectory across parent directories
            combined_content = []
            merged_file_count = 0
            for parent in parent_dirs:
                subdir_path = os.path.join(parent, subdir)
                if os.path.exists(subdir_path):
                    out_files = [file for file in os.listdir(subdir_path) if file.endswith(".out")]
                    if out_files:
                        for file in out_files:
                            file_path = os.path.join(subdir_path, file)
                            with open(file_path, "r") as f:
                                combined_content.append(f.read())
                                merged_file_count += 1
                    else:
                        # Log missing .out files
                        error_log.write(f"Missing .out files in: {subdir_path}\n")
                else:
                    # Log missing subdirectory
                    error_log.write(f"Missing subdirectory: {subdir_path}\n")

            # Write the concatenated content to a single file in the combined directory
            if combined_content:
                output_file_name = f"{merged_file_count}_files_merged.out"
                output_file_path = os.path.join(combined_subdir_path, output_file_name)
                with open(output_file_path, "w") as output_file:
                    output_file.write("\n".join(combined_content))

parent_dirs = []
# Example usage
# parent_dirs = ["outputs/euler/2024_12_30__20-08-58",
#                 "outputs/euler/2024_12_30__20-08-48",
#                   "outputs/euler/2024_12_30__20-08-43",
#                   "outputs/euler/2024_12_30__20-08-36",
#                     "outputs/euler/2024_12_30__20-08-30",
#                     "outputs/euler/2024_12_30__20-08-28",
#                     "outputs/euler/2024_12_30__20-08-22",
#                     "outputs/euler/2024_12_30__20-08-16",
#                     "outputs/euler/2024_12_30__20-08-13",
#                     "outputs/euler/2024_12_30__20-08-06",
#                     "outputs/euler/2024_12_30__20-08-00",
#                     "outputs/euler/2024_12_30__20-07-58",
#                     "outputs/euler/2024_12_30__20-07-53",
#                     "outputs/euler/2024_12_30__20-07-48",
#                     "outputs/euler/2024_12_30__20-07-45",
#                     "outputs/euler/2024_12_30__20-07-40",
#                     "outputs/euler/2024_12_30__20-07-32",
#                     "outputs/euler/2024_12_30__20-07-27",
#                     "outputs/euler/2024_12_30__20-07-24",
#                     "outputs/euler/2024_12_30__20-07-19",
#                     "outputs/euler/2024_12_30__20-07-13",
#                     "outputs/euler/2024_12_30__20-07-10",
#                     "outputs/euler/2024_12_30__20-07-03",
#                     "outputs/euler/2024_12_30__20-06-58",
#                     "outputs/euler/2024_12_30__20-06-52",
#                     "outputs/euler/2024_12_30__20-06-34",
#                     "outputs/euler/2024_12_30__20-06-22",
#                     "outputs/euler/2024_12_30__20-06-00",
#                     "outputs/euler/2024_12_30__20-05-38",
#                     "outputs/euler/2024_12_30__20-05-08",
#                     "outputs/euler/2024_12_30__20-04-46",
#                     "outputs/euler/2024_12_30__20-04-23",
#                     "outputs/euler/2024_12_30__20-03-59",
#                     "outputs/euler/2024_12_30__20-03-30",
#                     "outputs/euler/2024_12_30__20-03-25",
#                     "outputs/euler/2024_12_30__20-03-18",
#                     "outputs/euler/2024_12_30__20-03-11",
#                     "outputs/euler/2024_12_30__20-03-02",
#                     ]  # List of parent directories

patterns = ["2024_12_30_20","2024_12_31", "2025"]
# patterns = ["2024_12_20__13-00-00","2024_12_20__08-29-26","2024_12_19__2","2024_12_19__15-13-27","2024_12_19__14-01-25","2024_12_19__13"]  # for 20k iterations 

# Collect directories matching the patterns
for entry in os.listdir(base_dir):
    if any(entry.startswith(pattern) for pattern in patterns):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):  # Ensure it's a directory
            parent_dirs.append(full_path)
parent_dirs = list(set(parent_dirs))  # Remove duplicates
parent_dirs.sort()  # Sort the list
combined_dir = "myRuns/merged/2025_01_03__10-00-00--WeakScalingData_ALL"    # Name of the combined output directory
combine_out_files(parent_dirs, combined_dir)
