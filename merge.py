import os
import shutil

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
            combined_file_names = []
            for parent in parent_dirs:
                subdir_path = os.path.join(parent, subdir)
                if os.path.exists(subdir_path):
                    out_files = [file for file in os.listdir(subdir_path) if file.endswith(".out")]
                    if out_files:
                        for file in out_files:
                            file_path = os.path.join(subdir_path, file)
                            with open(file_path, "r") as f:
                                combined_content.append(f.read())
                                combined_file_names.append(file.split('.')[0])
                    else:
                        # Log missing .out files
                        error_log.write(f"Missing .out files in: {subdir_path}\n")
                else:
                    # Log missing subdirectory
                    error_log.write(f"Missing subdirectory: {subdir_path}\n")

            # Write the concatenated content to a single file in the combined directory
            if combined_content:
                output_file_name = "_".join(combined_file_names) + ".out"
                output_file_path = os.path.join(combined_subdir_path, output_file_name)
                with open(output_file_path, "w") as output_file:
                    output_file.write("\n".join(combined_content))

# Example usage
parent_dirs = ["outputs/euler/2024_12_20__09-57-12",
                "outputs/euler/2024_12_20__10-42-19",
                  "outputs/euler/2024_12_20__11-10-03",
                  "outputs/euler/2024_12_20__11-42-54",
                    "outputs/euler/theas/2024_12_20__09-50-30",
                    "outputs/euler/theas/2024_12_20__10-18-23",
                    "outputs/euler/theas/2024_12_20__11-10-23",
                    "outputs/euler/theas/2024_12_20__11-43-05",
                    "outputs/euler/2024_12_20__12-25-07"
                    ]  # List of parent directories
combined_dir = "outputs/euler/2024_12_20__13-00-00"    # Name of the combined output directory
combine_out_files(parent_dirs, combined_dir)
