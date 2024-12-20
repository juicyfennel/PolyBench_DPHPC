import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Plot runtime, speedup, and efficiency
def plot_metrics(df, size, output_dir):
    interfaces = df['Type'].unique()
    x_label = 'Number of Processes'

    # Runtime plot
    plt.figure()
    for iface in interfaces:
        iface_data = df[df['Type'] == iface].sort_values(by='Processes')
        plt.plot(iface_data['Processes'], iface_data['Mean Runtime'], label=iface, marker="o")
        plt.fill_between(iface_data['Processes'],
                         iface_data['Mean Runtime'] - iface_data['STD'],
                         iface_data['Mean Runtime'] + iface_data['STD'], alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel('Runtime (s)')
    plt.title(f'Runtime vs {x_label} (Size {size})')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig(f"{output_dir}/runtime_vs_processes.png")
    plt.close()

    # Speedup plot
    plt.figure()
    for iface in interfaces:
        iface_data = df[df['Type'] == iface].sort_values(by='Processes')
        plt.plot(iface_data['Processes'], iface_data['Speedup'], label=iface, marker="o")
    plt.xlabel(x_label)
    plt.ylabel('Speedup')
    plt.title(f'Speedup vs {x_label} (Size {size})')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/speedup_vs_processes.png")
    plt.close()

    # Efficiency plot
    plt.figure()
    for iface in interfaces:
        iface_data = df[df['Type'] == iface].sort_values(by='Processes')
        plt.plot(iface_data['Processes'], iface_data['Efficiency'], label=iface, marker="o")
    plt.xlabel(x_label)
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency vs {x_label} (Size {size})')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/efficiency_vs_processes.png")
    plt.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Plot runtime, speedup, and efficiency from a CSV file.")
    parser.add_argument('--file', type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    input_file = args.file
    output_dir_base = "runtime_speedup_efficiency"

    # Read the data
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    required_columns = ['Size', 'Processes', 'Type', 'Mean Runtime', 'STD']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing from the CSV.")
            exit(1)

    # Calculate speedup and efficiency
    reference_runtime = df[(df['Processes'] == 1) & (df['Type'] == 'mpi_gather')]
    if not reference_runtime.empty:
        reference_runtime = reference_runtime['Mean Runtime'].iloc[0]
    else:
        reference_runtime = 31.412742499999997  # hardcoded for gemver 
    df['Speedup'] = reference_runtime / df['Mean Runtime']
    df['Efficiency'] = df['Speedup'] / df['Processes']

    # Extract size and create output directory
    size = df['Size'].iloc[0]
    output_dir = os.path.join(output_dir_base, "size_"+str(size))
    os.makedirs(output_dir, exist_ok=True)

    # Plot metrics
    plot_metrics(df, size, output_dir)

if __name__ == "__main__":
    main()
