import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

kernels = {
    "gemver": "./kernels/gemver",
    "jacobi-2d": "./kernels/jacobi-2d"
}

inputsizes = {
    "jacobi-2d": {
        "TSTEPS": 500,
        "N": 3362
    },
    "gemver": {
        "N": 10000
    }
}

# Number of processes to test, always include 1 if you want to test the serial version
num_processes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]  # MAX 48 on Euler
# num_processes = [1, 2, 4, 8, 12]  # MAX 12 on Apple M3 Max

num_nodes = [1, 2, 4, 8, 16, 32]  # MAX UNKNOWN on Euler
# num_nodes = [1]  # MAX 1 on Apple M3 Max

# interfaces = {"std": "", "omp": "_omp", "mpi": "_mpi"}
interfaces = {"std": "", "omp": "_omp", "mpi": "_mpi", "omp+mpi": "_omp+mpi"}

# Look into affinity, for now this is fine

# For OMP, total memory you need is (assuming double = 8 bytes) is dominated by matrix A
# Array A = N * N * 8 = 40000 * 40000 * 8 / (1024 * 1024) = 12200 MB
omp_config = {
    "num_threads": num_processes,
    "total_memory": 15000,  # Memory is shared among threads. Guest users can use up to 128GB of data.
    "places": "cores",  # OMP_PLACES: cores (no hyperthreading) | threads (logical threads) | sockets | numa_domains
    "proc_bind": "close"  # spread (spread out around threads/cores/sockets/NUMA domains) | close (as much as possible close to thread/core/same NUMA domains)
}

mpi_config = {
    "num_processes": num_processes,  # Guest users can only use up to 48 processors
    "nodes": num_nodes,
    "total_memory": 60000
}

parser = argparse.ArgumentParser(description="Python script that wraps PolyBench")
parser.add_argument(
    "--kernels",
    type=str,
    nargs="+",
    help="Kernels to run (default = all)",
    default=kernels.keys(),
)
parser.add_argument(
    "--interfaces",
    type=str,
    nargs="+",
    help="Interfaces to run (default = all) (selection: 'std', 'omp', 'mpi')",
    default=["std", "omp", "mpi", "omp+mpi"]
    # default=["std", "omp", "mpi"]
)
parser.add_argument(
    "--no-compile", action="store_true", help="Generate makefiles and compile"
)
parser.add_argument(
    "--num-runs",
    type=int,
    help="Number of times to run the program",
    default=1,
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument(
    "--size",
    type=int,
    help="Input size for the kernel (e.g., 10000, 25000, 40000)",
    default=None,
)
args = parser.parse_args()

if args.size:
    inputsizes["gemver"]["N"] = args.size
    inputsizes["jacobi-2d"]["N"] = args.size

def compile(datasets):
    print(
        "**************************************************\n"
        "Generating makefiles\n"
        "**************************************************"
    )

    lm_flag = ["cholesky", "gramschmidt", "correlation", "jacobi-2d"]
    extra_flags = ""

    for kernel in args.kernels:
        if args.verbose:
            print(kernel)

        rel_root = os.path.relpath(".", kernels[kernel])
        utilities_path = os.path.join(rel_root, "utilities")
        pb_source_path = os.path.join(utilities_path, "polybench.c")

        content = f"include {rel_root}/config.mk\n\n"
        content += f"EXTRA_FLAGS={extra_flags}"

        if kernel in lm_flag:
            content += " -lm"

        content += "\n\n"

        for filename, inputsize_flags in datasets[kernel].items():
            for interface in args.interfaces:
                content += f"{filename}_{interface}: {kernel}{interfaces[interface]}.c {kernel}.h\n"
                content += "\t@mkdir -p bin\n\t${VERBOSE} "
                content += "${MPI_CC}" if "mpi" in interface else "${CC}"
                content += f" -o bin/{filename}{interfaces[interface]} "
                content += f"{kernel}{interfaces[interface]}.c ${{CFLAGS}} -I. -I{utilities_path} "
                content += f"{pb_source_path} {inputsize_flags} ${{EXTRA_FLAGS}}"
                # content += " -lnuma"
                content += " -fopenmp" if interface == "omp" else "" # Only for omp, not for omp+mpi
                content += "\n\n"

        content += "clean:\n"
        for filename, inputsize_flags in datasets[kernel].items():
            for interface in args.interfaces:
                content += f"\t@rm -f bin/{filename}{interfaces[interface]}\n"

        with open(os.path.join(kernels[kernel], "Makefile"), "w") as makefile:
            makefile.write(content)

        make_cmd = ["make", "clean"]
        make_process = subprocess.run(
            make_cmd, cwd=kernels[kernel], capture_output=True, text=True
        )

    print(
        "**************************************************\n"
        "Running make\n"
        "**************************************************"
    )

    for kernel in args.kernels:
        if args.verbose:
            print(kernel)

        make_cmd = ["make"]
        for filename, _ in datasets[kernel].items():
            for interface in args.interfaces:
                make_cmd.append(f"{filename}_{interface}")
                make_process = subprocess.run(
                    make_cmd, cwd=kernels[kernel], capture_output=True, text=True
                )

        if make_process.returncode != 0:
            sys.stderr.write(f"Error running make for kernel {kernel}\n")
            sys.stderr.write(make_process.stderr)
            sys.exit(1)
        if args.verbose:
            sys.stdout.write(make_process.stdout)

def run_local(kernel, interface, p, filename, out_dir_run):
    for i in range(args.num_runs):
        cmd = [os.path.join(".", "bin", f"{filename}{interfaces[interface]}")]
        if "mpi" in interface:
            cmd = ["mpiexec", "-np", str(p)] + cmd
        if "omp" in interface:
            os.environ["OMP_NUM_THREADS"] = str(p)

        with (
            open(os.path.join(out_dir_run, f"{i}.out"), "w") as out,
            open(os.path.join(out_dir_run, f"{i}.err"), "w") as err,
        ):
            driver_process = subprocess.run(
                cmd,
                cwd=kernels[kernel],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Write output to files
            out.write(driver_process.stdout)
            err.write(driver_process.stderr)

            # If verbose, write to sys.stdout and sys.stderr
            if args.verbose:
                sys.stdout.write(driver_process.stdout)
                sys.stderr.write(driver_process.stderr)

        if driver_process.returncode != 0:
            sys.stderr.write(
                f"Error running driver for kernel {filename}{interfaces[interface]}\n"
            )
            sys.stderr.write(driver_process.stderr)
            sys.exit(1)

def run_euler(kernel, interface, p, n, filename, out_dir_run):
    sbatch_dir = os.path.join(kernels[kernel], "sbatch")
    os.makedirs(sbatch_dir, exist_ok=True)

    # Prepare paths and job name
    binary_path = os.path.join(
        kernels[kernel], "bin", f"{filename}{interfaces[interface]}"
    )
    sbatch_file = os.path.join(sbatch_dir, f"{filename}{interfaces[interface]}.sbatch")

    content = "#!/bin/bash\n"
    content += "#SBATCH --time=00:04:00\n"
    content += f"#SBATCH -o ./{out_dir_run}/%j.out\n"
    content += f"#SBATCH -e ./{out_dir_run}/%j.err\n"
    # content += "#SBATCH --mem-bind=local\n"

    nodelist = [f"eu-g9-0{i+1:02}-{j+1}" for i in range(48) for j in range(4)]
    # nodelist = ["eu-g9-024-1", "eu-g9-024-2", "eu-g9-024-3", "eu-g9-024-4"]

    # content += "#SBATCH --nodelist=eu-g9-028-4\n"
    content += f"#SBATCH --nodelist={','.join(nodelist)}\n"

    if "mpi" in interface:
        content += f"#SBATCH --nodes={n}\n"
        content += f"#SBATCH --ntasks={p}\n"
        content += f"#SBATCH --mem-per-cpu={int(mpi_config['total_memory']/p)}\n\n"
        content += "#SBATCH -C ib\n\n"
    elif interface == "omp":
        content += "#SBATCH --nodes=1\n"
        content += "#SBATCH --ntasks=1\n"
        content += f"#SBATCH --cpus-per-task={p}\n"
        content += f"#SBATCH --mem-per-cpu={int(omp_config['total_memory']/p)}\n\n"
    else:
        content += "#SBATCH --nodes=1\n"
        content += "#SBATCH --ntasks=1\n"
        content += f"#SBATCH --mem-per-cpu={omp_config['total_memory']}\n\n"
        
    if "omp" in interface:
        content += "export OMP_DISPLAY_ENV=TRUE\n"
        content += f"export OMP_NUM_THREADS={p}\n"
        content += f"export OMP_PLACES={omp_config['places']}\n"
        content += f"export OMP_PROC_BIND={omp_config['proc_bind']}\n\n"

    content += (
        "module load stack/2024-06 openmpi/4.1.6 openblas/0.3.24 2> /dev/null\n\n"
    )

    content += f"for i in {{1..{args.num_runs}}}; do\n"
    if "mpi" in interface:
        content += "srun "

    content += "perf stat " + binary_path + "\n"
    content += 'echo "==============="\n'  # stdout
    content += 'echo "===============" >&2\n'  # stderr
    content += "done\n\n"

    content += f"srun hostname > ./{out_dir_run}/hostname.txt\n"

    # content += (
    #     f"for i in {{1..{args.num_runs}}}; do\n"
    #     "    perf stat -e cycles,instructions,cache-misses,context-switches,cpu-migrations,dTLB-load-misses,iTLB-load-misses "
    #     + binary_path
    #     + "\n"
    #     "done\n"
    # )
    # content += f"\nsrun hostname > ./{hostname_dir}/${{SLURM_JOB_ID}}.txt\n"

    with open(sbatch_file, "w") as file:
        file.write(content)

        if args.verbose:
            print(f"Sbatch file generated: {sbatch_file}")

    # Submit sbatch files
    # for i in range(args.num_runs):
    submission = subprocess.run(
        [
            "sbatch",
            f"--job-name={filename}{interfaces[interface]}_np{p}",
            sbatch_file,
        ],
        capture_output=True,
        text=True,
    )
    print(submission.stdout)
    if submission.returncode != 0:
        print(f"Error submitting job: {submission.stderr}")
        sys.exit(1)

def run(datasets, on_euler):
    print(
        "**************************************************\n"
        f"Running Kernels {'locally' if not on_euler else 'on Euler'}\n"
        "**************************************************"
    )

    date = datetime.now().strftime("%Y_%m_%d__%H-%M-%S")
    output_dir = os.path.join("outputs/%s" % ("euler" if on_euler else "local"), date)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "inputsizes.json"), "w") as f:
        json.dump(inputsizes, f, indent=4)

    for kernel in args.kernels:
        if args.verbose:
            print("Running kernel: %s" % kernel)
        for filename, _ in datasets[kernel].items():
            for interface in args.interfaces:
                if args.verbose:
                    print("-Running interface: %s" % interface)
                for p in num_processes:
                    if interface == "std" and p != 1 or interface != "std" and p == 1:
                        continue
                    if args.verbose:
                        print("--Running processes: %s" % p)
                    for n in num_nodes:
                        if interface == "mpi" and n == 1 or interface != "mpi" and n != 1:
                            continue
                        if args.verbose:
                            print("---Running num_nodes: %s" % n)

                        out_dir_run = os.path.join(
                            # np = number of processes, nn = number of nodes
                            output_dir, f"{filename}_np_{p}_nn_{n}_{interface}"
                        )
                        os.makedirs(out_dir_run, exist_ok=True)
                        
                        with open(
                            os.path.join(output_dir, f"{interface}.json"),
                            "w",
                        ) as f:
                            json.dump(mpi_config, f, indent=4)

                        # Local
                        if on_euler:
                            run_euler(
                                kernel,
                                interface,
                                p,
                                n,
                                filename,
                                out_dir_run,
                            )
                        # Euler
                        else:
                            run_local(
                                kernel,
                                interface,
                                p,
                                filename,
                                out_dir_run,
                            )


def main():
    datasets = {}
    # Generate necessary compiler flags based on datasets to test
    for kernel, sizes in inputsizes.items():
        datasets[kernel] = {}
        filename = f"{kernel}"
        flags = ""
        for inputsize_key in sizes:
            inputsize_value = sizes[inputsize_key]
            filename += f"_{inputsize_key}_{str(inputsize_value)}"
            if flags != "":
                flags += " "
            flags += f"-D{inputsize_key}={str(inputsize_value)}"
        datasets[kernel][filename] = flags

    # Detect cluster
    cwd = os.getcwd()
    on_euler = cwd.startswith("/cluster/")

    if not args.no_compile:
        compile(datasets)
    # return

    if args.kernels:
        run(datasets, on_euler)


# pass
if __name__ == "__main__":
    main()
