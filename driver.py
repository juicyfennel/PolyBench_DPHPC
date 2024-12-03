import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

kernels = {
    "2mm": "./linear-algebra/kernels/2mm",
    "3mm": "./linear-algebra/kernels/3mm",
    "atax": "./linear-algebra/kernels/atax",
    "bicg": "./linear-algebra/kernels/bicg",
    "doitgen": "./linear-algebra/kernels/doitgen",
    "mvt": "./linear-algebra/kernels/mvt",
    "gemm": "./linear-algebra/blas/gemm",
    "gemver": "./linear-algebra/blas/gemver",
    "gesummv": "./linear-algebra/blas/gesummv",
    "symm": "./linear-algebra/blas/symm",
    "syr2k": "./linear-algebra/blas/syr2k",
    "syrk": "./linear-algebra/blas/syrk",
    "trmm": "./linear-algebra/blas/trmm",
    "cholesky": "./linear-algebra/solvers/cholesky",
    "durbin": "./linear-algebra/solvers/durbin",
    "gramschmidt": "./linear-algebra/solvers/gramschmidt",
    "lu": "./linear-algebra/solvers/lu",
    "ludcmp": "./linear-algebra/solvers/ludcmp",
    "trisolv": "./linear-algebra/solvers/trisolv",
    "correlation": "./datamining/correlation",
    "covariance": "./datamining/covariance",
    "deriche": "./medley/deriche",
    "floyd-warshall": "./medley/floyd-warshall",
    "nussinov": "./medley/nussinov",
    "adi": "./stencils/adi",
    "fdtd-2d": "./stencils/fdtd-2d",
    "heat-3d": "./stencils/heat-3d",
    "jacobi-1d": "./stencils/jacobi-1d",
    "jacobi-2d": "./stencils/jacobi-2d",
    "seidel-2d": "./stencils/seidel-2d",
}

inputsizes = {
    "jacobi-2d": [
        {"TSTEPS": 100, "N": 1000},
        {"TSTEPS": 500, "N": 2000},
        {"TSTEPS": 1000, "N": 3000},
    ],
    "gemver": [{"N": 1000}],
}


interfaces = {"std": "", "omp": "_omp", "mpi": "_mpi"}

# Look into affinity, for now this is fine

omp_config = {
    "num_threads": 8,
    "mem_per_thread": 8000,  # Guest users can only use up to 128GB of data
    "places": "cores",  # OMP_PLACES: cores (no hyperthreading) | threads (logical threads) | sockets | numa_domains
    "proc_bind": "close",  # spread (spread out around threads/cores/sockets/NUMA domains) | close (as much as possible close to thread/core/same NUMA domains)
}

mpi_config = {
    "num_processes": 10,  # Guest users can only use up to 48 processors
    "nodes": 3,
    "mem_per_process": 400,
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
    default=["std", "omp", "mpi"],
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

args = parser.parse_args()


# # compile
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
        make_cmd = ["make", "clean"]
        make_process = subprocess.run(
            make_cmd, cwd=kernels[kernel], capture_output=True, text=True
        )

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
                content += "${MPI_CC}" if interface == "mpi" else "${CC}"
                content += f" -o bin/{filename}{interfaces[interface]} "
                content += f"{kernel}{interfaces[interface]}.c ${{CFLAGS}} -I. -I{utilities_path} "
                content += f"{pb_source_path} {inputsize_flags} ${{EXTRA_FLAGS}}"
                content += " -fopenmp" if interface == "omp" else ""
                content += "\n\n"

        content += "clean:\n\t@ rm -f bin/*\n\n"

        with open(os.path.join(kernels[kernel], "Makefile"), "w") as makefile:
            makefile.write(content)

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


def run_local(kernel, interface, filename, out_dir, err_dir):
    for i in range(args.num_runs):
        cmd = [os.path.join(".", "bin", f"{filename}{interfaces[interface]}")]
        if interface == "mpi":
            cmd = ["mpiexec", "-np", str(mpi_config["num_processes"])] + cmd
        elif interface == "omp":
            os.environ["OMP_NUM_THREADS"] = str(omp_config["num_threads"])

        with (
            open(os.path.join(out_dir, f"{i}.out"), "w") as out,
            open(os.path.join(err_dir, f"{i}.err"), "w") as err,
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


def run_euler(kernel, interface, filename, out_dir, err_dir):
    # date = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")

    sbatch_dir = os.path.join(kernels[kernel], "sbatch")
    os.makedirs(sbatch_dir, exist_ok=True)

    # Prepare paths and job name
    binary_path = os.path.join(
        kernels[kernel], "bin", f"{filename}{interfaces[interface]}"
    )
    sbatch_file = os.path.join(sbatch_dir, f"{filename}{interfaces[interface]}.sbatch")

    content = "#!/bin/bash\n"
    content += "#SBATCH --time=00:02:00\n"
    content += f"#SBATCH -o ./{out_dir}/%j.out\n"
    content += f"#SBATCH -e ./{err_dir}/%j.err\n"

    if interface == "mpi":
        content += f"#SBATCH --nodes={mpi_config['nodes']}\n"
        content += f"#SBATCH --ntasks={mpi_config['num_processes']}\n"
        content += f"#SBATCH --mem-per-cpu={mpi_config['mem_per_process']}\n"
        content += "#SBATCH -C ib\n\n"

    elif interface == "omp":
        content += "#SBATCH --nodes=1\n"
        content += "#SBATCH --ntasks=1\n"
        content += f"#SBATCH --cpus-per-task={omp_config['num_threads']}\n"
        content += f"#SBATCH --mem-per-cpu={omp_config['mem_per_thread']}\n\n"

        content += f"export OMP_NUM_THREADS={omp_config['num_threads']}\n"
        content += f"export OMP_PLACES={omp_config['places']}\n"
        content += f"export OMP_PROC_BIND={omp_config['proc_bind']}\n\n"
    else:
        content += "#SBATCH --nodes=1\n"
        content += "#SBATCH --ntasks=1\n"
        content += "#SBATCH --mem-per-cpu=20000\n\n"

    content += (
        "module load stack/2024-06 openmpi/4.1.6 openblas/0.3.24 2> /dev/null\n\n"
    )

    if interface == "mpi":
        content += "srun "

    content += binary_path

    with open(sbatch_file, "w") as file:
        file.write(content)

        if args.verbose:
            print(f"Sbatch file generated: {sbatch_file}")

    # Submit sbatch files
    for _ in range(args.num_runs):
        submission = subprocess.run(
            ["sbatch", sbatch_file], capture_output=True, text=True
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

    for kernel in args.kernels:
        if args.verbose:
            print(kernel)
        for filename, _ in datasets[kernel].items():
            for interface in args.interfaces:
                if args.verbose:
                    print(interface)

                output_dir = os.path.join("outputs", date, f"{kernel}_{interface}")
                out_dir = os.path.join(output_dir, "out")
                err_dir = os.path.join(output_dir, "err")

                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(err_dir, exist_ok=True)

                if interface == "omp":
                    with open(os.path.join(output_dir, "omp.json"), "w") as f:
                        json.dump(omp_config, f, indent=4)

                if interface == "mpi":
                    with open(os.path.join(output_dir, "mpi.json"), "w") as f:
                        json.dump(mpi_config, f, indent=4)

                # Local
                if on_euler:
                    run_euler(kernel, interface, filename, out_dir, err_dir)
                # Euler
                else:
                    run_local(kernel, interface, filename, out_dir, err_dir)
                    # run_local()


def main():
    datasets = {}

    # Generate necessary compiler flags based on datasets to test
    for kernel, sizes in inputsizes.items():
        datasets[kernel] = {}
        for inputsize in sizes:
            filename = f"{kernel}_"
            flags = ""
            for flag, val in inputsize.items():
                filename += f"{flag}_{str(val)}_"
                flags += f"-D{flag}={str(val)} "
            filename = filename[:-1]
            datasets[kernel][filename] = flags

    # Detect cluster
    cwd = os.getcwd()
    on_euler = cwd.startswith("/cluster/")

    if not args.no_compile:
        compile(datasets)

    run(datasets, on_euler)


# pass
if __name__ == "__main__":
    main()
