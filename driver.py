import argparse
import json
import os
import re
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

datasets = {
    "mini": "-DMINI_DATASET",
    "small": "-DSMALL_DATASET",
    "medium": "-DMEDIUM_DATASET",
    "large": "-DLARGE_DATASET",
    "extralarge": "-DEXTRALARGE_DATASET",
}

interfaces = {"std": "", "omp": "_omp", "mpi": "_mpi"}

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
parser.add_argument("--no-gen", action="store_true", help="Do not regenerate makefiles")
parser.add_argument("--no-make", action="store_true", help="Do not run make")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("--num-runs", type=int, help="Number of runs", default=1)
parser.add_argument(
    "--validate", action="store_true", help="Validate results against reference"
)
parser.add_argument(
    "--input-size",
    type=str,
    nargs="+",
    help="Input size for kernels (default = medium) (selection: 'mini', 'small', 'medium', 'large', 'extralarge')",
    default=["medium"],
)
parser.add_argument(
    "--mpi-processes",
    type=int,
    help="Number of MPI processes to run MPI with (default=4)",
    default=4,
)

args = parser.parse_args()

# Make sure the standard kernel is ran first if validation is enabled
if args.validate:
    if "std" in args.interfaces:
        args.interfaces.remove("std")
    args.interfaces.insert(0, "std")

if not args.no_gen:
    print(
        "**************************************************\n"
        "Generating makefiles\n"
        "**************************************************"
    )

    lm_flag = ["cholesky", "gramschmidt", "correlation"]

    extra_flags = ""
    if args.validate:
        extra_flags += " -DPOLYBENCH_DUMP_ARRAYS"

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

        for dataset in args.input_size:
            content += f"{kernel}_{dataset}: {kernel}.c {kernel}.h\n"
            for interface in args.interfaces:
                if interface == "mpi":
                    content += f"\t${{VERBOSE}} ${{MPI_CC}} -o {kernel}_{dataset}{interfaces[interface]} "
                else:
                    content += f"\t${{VERBOSE}} ${{CC}} -o {kernel}_{dataset}{interfaces[interface]} "
                content += f"{kernel}{interfaces[interface]}.c ${{CFLAGS}} -I. -I{utilities_path} "
                content += f"{pb_source_path} {datasets[dataset]} ${{EXTRA_FLAGS}}"
                if interface == "omp":
                    content += " -fopenmp"
                content += "\n\n"

        content += "clean:\n\t@ rm -f "
        for interface in interfaces:
            for dataset in datasets:
                content += f"{kernel}_{dataset}{interfaces[interface]} "
        content += "\n\n"

        with open(os.path.join(kernels[kernel], "Makefile"), "w") as makefile:
            makefile.write(content)


if not args.no_make:
    print(
        "**************************************************\n"
        "Running make\n"
        "**************************************************"
    )

    for kernel in args.kernels:
        if args.verbose:
            print(kernel)

        make_cmd = ["make"]
        for dataset in args.input_size:
            make_cmd.append(f"{kernel}_{dataset}")
            make_process = subprocess.run(
                make_cmd, cwd=kernels[kernel], capture_output=True, text=True
            )

        if make_process.returncode != 0:
            sys.stderr.write(f"Error running make for kernel {kernel}\n")
            sys.stderr.write(make_process.stderr)
            sys.exit(1)
        if args.verbose:
            sys.stdout.write(make_process.stdout)


print(
    "**************************************************\n"
    "Running Kernels\n"
    "**************************************************"
)


measurements = {}

for kernel in args.kernels:
    measurements[kernel] = {}
    for dataset in args.input_size:
        measurements[kernel][dataset] = {}
        for interface in args.interfaces:
            measurements[kernel][dataset][interface] = []


def run_kernel(kernel, interface, dataset, dump_strs):
    if interface == "mpi":
        cmd = [
            "mpiexec",
            "-np",
            str(args.mpi_processes),
            f"./{kernel}_{dataset}{interfaces[interface]}",
        ]
    else:
        cmd = [f"./{kernel}_{dataset}{interfaces[interface]}"]
    driver_process = subprocess.run(
        cmd, cwd=kernels[kernel], capture_output=True, text=True
    )
    measurements = float(driver_process.stdout)
    if driver_process.returncode != 0:
        sys.stderr.write(
            f"Error running driver for kernel {kernel}_{dataset}{interfaces[interface]}\n"
        )
        sys.stderr.write(driver_process.stderr)
        sys.exit(1)
    if args.verbose:
        sys.stdout.write(driver_process.stdout)
    if args.validate:
        regex_str = r"==BEGIN +DUMP_ARRAYS==\n(?P<dump>(.|\n)*)==END +DUMP_ARRAYS=="
        if interface == "std":
            dump_strs[dataset] = re.search(regex_str, driver_process.stderr).group(
                "dump"
            )
        else:
            dump_str = re.search(regex_str, driver_process.stderr).group("dump")
            if not dump_strs[dataset] == dump_str:
                sys.stderr.write(
                    f"Validation failed for kernel {kernel}_{dataset}{interfaces[interface]}\n"
                )
                sys.exit(1)
    return measurements


for kernel in args.kernels:
    if args.verbose:
        print(kernel)

    dump_strs = {}

    for dataset in args.input_size:
        for interface in args.interfaces:
            for i in range(args.num_runs):
                measurements[kernel][dataset][interface].append(
                    run_kernel(kernel, interface, dataset, dump_strs)
                )

if not os.path.exists("measurements"):
    os.makedirs("measurements")

for kernel in args.kernels:
    measurement_dir = os.path.join("measurements", kernel)
    if not os.path.exists(measurement_dir):
        os.makedirs(measurement_dir)
    with open(
        f"{measurement_dir}/{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}.json", "w+"
    ) as f:
        json.dump(measurements[kernel], f)

sys.exit(0)
