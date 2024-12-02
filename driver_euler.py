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

inputsizes = {
    "jacobi-2d": [
        {"TSTEPS": 100, "N": 1000},
        {"TSTEPS": 500, "N": 2000},
        {"TSTEPS": 1000, "N": 3000}
    ],
    "gemver": [
        {"N": 1000},
        {"N": 2000},
        {"N": 3000}
    ]
}

datasets = {}

# generate filenames and inputsize flags
for kernel, inputsizes in inputsizes.items():
    datasets[kernel] = {}
    for inputsize in inputsizes:
        filename = f"{kernel}_"
        flags = ""
        for (flag, val) in inputsize.items():
            filename += f"{flag}_{str(val)}_"
            flags += f"-D{flag}={str(val)} "
        filename = filename[:-1]
        datasets[kernel][filename] = flags

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
parser.add_argument("-no-gen-batch", action="store_true", help="Do not generate sbatch bash scripts")
parser.add_argument("-no-batch", action="store_true", help="Do not schedule jobs in batch")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("--num-runs", type=int, help="Number of runs", default=1)
parser.add_argument(
     "--num-p",
     type=int,
     help="Number of processes/threads to use for OpenMP and MPI",
     default=4
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

    lm_flag = ["cholesky", "gramschmidt", "correlation", "jacobi-2d"]

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

        for filename, inputsize_flags in datasets[kernel].items():
            for interface in args.interfaces:
                content += f"{filename}_{interface}: {kernel}{interfaces[interface]}.c {kernel}.h\n"
                content += f"\t@mkdir -p bin\n\t${{VERBOSE}} "
                content += f"${{MPI_CC}}" if interface == "mpi" else f"${{CC}}"
                content += f" -o bin/{filename}{interfaces[interface]} "
                content += f"{kernel}{interfaces[interface]}.c ${{CFLAGS}} -I. -I{utilities_path} "
                content += f"{pb_source_path} {inputsize_flags} ${{EXTRA_FLAGS}}"
                content += " -fopenmp" if interface == "omp" else ""
                content += "\n\n"
        
        content += "clean:\n\t@ rm -f bin/*\n\n"

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
        for filename, _ in datasets[kernel].items():
            for interface in args.interfaces:
                make_cmd.append(f"{filename}_{interface}")
                make_process = subprocess.run(make_cmd, cwd=kernels[kernel], capture_output=True, text=True)

        if make_process.returncode != 0:
            sys.stderr.write(f"Error running make for kernel {kernel}\n")
            sys.stderr.write(make_process.stderr)
            sys.exit(1)
        if args.verbose:
            sys.stdout.write(make_process.stdout)


if not args.no_gen_batch:
    print(
        "**************************************************\n"
        "Generating sbatch bash scripts\n"
        "**************************************************"
    )

    for kernel in args.kernels:
        for filename, inputsize_flags in datasets[kernel].items():
            for interface in args.interfaces:
                sbatch_script = f""


if not args.no_batch:
    print(
        "**************************************************\n"
        "Scheduling jobs in batch\n"
        "**************************************************"
    )

    for kernel in args.kernels:
        for filename, inputsize_flags in datasets[kernel].items():
            for interface in args.interfaces:
                for i in range(args.num_runs):
                    cmd = ["sbatch", f"{filename}_{interface}.sh"]
                    subprocess.run(cmd, cwd=kernels[kernel], capture_output=True, text=True)
                    






sys.exit(0)
