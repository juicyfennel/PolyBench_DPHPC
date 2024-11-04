import argparse
import subprocess
import sys
import os
import json
from datetime import datetime

categories = ["linear-algebra/kernels",
              "linear-algebra/blas",
              "linear-algebra/solvers",
              "datamining",
              "medley",
              "stencils"]

parser = argparse.ArgumentParser(description="Python script that wraps PolyBench")
parser.add_argument("--kernels", type=str, nargs="+", help="Kernels to run (default = all)", default=[])
parser.add_argument("--no-gen", action="store_true", help="Do not regenerate makefiles")
parser.add_argument("--no-make", action="store_true", help="Do not run make")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("--num-runs", type=int, help="Number of runs", default=1)

args = parser.parse_args()

if not args.no_gen:
    print("**************************************************\n"
          "Generating makefiles\n"
          "**************************************************")
    lm_flag = ["cholesky",
               "gramschmidt",
               "correlation"]
    extra_flags = ""
    for category in categories:
        for root, dirs, files in os.walk(f"./{category}"):
            for kernel in dirs:
                if not args.kernels == [] and kernel not in args.kernels:
                    continue
                if args.verbose:
                    print(kernel)
                make_cmd = ["make", "clean"]
                subprocess.run(make_cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)

                kernel_path = os.path.join(root, kernel)
                rel_root = os.path.relpath(".", kernel_path)

                content = f"include {rel_root}/config.mk\n\n"
                content += f"EXTRA_FLAGS={extra_flags}"

                if kernel in lm_flag:
                    content += " -lm"

                content += "\n\n"
                content += f"{kernel}: {kernel}.c {kernel}.h\n"
                content += f"\t${{VERBOSE}} ${{CC}} -o {kernel} {kernel}.c ${{CFLAGS}} -I. -I{rel_root}/utilities {rel_root}/utilities/polybench.c ${{EXTRA_FLAGS}}\n\n"

                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_omp.c")):
                    content += f"\t${{VERBOSE}} ${{CC}} -o {kernel}_omp {kernel}_omp.c ${{CFLAGS}} -I. -I{rel_root}/utilities {rel_root}/utilities/polybench.c -fopenmp ${{EXTRA_FLAGS}}\n\n"

                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_mpi.c")):
                    content += f"\t${{VERBOSE}} ${{CC}} -o {kernel}_mpi {kernel}_mpi.c ${{CFLAGS}} -I. -I{rel_root}/utilities {rel_root}/utilities/polybench.c ${{EXTRA_FLAGS}}\n\n"

                content += f"clean:\n\t@ rm -f {kernel} {kernel}_omp {kernel}_mpi\n"

                with open(os.path.join(root, kernel, "Makefile"), 'w') as makefile:
                    makefile.write(content)

if not args.no_make:
    print("**************************************************\n"
          "Running make\n"
          "**************************************************")
    for category in categories:
        for root, dirs, files in os.walk(f"./{category}"):
            for kernel in dirs:
                if not args.kernels == [] and kernel not in args.kernels:
                    continue
                if args.verbose:
                    print(kernel)

                make_cmd = ["make"]
                make_process = subprocess.run(make_cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)

                if make_process.returncode != 0:
                    sys.stderr.write(f"Error running make for kernel {kernel}\n")
                    sys.stderr.write(make_process.stderr)
                    sys.exit(1)
                if args.verbose:
                    sys.stdout.write(make_process.stdout)

measurements = {}

# Special dataset sizes for gemver
dataset_sizes = {
    "MINI_DATASET": 40,
    "SMALL_DATASET": 120,
    "MEDIUM_DATASET": 400,
    "LARGE_DATASET": 2000,
    "EXTRALARGE_DATASET": 4000
}

print("**************************************************\n"
      "Running Kernels\n"
      "**************************************************")
for category in categories:
    for root, dirs, files in os.walk(f"./{category}"):
        for kernel in dirs:
            if not args.kernels == [] and kernel not in args.kernels:
                continue
            if args.verbose:
                print(kernel)

            # Initialize measurements for the kernel
            measurements[kernel] = {}
            for dataset, size in dataset_sizes.items():
                measurements[kernel][dataset] = {"default": []}

                # Check for parallel versions
                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_omp.c")):
                    measurements[kernel][dataset]["omp"] = []
                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_mpi.c")):
                    measurements[kernel][dataset]["mpi"] = []

                # Run the kernel for the specified number of times
                for _ in range(args.num_runs):
                    # Set the dataset size definition in CFLAGS
                    env = os.environ.copy()
                    env["CFLAGS"] = f"-D{dataset}"

                    cmd = [f"./{kernel}"]
                    process = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                    try:
                        measurements[kernel][dataset]["default"].append(float(process.stdout.strip()))
                    except ValueError:
                        sys.stderr.write(f"Error parsing output for {kernel} with dataset {dataset}\n")

                    # Run the omp version if it exists
                    if "omp" in measurements[kernel][dataset]:
                        cmd = [f"./{kernel}_omp"]
                        process = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                        try:
                            measurements[kernel][dataset]["omp"].append(float(process.stdout.strip()))
                        except ValueError:
                            sys.stderr.write(f"Error parsing output for {kernel}_omp with dataset {dataset}\n")

                    # Run the mpi version if it exists
                    if "mpi" in measurements[kernel][dataset]:
                        cmd = [f"./{kernel}_mpi"]
                        process = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                        try:
                            measurements[kernel][dataset]["mpi"].append(float(process.stdout.strip()))
                        except ValueError:
                            sys.stderr.write(f"Error parsing output for {kernel}_mpi with dataset {dataset}\n")

            # Save the measurements for the current kernel
            kernel_dir = os.path.join("measurements", kernel)
            if not os.path.exists(kernel_dir):
                os.makedirs(kernel_dir)
            with open(f"{kernel_dir}/{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}.json", "w+") as f:
                json.dump(measurements[kernel], f, indent=4)


sys.exit(0)
