import argparse
import subprocess
import sys
import os
import json
import re
from datetime import datetime

categories = ["linear-algebra/kernels",
              "linear-algebra/blas",
              "linear-algebra/solvers",
              "datamining",
              "medley",
              "stencils"]

datasets = {"mini": "-DMINI_DATASET",
            "small": "-DSMALL_DATASET",
            "medium": "-DMEDIUM_DATASET",
            "large": "-DLARGE_DATASET",
            "extralarge": "-DEXTRALARGE_DATASET"}

parser = argparse.ArgumentParser(description="Python script that wraps PolyBench")
parser.add_argument("--kernels", type=str, nargs="+", help="Kernels to run (default = all)", default=[])
parser.add_argument("--no-gen", action="store_true", help="Do not regenerate makefiles (works only with a single dataset)")
parser.add_argument("--no-make", action="store_true", help="Do not run make (works only with a single dataset)")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("--num-runs", type=int, help="Number of runs", default=1)
parser.add_argument("--validate", action="store_true", help="Validate results against reference")
parser.add_argument("--input-size", type=str, nargs="+", help="Input size for kernels (select from 'mini', 'small', 'medium', 'large', 'extralarge')", default=["medium"])


args = parser.parse_args()


measurements = {}

for dataset in args.input_size:

    if not args.no_gen:
        print("**************************************************\n"
            "Generating makefiles\n"
            "**************************************************")
        
        lm_flag = ["cholesky",
                "gramschmidt",
                "correlation"]
        extra_flags = datasets[dataset]
        if args.validate:
            extra_flags += " -DPOLYBENCH_DUMP_ARRAYS"
        for category in categories:
            for root, dirs, files in os.walk(f"./{category}"):
                for kernel in dirs:
                    if not args.kernels == [] and kernel not in args.kernels:
                        continue
                    if args.verbose:
                        print(kernel)
                    make_cmd = ["make", "clean"]
                    make_process = subprocess.run(make_cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)

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


    print("**************************************************\n"
        "Running Kernels\n"
        "**************************************************")
    
    measurements[dataset] = {}

    for category in categories:
        for root, dirs, files in os.walk(f"./{category}"):
            for kernel in dirs:
                if not args.kernels == [] and kernel not in args.kernels:
                    continue

                if args.verbose:
                    print(kernel)

                measurements[dataset][kernel] = []
                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_omp.c")):
                    measurements[dataset][f"{kernel}_omp"] = []
                if os.path.isfile(os.path.join(root, kernel, f"{kernel}_mpi.c")):
                    measurements[dataset][f"{kernel}_mpi"] = []

                for i in range(args.num_runs):
                    cmd = [f"./{kernel}"]
                    driver_process = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                    measurements[dataset][kernel].append(float(driver_process.stdout))
                    if driver_process.returncode != 0:
                        sys.stderr.write(f"Error running driver for kernel {kernel}\n")
                        sys.stderr.write(driver_process.stderr)
                        sys.exit(1)
                    if args.verbose:
                        sys.stdout.write(driver_process.stdout)
                    if args.validate:
                        regex_str = r"==BEGIN +DUMP_ARRAYS==\n(?P<dump>(.|\n)*)==END +DUMP_ARRAYS=="
                        dump_str = re.search(regex_str, driver_process.stderr).group("dump")

                    if os.path.isfile(os.path.join(root, kernel, f"{kernel}_omp.c")):
                        cmd = [f"./{kernel}_omp"]
                        driver_process_omp = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                        measurements[dataset][f"{kernel}_omp"].append(float(driver_process_omp.stdout))
                        if driver_process_omp.returncode != 0:
                            sys.stderr.write(f"Error running driver for kernel {kernel}_omp\n")
                            sys.stderr.write(driver_process_omp.stderr)
                            sys.exit(1)
                        if args.verbose:
                            sys.stdout.write(driver_process_omp.stdout)
                        if args.validate:
                            dump_str_omp = re.search(regex_str, driver_process_omp.stderr).group("dump")
                            if not dump_str == dump_str_omp:
                                sys.stderr.write(f"Validation failed for kernel {kernel}_omp\n")
                                sys.exit(1)

                    if os.path.isfile(os.path.join(root, kernel, f"{kernel}_mpi.c")):
                        cmd = [f"./{kernel}_mpi"]
                        driver_process_mpi = subprocess.run(cmd, cwd=os.path.join(root, kernel), capture_output=True, text=True)
                        measurements[dataset][f"{kernel}_mpi"].append(float(driver_process_mpi.stdout))
                        if driver_process_mpi.returncode != 0:
                            sys.stderr.write(f"Error running driver for kernel {kernel}_mpi\n")
                            sys.stderr.write(driver_process_mpi.stderr)
                            sys.exit(1)
                        if args.verbose:
                            sys.stdout.write(driver_process_mpi.stdout)
                        if args.validate:
                            dump_str_mpi = re.search(regex_str, driver_process_mpi.stderr).group("dump")
                            if not dump_str == dump_str_mpi:
                                sys.stderr.write(f"Validation failed for kernel {kernel}_mpi\n")
                                sys.exit(1)

if not os.path.exists("measurements"):
    os.makedirs("measurements")
    
with open(f"measurements/{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}.json", "w+") as f:
    json.dump(measurements, f)

sys.exit(0)
