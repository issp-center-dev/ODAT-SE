import sys, os, subprocess
import math
import argparse

import odatse

functions = {
    "ackley": [(-32.768, 32.768), 10],
    "alpine": [(-10.0, 10.0), 10],
    "exponential": [(-1.0, 1.0), 10],
    "griewank": [(-600.0, 600.0), 10],
    "himmelblau": [(-6.0, 6.0), 2],
    "michalewicz": [(0.0, math.pi), 10],
    "qing": [(-500.0, 500.0), 10],
    "rastrigin": [(-5.12, 5.12), 10],
    "rosenbrock": [(-5.0, 5.0), 2],
    "schaffer": [(-100.0, 100.0), 10],
    "schwefel": [(-500.0, 500.0), 10],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run TTOpt on a suite of analytical benchmark functions."
    )
    parser.add_argument(
        "nprocs",
        nargs="?",
        type=int,
        default=1,
        help="number of MPI processes to launch (default: 1)",
    )
    args = parser.parse_args()
    if args.nprocs < 1:
        parser.error("nprocs must be a positive integer")
    nprocs = str(args.nprocs)

    for func_name, (bounds, dim) in functions.items():
        if odatse.mpi.rank() == 0:
            print(f"Optimizing the {dim}-dimensional {func_name} function...", flush=True)
        output_dir = f"output/output_{func_name}"
        os.makedirs(output_dir, exist_ok=True)
        min_list = [bounds[0]] * dim
        max_list = [bounds[1]] * dim
        p_points = [2] * dim
        q_points = [25] * dim

        # generate input file
        toml_content = f"""[base]
dimension = {dim}
output_dir = "{output_dir}"

[solver]
name = "analytical"
function_name = "{func_name}"

[algorithm]
name = "ttopt"
seed = 12345

[algorithm.param]
max_list = {max_list}
min_list = {min_list}

[algorithm.ttopt]
p_points = {p_points}
q_points = {q_points}
r_max = 4
max_f_eval = 100000
"""
        input_filename = f"input_{func_name}.toml"
        with open(input_filename, "w") as f:
            f.write(toml_content)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        cmd = ["mpiexec", "-np", nprocs, sys.executable, "-m", "odatse", input_filename]
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {func_name}: {e}")

        # cleanup
        if os.path.exists(input_filename):
            os.remove(input_filename)


if __name__ == "__main__":
    main()
