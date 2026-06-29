import os, time
import numpy as np
import pandas as pd
import odatse
from odatse.algorithm import choose_algorithm

from qubo_instances import functions, generate_instances

np.set_printoptions(threshold=np.inf)


def qubo(x: np.ndarray, q_mat: np.ndarray) -> np.ndarray:
    return np.einsum("i...,ij,j...", x, q_mat, x)


class QUBOSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.q_mat = kwargs.get("q_mat", None)

    def evaluate(self, x, args=()):
        return qubo(x, self.q_mat)


n_trials = 10


def run_one_case(func_name, q_mat, dim, seed, output_dir):
    """Run a single ODAT-SE optimization for one QUBO instance/trial.

    Assumes ``odatse.mpi.setup()`` has already been called once. Building the
    ``Info`` with ``Info.from_file`` (rather than ``odatse.initialize()``) avoids
    re-running ``mpi.setup()``, which may only be called once per process.
    """
    toml_content = f"""[base]
dimension = {dim}
output_dir = "{output_dir}"

[solver]
name = "custom"
function_name = "{func_name}"

[algorithm]
name = "ttopt"
seed = {seed}

[algorithm.param]
max_list = {[1] * dim}
min_list = {[0] * dim}

[algorithm.ttopt]
p_points = {[2] * dim}
q_points = {[1] * dim}
r_max = 4
max_f_eval = 100000
"""
    toml_filename = "input_qubo.toml"
    if odatse.mpi.rank() == 0:
        with open(toml_filename, "w") as f:
            f.write(toml_content)

    info = odatse.Info.from_file(toml_filename)
    solver = QUBOSolver(info, q_mat=q_mat)
    runner = odatse.Runner(solver, info)
    alg_module = choose_algorithm(info.algorithm["name"])
    alg = alg_module.Algorithm(info, runner, run_mode="initial")

    time0 = time.time()
    result = alg.main()
    time1 = time.time()

    if odatse.mpi.rank() == 0 and os.path.exists(toml_filename):
        os.remove(toml_filename)
    return result["x"], result["fx"], time1 - time0


def main():
    # Partition the MPI communicator once; the per-case loop below reuses it.
    odatse.mpi.setup()

    records = []
    for func_name, dims in functions.items():
        for dim in dims:
            q_mats = generate_instances(func_name, dim)
            for i in range(len(q_mats)):
                for j in range(n_trials):
                    output_dir = "output/output_%s_d%d_i%d/trial%d" % (func_name, dim, i, j)
                    if odatse.mpi.rank() == 0:
                        os.makedirs(output_dir, exist_ok=True)
                    x, fx, elapsed_time = run_one_case(func_name, q_mats[i], dim, 12345 + j, output_dir)
                    records.append([func_name, dim, i, x, fx, elapsed_time])

    if odatse.mpi.rank() == 0:
        df = pd.DataFrame(records, columns=["f", "dim", "instance", "min_params", "min_f", "time"])
        print("QUBO problems")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df[["f", "dim", "instance", "min_f", "time"]].groupby(["f", "dim", "instance"]).agg(["mean", "std", "min"]))
        df.to_csv("qubo_results.csv", index=False)
        df[["f", "dim", "instance", "min_f", "time"]].groupby(["f", "dim", "instance"]).agg(["mean", "std", "min"]).to_csv("qubo_results_agg.csv")


if __name__ == "__main__":
    main()
