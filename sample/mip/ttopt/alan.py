import sys, os, time
import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

class AlanSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.penalty = kwargs.get("penalty", 100)

        self.opt_x = None
        self.opt_fx = np.inf

    def evaluate(self, x, args=()):
        # Original (un-reduced) 8-variable formulation:
        # objvar = xs[0]*(4*xs[0] + 3*xs[1] - xs[2]) + xs[1]*(3*xs[0] + 6*xs[1] + xs[2]) + xs[2]*(xs[1] - xs[0] + 10*xs[2])
        # cons[0] = (np.sum(xs[:4]) - 1) ** 2
        # cons[1] = (8*xs[0] + 9*xs[1] + 12*xs[2] + 7*xs[3] - 10) ** 2
        # cons[2] = np.clip(np.sum(xs[4:]) - 3, 0, np.inf) ** 2
        # cons[3] = np.clip(xs[0] - xs[4], 0, np.inf) ** 2
        # cons[4] = np.clip(xs[1] - xs[5], 0, np.inf) ** 2
        # cons[5] = np.clip(xs[2] - xs[6], 0, np.inf) ** 2
        # cons[6] = np.clip(xs[3] - xs[7], 0, np.inf) ** 2

        y0 = (3 - x[0] - x[1]) / 5
        y1 = 1 - x[0] - x[1] - y0
        objvar = x[0]*(4*x[0] + 3*x[1] - y0) + x[1]*(3*x[0] + 6*x[1] + y0) + y0*(x[1] - x[0] + 10*y0)
        cons = np.array([
            np.clip(np.sum(x[2:]) - 3, 0, np.inf) ** 2,
            np.clip(x[0] - x[2], 0, np.inf) ** 2,
            np.clip(x[1] - x[3], 0, np.inf) ** 2,
            np.clip(y0 - x[4], 0, np.inf) ** 2,
            np.clip(y1 - x[5], 0, np.inf) ** 2,
        ])

        cost = objvar + np.sum(self.penalty * cons)
        if cost < self.opt_fx:
            self.opt_fx = cost
            self.opt_x = x
            self.opt_objvar = objvar
            self.opt_cons = cons
        return cost

# dim = 8
# min_list = [0, 0, 0, 0, 0, 0, 0, 0]
# max_list = [1, 1, 1, 1, 1, 1, 1, 1]
# p_points = [2, 2, 2, 2, 2, 2, 2, 2]
# q_points = [20, 20, 20, 20, 1, 1, 1, 1]
dim = 6
min_list = [0, 0, 0, 0, 0, 0]
max_list = [1, 1, 1, 1, 1, 1]
p_points = [2, 2, 2, 2, 2, 2]
q_points = [20, 20, 1, 1, 1, 1]
init_points = []

output_dir = f"output/output_alan"
if odatse.mpi.rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

penalty = 10
for n in range(20):
    if odatse.mpi.rank() == 0:
        print(f"\nIteration {n+1} with penalty={penalty}")
    # generate input file
    toml_content = f"""[base]
    dimension = {dim}
    output_dir = "{output_dir}"

    [solver]
    name = "custom"

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
    init_points = {init_points}
    """
    toml_filename = f"input_alan.toml"
    if odatse.mpi.rank() == 0:
        with open(toml_filename, "w") as f:
            f.write(toml_content)
    sys.argv = ["script.py", toml_filename, "--init"]
    info, run_mode = odatse.initialize()
    output_dir = info.base.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    solver = AlanSolver(info, penalty=penalty)
    runner = odatse.Runner(solver, info)
    alg_module = choose_algorithm(info.algorithm["name"])
    alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
    time0 = time.time()
    result = alg.main()
    time1 = time.time()
    elapsed_time = time1 - time0
    if result["x"].tolist() not in init_points:
        init_points.append(result["x"].tolist())
    penalty *= 2

    if odatse.mpi.rank() == 0:
        print(f"\nopt_x={solver.opt_x}")
        print(f"opt_fx={solver.opt_fx} (objective={solver.opt_objvar})")
        print(f"infeasibility={np.sum(solver.opt_cons)}\n")

    # cleanup
    if odatse.mpi.rank() == 0:
        if os.path.exists(toml_filename):
            os.remove(toml_filename)

if odatse.mpi.rank() == 0:
    # true_opt_x = [0.375, 0, 0.525, 0.1, 1, 0, 1, 1]
    true_opt_x = [0.375, 0, 1, 0, 1, 1]
    print(f"global optimum: {solver.evaluate(np.array(true_opt_x))} at {true_opt_x}")