import sys, os, time
import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

class Synthes2Solver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.penalty = kwargs.get("penalty", 100)

        self.opt_x = None
        self.opt_fx = np.inf

    def evaluate(self, x, args=()):
        # Original (un-reduced) 11-variable formulation:
        # objvar = np.exp(xs[0]) - 10*xs[0] + np.exp(xs[1]/1.2) - 15*xs[1] - 60*np.log(1 + xs[3] + xs[4]) + 15*xs[3] + 5*xs[4] - 15*xs[2] - 20*xs[5] + 5*xs[6] + 8*xs[7] + 6*xs[8] + 10*xs[9] + 6*xs[10] + 140
        # cons[0] = np.clip(-np.log(1 + xs[3] + xs[4]), 0, np.inf) ** 2
        # cons[1] = np.clip(np.exp(xs[0]) - 10*xs[6] - 1, 0, np.inf) ** 2
        # cons[2] = np.clip(np.exp(xs[1]/1.2) - 10*xs[7] - 1, 0, np.inf) ** 2
        # cons[3] = np.clip(1.25*xs[2] - 10*xs[8], 0, np.inf) ** 2
        # cons[4] = np.clip(xs[3] + xs[4] - 10*xs[9], 0, np.inf) ** 2
        # cons[5] = np.clip(-2*xs[2] + 2*xs[5] - 10*xs[10], 0, np.inf) ** 2
        # cons[6] = np.clip(-xs[0] - xs[1] - 2*xs[2] + xs[3] + 2*xs[5], 0, np.inf) ** 2
        # cons[7] = np.clip(-xs[0] - xs[1] - 0.75*xs[2] + xs[3] + 2*xs[5], 0, np.inf) ** 2
        # cons[8] = np.clip(xs[2] - xs[5], 0, np.inf) ** 2
        # cons[9] = np.clip(2*xs[2] - xs[3] - 2*xs[5], 0, np.inf) ** 2
        # cons[10] = np.clip(-0.5*xs[3] + xs[4], 0, np.inf) ** 2
        # cons[11] = np.clip(-0.2*xs[3] - xs[4], 0, np.inf) ** 2
        # cons[12] = (xs[6] + xs[7] - 1) ** 2
        # cons[13] = np.clip(xs[9] + xs[10] - 1, 0, np.inf) ** 2

        y0 = 1 - x[6]
        objvar = np.exp(x[0]) - 10*x[0] + np.exp(x[1]/1.2) - 15*x[1] - 60*np.log(1 + x[3] + x[4]) + 15*x[3] + 5*x[4] - 15*x[2] - 20*x[5] + 5*x[6] + 8*y0 + 6*x[7] + 10*x[8] + 6*x[9] + 140
        cons = np.array([
            np.clip(np.exp(x[0]) - 10*x[6] - 1, 0, np.inf) ** 2,
            np.clip(np.exp(x[1]/1.2) - 10*y0 - 1, 0, np.inf) ** 2,
            np.clip(1.25*x[2] - 10*x[7], 0, np.inf) ** 2,
            np.clip(x[3] + x[4] - 10*x[8], 0, np.inf) ** 2,
            np.clip(-2*x[2] + 2*x[5] - 10*x[9], 0, np.inf) ** 2,
            np.clip(-x[0] - x[1] - 0.75*x[2] + x[3] + 2*x[5], 0, np.inf) ** 2,
            np.clip(x[2] - x[5], 0, np.inf) ** 2,
            np.clip(-0.5*x[3] + x[4], 0, np.inf) ** 2,
            np.clip(x[8] + x[9] - 1, 0, np.inf) ** 2,
        ])

        cost = objvar + np.sum(self.penalty * cons)
        if cost < self.opt_fx:
            self.opt_fx = cost
            self.opt_x = x
            self.opt_objvar = objvar
            self.opt_cons = cons
        return cost

# dim = 11
# min_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# max_list = [2, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1]
# p_points = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# q_points = [20, 20, 20, 20, 20, 20, 1, 1, 1, 1, 1]
dim = 10
min_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
max_list = [2, 2, 2, 1, 1, 3, 1, 1, 1, 1]
p_points = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
q_points = [20, 20, 20, 20, 20, 20, 1, 1, 1, 1]
init_points = []

output_dir = f"output/output_synthes2"
if odatse.mpi.rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

penalty = 100
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
    toml_filename = f"input_synthes2.toml"
    if odatse.mpi.rank() == 0:
        with open(toml_filename, "w") as f:
            f.write(toml_content)
    sys.argv = ["script.py", toml_filename, "--init"]
    info, run_mode = odatse.initialize()
    output_dir = info.base.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    solver = Synthes2Solver(info, penalty=penalty)
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
    # true_opt_x = [0, 2, 1.078388278576160, 0.652014651779925, 0.326007325889962, 1.078388278576160, 0, 1, 1, 1, 0]
    true_opt_x = [0, 2, 1.078388278576160, 0.652014651779925, 0.326007325889962, 1.078388278576160, 0, 1, 1, 0]
    solver.penalty = 0  # report the objective only, without the penalty term
    print(f"global optimum: {solver.evaluate(np.array(true_opt_x))} at {true_opt_x}")