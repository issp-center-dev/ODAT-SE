import sys, os, time
import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

class CvxNonSep_NormCon20Solver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.penalty = kwargs.get("penalty", 100)

        self.opt_x = None
        self.opt_fx = np.inf

    def evaluate(self, x, args=()):
        weights = np.asarray([0.175, 0.39, 0.83, 0.805, 0.06, 0.4, 0.52, 0.415, 0.655, 0.63, 0.29, 0.43, 0.015, 0.985, 0.165, 0.105, 0.37, 0.2, 0.49, 0.34])
        objvar = -np.dot(x, weights)
        cons = np.array([
            np.clip(np.sqrt(0.0001 + np.sum(x**2)) - 10, 0, np.inf) ** 2,
        ])

        cost = objvar + np.sum(self.penalty * cons)
        if cost < self.opt_fx:
            self.opt_fx = cost
            self.opt_x = x
            self.opt_objvar = objvar
            self.opt_cons = cons
        return cost

dim = 20
min_list = [0]*20
max_list = [5]*20
p_points = [6]*10 + [2]*10
q_points = [1]*10 + [20]*10
init_points = []
# init_points = [[1, 2, 4, 4, 0, 2, 2, 2, 3, 3] + [0]*10]

output_dir = "output/output_cvxnonsep_normcon20"
if odatse.mpi.rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

penalty = 1
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
    toml_filename = f"input_cvxnonsep_normcon20.toml"
    if odatse.mpi.rank() == 0:
        with open(toml_filename, "w") as f:
            f.write(toml_content)
    sys.argv = ["script.py", toml_filename, "--init"]
    info, run_mode = odatse.initialize()
    output_dir = info.base.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    solver = CvxNonSep_NormCon20Solver(info, penalty=penalty)
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
    true_opt_x = [1, 2, 4, 4, 0, 2, 2, 2, 3, 3, 1.238166456215540, 1.835901986779380, 0.064043093177772, 4.205496411514280, 0.704474018226710, 0.448301648012646, 1.579729616536840, 0.853907900860535, 2.092074357017180, 1.451643429643470]
    print(f"global optimum: {solver.evaluate(np.array(true_opt_x))} at {true_opt_x}")