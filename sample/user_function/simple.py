import numpy as np

import odatse
import odatse.util.toml
import odatse.algorithm.mapper_mpi as pm_alg
#import odatse.algorithm.min_search as pm_alg
import odatse.solver.function


def my_objective_fn(x: np.ndarray) -> float:
    return np.mean(x * x)


file_name = "input.toml"
inp = odatse.util.toml.load(file_name)
info = odatse.Info(inp)

solver = odatse.solver.function.Solver(info)
solver.set_function(my_objective_fn)

runner = odatse.Runner(solver, info)

alg = pm_alg.Algorithm(info, runner)
retv = alg.main()

print(retv)
