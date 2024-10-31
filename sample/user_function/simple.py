# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
