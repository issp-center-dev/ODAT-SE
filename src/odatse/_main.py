# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import sys
import odatse

def main():
    info, run_mode = odatse.initialize()
    alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])

    solvername = info.solver["name"]
    if solvername == "analytical":
        from .solver.analytical import Solver
    else:
        if odatse.mpi.rank() == 0:
            print(f"ERROR: Unknown solver ({solvername})")
        sys.exit(1)

    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    alg = alg_module.Algorithm(info, runner, run_mode=run_mode)

    result = alg.main()
