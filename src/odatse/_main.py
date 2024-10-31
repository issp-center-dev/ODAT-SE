# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
import odatse

def main():
    """
    Main function to run the data-analysis software for quantum beam diffraction experiments
    on 2D material structures. It parses command-line arguments, loads the input file,
    selects the appropriate algorithm and solver, and executes the analysis.
    """

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
