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

from sys import exit

import odatse
import odatse.mpi
import odatse.util.toml


def main():
    """
    Main function to run the data-analysis software for quantum beam diffraction experiments
    on 2D material structures. It parses command-line arguments, loads the input file,
    selects the appropriate algorithm and solver, and executes the analysis.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Data-analysis software of quantum beam "
            "diffraction experiments for 2D material structure"
        )
    )
    parser.add_argument("inputfile", help="input file with TOML format")
    parser.add_argument("--version", action="version", version=odatse.__version__)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--init", action="store_true", help="initial start (default)")
    mode_group.add_argument("--resume", action="store_true", help="resume intterupted run")
    mode_group.add_argument("--cont", action="store_true", help="continue from previous run")

    parser.add_argument("--reset_rand", action="store_true", default=False, help="new random number series in resume or continue mode")

    args = parser.parse_args()

    file_name = args.inputfile
    # inp = {}
    # if odatse.mpi.rank() == 0:
    #     inp = odatse.util.toml.load(file_name)
    # if odatse.mpi.size() > 1:
    #     inp = odatse.mpi.comm().bcast(inp, root=0)
    # info = odatse.Info(inp)
    info = odatse.Info.from_file(file_name)

    algname = info.algorithm["name"]
    if algname == "mapper":
        from .algorithm.mapper_mpi import Algorithm
    elif algname == "minsearch":
        from .algorithm.min_search import Algorithm
    elif algname == "exchange":
        from .algorithm.exchange import Algorithm
    elif algname == "pamc":
        from .algorithm.pamc import Algorithm
    elif algname == "bayes":
        from .algorithm.bayes import Algorithm
    else:
        print(f"ERROR: Unknown algorithm ({algname})")
        exit(1)

    solvername = info.solver["name"]
    if solvername == "surface":
        if odatse.mpi.rank() == 0:
            print(
                'WARNING: solver name "surface" is deprecated and will be unavailable in future.'
                ' Use "sim-trhepd-rheed" instead.'
            )
        #from .solver.sim_trhepd_rheed import Solver
        from sim_trhepd_rheed import Solver
    elif solvername == "sim-trhepd-rheed":
        #from .solver.sim_trhepd_rheed import Solver
        from sim_trhepd_rheed import Solver
    elif solvername == "sxrd":
        #from .solver.sxrd import Solver
        from sxrd import Solver
    elif solvername == "leed":
        #from .solver.leed import Solver
        from leed import Solver
    elif solvername == "analytical":
        from .solver.analytical import Solver
    else:
        print(f"ERROR: Unknown solver ({solvername})")
        exit(1)

    if args.init is True:
        run_mode = "initial"
    elif args.resume is True:
        run_mode = "resume"
        if args.reset_rand is True:
            run_mode = "resume-resetrand"
    elif args.cont is True:
        run_mode = "continue"
        if args.reset_rand is True:
            run_mode = "continue-resetrand"
    else:
        run_mode = "initial"  # default

    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    alg = Algorithm(info, runner, run_mode=run_mode)
    result = alg.main()
