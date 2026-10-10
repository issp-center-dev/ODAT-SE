"""A user-written main that follows the tutorial but does *not* use
odatse.mpi.fail_together(): the solver, runner and algorithm are built
directly. run.py injects a failure into Solver(info) or Runner(...) on one
global rank; the job must still end on every process, because the
constructors carry the agreement themselves (odatse.mpi.FailTogetherMeta),
as a host program embedding ODAT-SE relies on.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

import odatse
from odatse.algorithm import choose_algorithm
from odatse.solver import analytical


def main():
    info, run_mode = odatse.initialize()

    solver = analytical.Solver(info)
    runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    alg.main()


if __name__ == "__main__":
    try:
        main()
    except odatse.mpi.OtherAlgorithmProcessError:
        # another process failed and reports its error; leave quietly, as
        # the odatse command does, so that the job ends with its status
        sys.exit(0)
