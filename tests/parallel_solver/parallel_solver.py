# Parallel solver sample demonstrating two-level MPI parallelism:
#   nalg  algorithm processes each evaluate a disjoint subset of grid points (algcomm)
#   nsolve solver processes per group share the computation within one evaluation (solcomm)
#
# Problem: evaluate Himmelblau function in all solver processes and take the average
#

from typing import Optional, Sequence

import os, time, argparse
import numpy as np
from mpi4py import MPI
import odatse
from odatse.algorithm import choose_algorithm

class ParallelSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        # for debug purpose
        self.delay = info.solver.get("delay", 0.0)

        if odatse.mpi.rank()==0:
            print(f"nalg: {odatse.mpi.algsize()}")
            print(f"nsolve: {odatse.mpi.solsize()}")
        odatse.mpi.comm().barrier()

    def _func(self, xs):
        x, y = xs
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def _compute(self, xs): # called by all solcomm ranks
        f = self._func(xs)

        # gather and take average
        fs = odatse.mpi.solcomm().allgather(f)
        return np.average(fs)

    def evaluate(self, xs, args):
        if odatse.mpi.solrank() == 0:
            # Per-evaluation logging is verbose; comment out to suppress.
            # print(f"algrank: {odatse.mpi.algrank()}, x: {list(xs)}")
            pass

        result = self._compute(xs)

        if odatse.mpi.solrank() == 0:
            # Per-evaluation logging is verbose; comment out to suppress.
            # print(f"algrank: {odatse.mpi.algrank()}, result: {result}")

            # for debug purose
            if self.delay > 0.0:
                time.sleep(self.delay)  # delay controller process

        return result

def main(argv: Optional[Sequence[str]] = None):
    info, run_mode = odatse.initialize(argv)

    if odatse.mpi.rank() == 0:
        print(f"total mpi: {odatse.mpi.size()}")

    output_dir = info.base.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    solver = ParallelSolver(info)
    runner = odatse.Runner(solver, info)
    alg_module = choose_algorithm(info.algorithm["name"])
    alg = alg_module.Algorithm(info, runner, run_mode=run_mode)

    time0 = time.time()
    result = alg.main()
    odatse.mpi.comm().barrier()
    time1 = time.time()
    elapsed_time = time1 - time0

    odatse.mpi.comm().barrier()

    if odatse.mpi.rank() == 0:
        print(f"time: {elapsed_time:.6f}s")

if __name__ == "__main__":
    main()
