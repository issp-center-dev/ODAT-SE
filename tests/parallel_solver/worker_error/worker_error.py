# Regression test for an exception raised on a solver worker (solrank > 0).
#
# A worker is not a member of the algorithm communicator, so it cannot take
# part in the per-phase error consensus. Before the fix, an exception in
# solver.evaluate() on a worker killed only that process and left its
# controller (solrank == 0) blocked forever in a solcomm collective or in the
# next control-signal Bcast. Now the worker reports the error and aborts the
# whole job, so mpirun exits with a non-zero status instead of hanging.
#
# FAILMODE selects where the worker raises relative to the collective inside
# evaluate(): "before" leaves the controller stuck in allgather(), "after"
# leaves it stuck in the next Bcast of the control signal.

# Prefer the source tree over any installed odatse package, so that the tests
# always exercise the working copy. The path must be absolute because odatse
# changes the working directory during a run.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

FAILMODE = os.environ.get("FAILMODE", "after")
FAIL_AT = 3  # evaluation count at which the worker raises


class ParallelSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)
        self.count = 0

    def _func(self, xs):
        x, y = xs
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def evaluate(self, xs, args):
        self.count += 1
        fail = odatse.mpi.solrank() == 1 and self.count == FAIL_AT

        if fail and FAILMODE == "before":
            raise RuntimeError("worker failed before collective")

        fs = odatse.mpi.solcomm().allgather(self._func(xs))

        if fail and FAILMODE == "after":
            raise RuntimeError("worker failed after collective")

        return float(np.average(fs))


def main():
    info, run_mode = odatse.initialize(sys.argv[1:])
    os.makedirs(info.base.get("output_dir", "./output"), exist_ok=True)

    solver = ParallelSolver(info)
    runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    alg.main()


if __name__ == "__main__":
    main()
