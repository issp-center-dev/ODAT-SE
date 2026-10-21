# Regression test for an exception raised by the solver on one algorithm rank
# inside the main loop of a lockstep algorithm (exchange, pamc, ttopt).
#
# The phase wrappers agree on failures only after the phase. A rank that
# raised inside the loop used to leave for that consensus while the other
# ranks waited for it in their next collective (the replica exchange, the
# resampling, the allgather of a sweep): the job neither reported the error
# nor exited. Now a failure is kept until the next synchronisation point and
# shared there (AlgorithmBase._agree_or_fail), so every rank leaves the loop
# and the error is reported once.
#
# The solver raises ValueError (or calls sys.exit with FAILTYPE=exit) on
# algorithm rank FAIL_RANK at its FAIL_AT-th evaluation (environment variables); with one walker per rank (exchange),
# evaluation 1 is the initial configuration and evaluation k + 1 is step k.

# Prefer the source tree over any installed odatse package, so that the tests
# always exercise the working copy. The path must be absolute because odatse
# changes the working directory during a run.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import odatse
from odatse.algorithm import choose_algorithm

FAIL_RANK = int(os.environ.get("FAIL_RANK", "1"))
FAIL_AT = int(os.environ.get("FAIL_AT", "5"))
# FAILTYPE: "value" raises ValueError; "exit" calls sys.exit(1), which must be
# shared like any other failure (the phase wrappers and the in-loop agreement
# handle BaseException)
FAILTYPE = os.environ.get("FAILTYPE", "value")


class FailingSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)
        self.count = 0

    def evaluate(self, xs, args):
        self.count += 1
        if odatse.mpi.algrank() == FAIL_RANK and self.count == FAIL_AT:
            if FAILTYPE == "exit":
                sys.exit(f"injected exit on rank {FAIL_RANK} at evaluation {FAIL_AT}")
            raise ValueError(f"injected failure on rank {FAIL_RANK} at evaluation {FAIL_AT}")
        x, y = xs
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def main():
    info, run_mode = odatse.initialize(sys.argv[1:])
    os.makedirs(info.base.get("output_dir", "./output"), exist_ok=True)

    solver = FailingSolver(info)
    runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    try:
        alg.main()
    except BaseException as e:
        # one write per rank, for do.sh to grep
        print(f"[rank {odatse.mpi.rank()}] main() raised {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
