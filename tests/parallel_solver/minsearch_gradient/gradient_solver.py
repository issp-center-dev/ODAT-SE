# Integration test of gradients with a parallel solver (--nsolve > 1).
#
# Both evaluate() and gradient() of this solver use a collective on solcomm:
# every rank of the solver group computes the terms of its own components and
# the parts are summed with allgather. The workers (solrank > 0) therefore
# have to take part in every gradient() of their controller: the controller
# sends MSG_GRADIENT, xp and args, and the worker loop serves the request in
# Runner.serve_gradient(). If a worker did not compute its share the
# controller would wait in the allgather forever (do.sh kills the job after a
# time limit), and if it computed it at another x the gradient would be wrong.
#
# Objective: f(x) = sum_i W_i (x_i - C_i)^2 + (x_i - C_i)^4, minimum f = 0 at
# x = C. Component i is computed by the rank with solrank == i % solsize.
#
# FAILMODE selects a failure inside gradient(), always *after* the collective
# (raising before it on a subset of ranks is a mismatch of collectives and
# the solver's responsibility):
#   none    no failure
#   worker  solrank == 1 raises at its FAIL_AT-th gradient() call
# FAILTYPE selects the exception class: runtime (RuntimeError, which
# ignore_error turns into NaN) or value (ValueError, which it must not).

# Prefer the source tree over any installed odatse package, so that the tests
# always exercise the working copy. The path must be absolute because odatse
# changes the working directory during a run.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

FAILMODE = os.environ.get("FAILMODE", "none")
FAILTYPE = os.environ.get("FAILTYPE", "runtime")
FAIL_AT = 2  # gradient() call count at which the selected rank raises

_EXC = {"runtime": RuntimeError, "value": ValueError}[FAILTYPE]

C = np.array([1.5, -0.5])
W = np.array([4.0, 1.0])


def objective(x):
    d = np.asarray(x, dtype=float) - C
    return float(np.sum(W * d**2 + d**4))


def objective_gradient(x):
    d = np.asarray(x, dtype=float) - C
    return 2.0 * W * d + 4.0 * d**3


class GradientSolver(odatse.solver.SolverBase):
    supports_gradient = True

    def __init__(self, info, **kwargs):
        super().__init__(info)
        self.ngrad = 0

    def _mine(self):
        """Components computed on this rank of the solver group."""
        return np.arange(self.dimension) % odatse.mpi.solsize() == odatse.mpi.solrank()

    def evaluate(self, xs, args=()):
        d = (np.asarray(xs, dtype=float) - C)[self._mine()]
        part = float(np.sum(W[self._mine()] * d**2 + d**4))
        return float(sum(odatse.mpi.solcomm().allgather(part)))

    def gradient(self, xs, args=()):
        self.ngrad += 1
        part = np.where(self._mine(), objective_gradient(xs), 0.0)
        g = np.sum(odatse.mpi.solcomm().allgather(part), axis=0)
        if FAILMODE == "worker" and odatse.mpi.solrank() == 1 and self.ngrad == FAIL_AT:
            # a marker for do.sh: the failure path was really exercised
            print(f"[rank {odatse.mpi.rank()}] gradient_solver: raising {_EXC.__name__} "
                  f"at gradient {FAIL_AT}", file=sys.stderr, flush=True)
            raise _EXC(f"worker failed at gradient {FAIL_AT}")
        return g


def main():
    info, run_mode = odatse.initialize(sys.argv[1:])
    with odatse.mpi.fail_together():
        os.makedirs(info.base.get("output_dir", "./output"), exist_ok=True)
        solver = GradientSolver(info)
        runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    try:
        alg.main()
    except BaseException as e:
        # Report the exception in a single write for do.sh to grep (the last
        # line of an uncaught traceback is written piecewise and the two
        # controllers' stderr may interleave inside it).
        print(f"[rank {odatse.mpi.rank()}] main() raised {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except odatse.mpi.OtherAlgorithmProcessError:
        # another process failed and reports its error; leave quietly
        sys.exit(0)
