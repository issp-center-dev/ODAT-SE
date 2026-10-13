# Regression test for an exception raised inside solver.evaluate() when the
# solver group has more than one rank (--nsolve > 1).
#
# A worker (solrank > 0) is not a member of the algorithm communicator, so it
# cannot take part in the per-phase error consensus. Originally an exception on
# a worker killed only that process and left its controller blocked forever;
# then it aborted the whole job with MPI_Abort, which also killed jobs whose
# failure would have been ignored on the controller (ignore_error = true).
#
# Now every rank of the solver group exchanges its evaluate() status after the
# call (Runner._evaluate_group), so a failure on any rank becomes an ordinary
# evaluate failure on the controller: ignored (NaN) when ignore_error is set
# and every failing rank raised a RuntimeError, propagated otherwise.
#
# FAILMODE selects which rank(s) raise at evaluation FAIL_AT, always *after*
# the collective inside evaluate() (raising before it on a subset of ranks is
# a mismatch of collectives and is the solver's responsibility):
#   worker      solrank == 1 only
#   controller  solrank == 0 only
#   all         every rank of the group
#   rank1       global rank 1 only, i.e. the worker of the first solver group:
#               the other group stays healthy and must still terminate
#   before      solrank == 1 raises *before* the collective inside evaluate():
#               a mismatch of collectives, which the framework can only
#               detect (and abort) because the solver's collective is a
#               pickle-based allgather
# FAILTYPE selects the exception class: runtime (RuntimeError), value
# (ValueError, which ignore_error must not swallow) or exit (SystemExit from
# sys.exit(), which must not hang the group and is never ignored).

# Prefer the source tree over any installed odatse package, so that the tests
# always exercise the working copy. The path must be absolute because odatse
# changes the working directory during a run.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import numpy as np
import odatse
from odatse.algorithm import choose_algorithm

FAILMODE = os.environ.get("FAILMODE", "worker")
FAILTYPE = os.environ.get("FAILTYPE", "runtime")
FAIL_AT = 3  # evaluation count at which the selected rank(s) raise

_EXC = {"runtime": RuntimeError, "value": ValueError, "exit": SystemExit}[FAILTYPE]


class ParallelSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)
        self.count = 0

    def _func(self, xs):
        x, y = xs
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def evaluate(self, xs, args):
        self.count += 1

        if FAILMODE == "before" and self.count == FAIL_AT and odatse.mpi.solrank() == 1:
            raise _EXC(f"before failed at evaluation {FAIL_AT}")

        fs = odatse.mpi.solcomm().allgather(self._func(xs))

        if self.count == FAIL_AT:
            solrank = odatse.mpi.solrank()
            fail = (
                (FAILMODE == "worker" and solrank == 1)
                or (FAILMODE == "controller" and solrank == 0)
                or FAILMODE == "all"
                or (FAILMODE == "rank1" and odatse.mpi.rank() == 1)
            )
            if fail:
                raise _EXC(f"{FAILMODE} failed at evaluation {FAIL_AT}")

        return float(np.average(fs))


def main():
    info, run_mode = odatse.initialize(sys.argv[1:])
    os.makedirs(info.base.get("output_dir", "./output"), exist_ok=True)

    solver = ParallelSolver(info)
    runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    try:
        alg.main()
    except BaseException as e:
        # Report the exception in a single write for do.sh to grep. The
        # interpreter prints the last line of an uncaught traceback in several
        # writes ("SolverError", ": ", message), and both controllers fail at
        # the same evaluation, so their stderr can interleave inside that line.
        print(f"[rank {odatse.mpi.rank()}] main() raised {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
