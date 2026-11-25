"""Construction consensus under solver parallelism (run on 4 processes).

With ``--nalg 2 --nsolve 2`` the processes are laid out as

    global rank   0           1        2           3
    group         0           0        1           1
    role          controller  worker   controller  worker

A failure is injected into a constructor (algorithm, solver, runner) on one
process at a time, selected by *global* rank (selecting by algrank would
fail a whole group). The process that failed must re-raise its own
exception, every other process, in both groups, must raise
OtherAlgorithmProcessError, and no process may be left waiting.
tests/unit/test_construction_consensus.py covers the algorithm layer only
(nsolve = 1); this exercises the solver layer (the solcomm reduction and the
verdict sent to the workers). The solver and runner are constructed without
a fail_together() block, as a user script or a host program would do.
"""

import os
import pathlib
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../../src")
sys.path.insert(0, SOURCE_PATH)

import odatse
import odatse.mpi as mpi
import odatse.solver
from odatse import exception
from odatse.algorithm._algorithm import AlgorithmBase

mpi.setup(nalg=2, nsolve=2)
rank = mpi.rank()
assert mpi.size() == 4, "run on 4 processes"

info = odatse.Info({
    "base": {"dimension": 2, "output_dir": "output"},
    "algorithm": {"name": "test"},
    "solver": {},
    "runner": {},
})


class _Alg(AlgorithmBase):
    fail_rank = None   # global rank whose constructor raises
    exc = RuntimeError

    def __init__(self):
        if rank == self.fail_rank:
            raise self.exc(f"injected construction failure on rank {rank}")
        self.constructed = True

    def _initialize(self):
        pass

    def _prepare(self):
        pass

    def _run(self):
        pass

    def _post(self):
        pass


class _BaseAlg(_Alg):
    """Runs the real AlgorithmBase.__init__ (holds an algcomm collective)."""

    def __init__(self, info):
        AlgorithmBase.__init__(self, info)
        self.constructed = True


class _Solver(odatse.solver.SolverBase):
    """Fails in its constructor on ``fail_rank``; its ``name`` property, the
    first thing Runner.__init__ touches, fails on ``name_fails_on``."""
    fail_rank = None
    name_fails_on = None

    def __init__(self, info):
        if rank == self.fail_rank:
            raise exception.InputError(f"injected solver failure on rank {rank}")
        super().__init__(info)
        self.constructed = True

    @property
    def name(self):
        if rank == self.name_fails_on:
            raise exception.InputError(f"injected runner failure on rank {rank}")
        return "test"

    def evaluate(self, x, args=()):
        return 0.0


failures = []


def expect(name, fail_rank, construct, own_exc, rank_local=None):
    try:
        obj = construct()
    except mpi.OtherAlgorithmProcessError:
        if fail_rank is None or rank == fail_rank:
            failures.append(f"{name}: unexpected OtherAlgorithmProcessError")
    except own_exc as e:
        if rank != fail_rank:
            failures.append(f"{name}: unexpected {type(e).__name__}: {e}")
        elif rank_local is not None and getattr(e, "rank_local", None) != rank_local:
            failures.append(f"{name}: rank_local is {e.rank_local!r}")
    else:
        if fail_rank is not None:
            failures.append(f"{name}: constructor returned")
        elif not getattr(obj, "constructed", obj is not None):
            # test classes set `constructed` at the end of their __init__;
            # for framework classes (Runner) a returned instance is enough
            failures.append(f"{name}: no constructed instance returned")


def case(name, fail_rank, exc=RuntimeError, rank_local=None):
    _Alg.fail_rank = fail_rank
    _Alg.exc = exc
    expect(name, fail_rank, _Alg, exc, rank_local)


# --- Algorithm(...) ---------------------------------------------------------
case("success", None)
case("one worker of group 0 fails", 1)
case("one worker of group 1 fails", 3)
case("controller of group 1 fails", 2, exception.InputError, rank_local=True)
case("SystemExit on a worker", 3, SystemExit)


# a controller fails to create its output directory, before the
# synchronisation of the algorithm ranks in AlgorithmBase.__init__
_orig_mkdir = pathlib.Path.mkdir


def _mkdir(self, *args, **kwargs):
    raise PermissionError(f"injected mkdir failure: {self}")


for name, fail_rank in [("mkdir fails on the controller of group 1", 2),
                        ("mkdir fails on a worker of group 0", 1)]:
    if rank == fail_rank:
        pathlib.Path.mkdir = _mkdir
    expect(name, fail_rank, lambda: _BaseAlg(info), PermissionError)
    pathlib.Path.mkdir = _orig_mkdir


# --- Solver(info) and Runner(solver, info), no fail_together() block --------
def solver_case(name, fail_rank):
    _Solver.fail_rank = fail_rank
    _Solver.name_fails_on = None
    expect(name, fail_rank, lambda: _Solver(info), exception.InputError, rank_local=True)


solver_case("solver constructed everywhere", None)
solver_case("solver fails on a worker of group 0", 1)
solver_case("solver fails on the controller of group 1", 2)

_Solver.fail_rank = None
solver = _Solver(info)


def runner_case(name, fail_rank):
    _Solver.name_fails_on = fail_rank
    expect(name, fail_rank, lambda: odatse.Runner(solver, info), exception.InputError, rank_local=True)
    _Solver.name_fails_on = None


runner_case("runner constructed everywhere", None)
runner_case("runner fails on a worker of group 1", 3)
runner_case("runner fails on the controller of group 0", 0)


# report through rank 0 only: lines printed concurrently by several ranks
# may interleave in the launcher's output
from mpi4py import MPI
all_failures = MPI.COMM_WORLD.gather(failures, root=0)
if rank == 0:
    for r, fs in enumerate(all_failures):
        for f in fs:
            print(f"[rank {r}] FAILED {f}")
    print("ALL RANKS OK" if not any(all_failures) else "SOME RANKS FAILED", flush=True)
sys.exit(1 if failures else 0)
