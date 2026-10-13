"""Construction consensus under solver parallelism (run on 4 processes).

With ``--nalg 2 --nsolve 2`` the processes are laid out as

    global rank   0           1        2           3
    group         0           0        1           1
    role          controller  worker   controller  worker

A failure is injected into the constructor on one process at a time,
selected by *global* rank (selecting by algrank would fail a whole group).
The process that failed must re-raise its own exception, every other
process, in both groups, must raise OtherAlgorithmProcessError, and no
process may be left waiting. tests/unit/test_construction_consensus.py
covers the algorithm layer only (nsolve = 1); this exercises the solver
layer (the solcomm reduction and the verdict sent to the workers).
"""

import os
import pathlib
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../../src")
sys.path.insert(0, SOURCE_PATH)

import odatse
import odatse.mpi as mpi
from odatse import exception
from odatse.algorithm._algorithm import AlgorithmBase

mpi.setup(nalg=2, nsolve=2)
rank = mpi.rank()
assert mpi.size() == 4, "run on 4 processes"


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
        elif not getattr(obj, "constructed", False):
            failures.append(f"{name}: no constructed instance returned")


def case(name, fail_rank, exc=RuntimeError, rank_local=None):
    _Alg.fail_rank = fail_rank
    _Alg.exc = exc
    expect(name, fail_rank, _Alg, exc, rank_local)


case("success", None)
case("one worker of group 0 fails", 1)
case("one worker of group 1 fails", 3)
case("controller of group 1 fails", 2, exception.InputError, rank_local=True)
case("SystemExit on a worker", 3, SystemExit)


# a controller fails to create its output directory, before the
# synchronisation of the algorithm ranks in AlgorithmBase.__init__
info = odatse.Info({
    "base": {"dimension": 2, "output_dir": "output"},
    "algorithm": {"name": "test"},
    "solver": {},
})
_orig_mkdir = pathlib.Path.mkdir


def _mkdir(self, *args, **kwargs):
    raise PermissionError(f"injected mkdir failure: {self}")


for name, fail_rank in [("mkdir fails on the controller of group 1", 2),
                        ("mkdir fails on a worker of group 0", 1)]:
    if rank == fail_rank:
        pathlib.Path.mkdir = _mkdir
    expect(name, fail_rank, lambda: _BaseAlg(info), PermissionError)
    pathlib.Path.mkdir = _orig_mkdir

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
