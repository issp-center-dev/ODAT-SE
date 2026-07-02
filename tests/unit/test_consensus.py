import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse.mpi as mpi
from odatse.algorithm._algorithm import AlgorithmBase, AlgorithmStatus


class _StubRunner:
    def prepare(self, proc_dir):
        pass

    def post(self):
        pass


class _Alg(AlgorithmBase):
    """Minimal concrete algorithm whose _initialize fails on a chosen rank, to
    exercise prepare()'s dispatch-failure consensus."""
    fail_rank = None  # algrank whose _initialize raises

    def __init__(self):
        pass

    def _initialize(self):
        if self.fail_rank is not None and mpi.algrank() == self.fail_rank:
            raise RuntimeError(f"injected init failure on rank {self.fail_rank}")

    def _prepare(self):
        pass

    def _run(self):
        pass

    def _post(self):
        return {}


def _bare():
    return _Alg.__new__(_Alg)


# --- _reach_consensus in isolation (works serially) ---

def test_reach_consensus_no_error_does_not_raise():
    alg = _bare()
    alg._reach_consensus(None, np.array([1]))  # must not raise


def test_reach_consensus_reraises_own_error():
    alg = _bare()
    err = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        alg._reach_consensus(err, np.array([0]))


# --- the actual no-deadlock guarantee (needs >1 algorithm rank) ---

def test_prepare_dispatch_failure_does_not_deadlock():
    """If the checkpoint dispatch / _initialize fails on one rank only, every
    rank must raise and return -- not block at the consensus collective.
    Before the fix the dispatch ran outside the try/Allreduce, so the other
    ranks hung here (this test would time out under mpirun)."""
    if not (mpi.enabled() and mpi.algsize() > 1):
        pytest.skip("needs more than one algorithm rank (run under mpirun)")

    alg = _bare()
    alg.runner = _StubRunner()
    alg.mode = "init"
    alg.proc_dir = "."
    alg.status = AlgorithmStatus.INIT
    alg.fail_rank = 0  # only rank 0's _initialize raises

    raised = False
    try:
        alg.prepare()
    except Exception:
        raised = True

    # rank 0 raises its own error; the others raise OtherAlgorithmProcessError;
    # crucially every rank gets here (no deadlock).
    assert raised
    survived = mpi.algcomm().allgather(True)
    assert len(survived) == mpi.algsize() and all(survived)
