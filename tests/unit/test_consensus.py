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


# --- rank-local marking for the CLI error boundary (issue #60) ---

def test_reach_consensus_marks_own_odatse_error_rank_local():
    """When a rank re-raises its own odatse error through the consensus
    protocol, the exception must be marked rank-local so the CLI boundary
    (odatse._main.main) reports it from the owning rank, not only rank 0."""
    import odatse.exception

    alg = _bare()
    err = odatse.exception.CheckpointError("boom")
    assert err.rank_local is False
    with pytest.raises(odatse.exception.CheckpointError):
        alg._reach_consensus(err, np.array([0]))
    assert err.rank_local is True


def test_reach_consensus_leaves_foreign_exceptions_unmarked():
    """Non-odatse exceptions propagate as-is (they surface as tracebacks on
    the failing rank anyway)."""
    alg = _bare()
    err = ValueError("boom")
    with pytest.raises(ValueError):
        alg._reach_consensus(err, np.array([0]))
    assert not hasattr(err, "rank_local")


# --- SystemExit / KeyboardInterrupt go through the consensus too ---

def test_reach_consensus_reraises_base_exception():
    alg = _bare()
    with pytest.raises(SystemExit):
        alg._reach_consensus(SystemExit(3), np.array([0]))


@pytest.mark.parametrize("phase", ["prepare", "run", "post"])
@pytest.mark.parametrize("exc", [SystemExit(2), KeyboardInterrupt()])
def test_phase_wrapper_passes_base_exception_through_consensus(phase, exc, monkeypatch):
    """sys.exit() or Ctrl-C inside a phase hook is handed to _reach_consensus
    like any other failure (ok = 0), so that the other algorithm ranks are
    released instead of waiting in the Allreduce, and is then re-raised as
    it is. Before, the wrappers caught Exception only, so these bypassed the
    consensus and hung the other ranks."""
    alg = _bare()
    alg.runner = _StubRunner()
    alg.mode = "init"
    alg.proc_dir = "."
    alg.output_dir = "."

    def failing():
        raise exc
    if phase == "prepare":
        alg.status = AlgorithmStatus.INIT
        alg._initialize = failing
    elif phase == "run":
        alg.status = AlgorithmStatus.PREPARE
        alg._run = failing
    else:
        alg.status = AlgorithmStatus.RUN
        alg._post = failing

    seen = []
    def spy(error, ok):
        seen.append((error, int(ok[0])))
        raise error
    monkeypatch.setattr(alg, "_reach_consensus", spy)

    with pytest.raises(type(exc)):
        getattr(alg, phase)()
    assert seen == [(exc, 0)]


@pytest.mark.parametrize("exc_type", [SystemExit, KeyboardInterrupt])
def test_base_exception_on_one_rank_releases_the_others(exc_type):
    """The motivating case: sys.exit() / Ctrl-C in a phase hook on one
    algorithm rank only. That rank re-raises it after the consensus, every
    other rank raises OtherAlgorithmProcessError, and nobody is left in the
    Allreduce (this test would time out under mpirun before the fix)."""
    if not (mpi.enabled() and mpi.algsize() > 1):
        pytest.skip("needs more than one algorithm rank (run under mpirun)")

    alg = _bare()
    alg.runner = _StubRunner()
    alg.proc_dir = "."
    alg.status = AlgorithmStatus.PREPARE

    def failing_run():
        if mpi.algrank() == 0:
            raise exc_type(3)
    alg._run = failing_run

    try:
        alg.run()
        outcome = "returned"
    except exc_type:
        outcome = "own"
    except mpi.OtherAlgorithmProcessError:
        outcome = "other"

    expected = "own" if mpi.algrank() == 0 else "other"
    assert outcome == expected
    outcomes = mpi.algcomm().allgather(outcome)
    assert len(outcomes) == mpi.algsize() and outcomes.count("own") == 1
