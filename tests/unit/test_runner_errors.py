# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Unit tests for the error handling of Runner.submit() / Runner.serve().

Two layers are covered:

* a single-rank solver group (``nsolve = 1``, which is also what the no-MPI
  stub reports): ``ignore_error`` turns a ``RuntimeError`` from
  ``solver.evaluate()`` into ``NaN`` and nothing else, with no communication;
* the solver-group status exchange in ``Runner._evaluate_group()``, driven
  with a fake ``solcomm`` so that the controller / worker decision logic can
  be exercised on a single process (the real multi-rank protocol is covered
  by tests/parallel_solver/worker_error).
"""
import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse
import odatse.mpi as mpi
from odatse.exception import SolverError


def _info(ignore_error=False):
    return odatse.Info({
        "base": {"dimension": 1},
        "solver": {"name": "custom"},
        "runner": {"ignore_error": ignore_error},
        "algorithm": {"name": "mapper"},
    })


class _Solver(odatse.solver.SolverBase):
    """evaluate() raises ``exc`` if given, else returns 1.0."""

    def __init__(self, info, exc=None):
        super().__init__(info)
        self.exc = exc

    def evaluate(self, x, args=()):
        if self.exc is not None:
            raise self.exc
        return 1.0


def _runner(exc=None, ignore_error=False):
    info = _info(ignore_error)
    return odatse.Runner(_Solver(info, exc), info)


X = np.zeros(1)


# --------------------------------------------------------------------------- #
#  Single-rank solver group (nsolve = 1, and the no-MPI stub)
# --------------------------------------------------------------------------- #

def test_single_rank_precondition():
    # conftest calls setup() without nsolve, so every process is a solver
    # group of its own; the no-MPI stub reports the same.
    assert mpi.solsize() == 1


def test_submit_success_returns_value():
    assert _runner().submit(X) == 1.0


def test_submit_runtime_error_propagates_without_ignore_error():
    with pytest.raises(RuntimeError, match="boom"):
        _runner(RuntimeError("boom")).submit(X)


def test_submit_runtime_error_becomes_nan_with_ignore_error():
    assert np.isnan(_runner(RuntimeError("boom"), ignore_error=True).submit(X))


@pytest.mark.parametrize("ignore_error", [False, True])
def test_submit_other_exception_always_propagates(ignore_error):
    with pytest.raises(ValueError, match="boom"):
        _runner(ValueError("boom"), ignore_error=ignore_error).submit(X)


def test_single_rank_uses_no_collective(monkeypatch):
    def forbidden():
        raise AssertionError("solcomm() must not be used when solsize() == 1")
    monkeypatch.setattr(mpi, "solcomm", forbidden)
    assert np.isnan(_runner(RuntimeError("boom"), ignore_error=True).submit(X))


# --------------------------------------------------------------------------- #
#  Solver-group status exchange with a fake solcomm
# --------------------------------------------------------------------------- #

class _FakeSolcomm:
    """Stand-in for solcomm: allgather() returns the prepared statuses of the
    other ranks around this rank's own entry, and counts its calls so that the
    test can check that exactly one collective is issued per evaluation."""

    def __init__(self, others, me):
        self.others = list(others)   # entries of the other ranks, in rank order
        self.me = me                 # this rank's position in the group
        self.calls = 0               # allgather() calls (the status exchange)
        self.broadcasts = 0          # Bcast()/bcast() calls (the control protocol)

    def allgather(self, own):
        self.calls += 1
        return self.others[:self.me] + [own] + self.others[self.me:]

    # submit() broadcasts the control message, x and args before evaluating
    def Bcast(self, buf, root=0):
        self.broadcasts += 1

    def bcast(self, obj, root=0):
        self.broadcasts += 1
        return obj


def _fake_group(monkeypatch, solrank, others, nsolve=2, global_rank=None):
    """Make odatse.mpi report a solver group of ``nsolve`` ranks in which this
    process is ``solrank`` and the other ranks' statuses are ``others``."""
    comm = _FakeSolcomm(others, solrank)
    monkeypatch.setattr(mpi, "solsize", lambda: nsolve)
    monkeypatch.setattr(mpi, "solrank", lambda: solrank)
    monkeypatch.setattr(mpi, "solcomm", lambda: comm)
    monkeypatch.setattr(mpi, "rank", lambda: solrank if global_rank is None else global_rank)
    return comm


OK = None
WORKER_RTE = (True, "[rank 3] RuntimeError: worker boom")
WORKER_VAL = (False, "[rank 3] ValueError: worker boom")


def test_group_all_succeed(monkeypatch):
    comm = _fake_group(monkeypatch, solrank=0, others=[OK])
    result, error = _runner()._evaluate_group(X, ())
    assert result == 1.0 and error is None
    assert comm.calls == 1


def test_group_worker_runtime_error_is_runtime_error_on_controller(monkeypatch):
    comm = _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE])
    result, error = _runner()._evaluate_group(X, ())
    assert type(error) is RuntimeError
    assert "failed on 1 rank(s)" in str(error)
    assert "[rank 3] RuntimeError: worker boom" in str(error)
    assert error.__cause__ is None          # the controller itself succeeded
    assert comm.calls == 1


def test_group_worker_runtime_error_honours_ignore_error(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE])
    assert np.isnan(_runner(ignore_error=True).submit(X))


def test_group_worker_runtime_error_propagates_without_ignore_error(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE])
    with pytest.raises(RuntimeError, match=r"\[rank 3\] RuntimeError: worker boom"):
        _runner().submit(X)


@pytest.mark.parametrize("ignore_error", [False, True])
def test_group_worker_other_exception_is_solver_error(monkeypatch, ignore_error):
    _fake_group(monkeypatch, solrank=0, others=[WORKER_VAL])
    with pytest.raises(SolverError, match=r"\[rank 3\] ValueError: worker boom") as excinfo:
        _runner(ignore_error=ignore_error).submit(X)
    assert excinfo.value.rank_local
    assert not isinstance(excinfo.value, RuntimeError)   # never NaN-able


def test_group_only_controller_failed_returns_own_exception(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[OK])
    own = RuntimeError("controller boom")
    result, error = _runner(own)._evaluate_group(X, ())
    assert error is own
    assert np.isnan(result)


def test_group_controller_and_worker_failed_lists_both(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE], global_rank=2)
    own = RuntimeError("controller boom")
    _, error = _runner(own)._evaluate_group(X, ())
    assert type(error) is RuntimeError
    assert "failed on 2 rank(s)" in str(error)
    assert "[rank 2] RuntimeError: controller boom" in str(error)
    assert "[rank 3] RuntimeError: worker boom" in str(error)
    assert error.__cause__ is own


def test_group_mixed_types_is_solver_error_with_cause(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[WORKER_VAL], global_rank=2)
    own = RuntimeError("controller boom")
    _, error = _runner(own)._evaluate_group(X, ())
    assert isinstance(error, SolverError)
    assert error.__cause__ is own
    with pytest.raises(SolverError):
        _runner(own, ignore_error=True).submit(X)


def test_group_controller_other_exception_with_worker_failure_is_solver_error(monkeypatch):
    # SolverError is not only "a worker raised a non-RuntimeError": a
    # non-RuntimeError on the controller combined with any worker failure
    # also yields it ...
    _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE], global_rank=2)
    own = ValueError("controller boom")
    _, error = _runner(own)._evaluate_group(X, ())
    assert isinstance(error, SolverError)
    assert error.__cause__ is own
    with pytest.raises(SolverError):
        _runner(own, ignore_error=True).submit(X)


def test_group_only_controller_failed_other_exception_is_unchanged(monkeypatch):
    # ... whereas the controller failing alone re-raises its own exception.
    _fake_group(monkeypatch, solrank=0, others=[OK])
    own = ValueError("controller boom")
    _, error = _runner(own)._evaluate_group(X, ())
    assert error is own


def test_group_worker_serve_never_raises(monkeypatch, capsys):
    comm = _fake_group(monkeypatch, solrank=1, others=[OK], global_rank=3)
    _runner(RuntimeError("worker boom")).serve(X, ())   # must not raise
    assert comm.calls == 1
    err = capsys.readouterr().err
    assert "[rank 3] ERROR: solver worker failed: worker boom" in err
    assert "Traceback" in err


def test_group_worker_is_silent_when_error_will_be_ignored(monkeypatch, capsys):
    _fake_group(monkeypatch, solrank=1, others=[OK], global_rank=3)
    _runner(RuntimeError("worker boom"), ignore_error=True).serve(X, ())
    assert capsys.readouterr().err == ""


def test_group_worker_reports_non_runtime_error_despite_ignore_error(monkeypatch, capsys):
    _fake_group(monkeypatch, solrank=1, others=[OK], global_rank=3)
    _runner(ValueError("worker boom"), ignore_error=True).serve(X, ())
    assert "ValueError: worker boom" in capsys.readouterr().err


def test_group_worker_is_silent_when_only_controller_failed(monkeypatch, capsys):
    comm = _fake_group(monkeypatch, solrank=1, others=[(True, "[rank 2] RuntimeError: controller boom")])
    _runner().serve(X, ())
    assert comm.calls == 1
    assert capsys.readouterr().err == ""
