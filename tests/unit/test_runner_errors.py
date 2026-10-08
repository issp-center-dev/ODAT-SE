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
        self.reductions = 0          # Allreduce() calls (the failure flag)
        self.calls = 0               # allgather() calls (the failure details)
        self.broadcasts = 0          # Bcast()/bcast() calls (the control protocol)

    def Allreduce(self, sendbuf, recvbuf, op=None):
        """Sum of the 0/1 flags: an other rank counts as failed unless its
        prepared entry is the plain success entry."""
        from odatse._runner import _status_entry
        self.reductions += 1
        # what the real buffer collective requires of its arguments
        for buf in (sendbuf, recvbuf):
            assert isinstance(buf, np.ndarray) and buf.shape == (1,) and buf.dtype == np.int32
        assert sendbuf[0] in (0, 1)
        others_failed = sum(0 if o == _status_entry() else 1 for o in self.others)
        recvbuf[0] = sendbuf[0] + others_failed

    def allgather(self, own):
        self.calls += 1
        self.sent = own              # this rank's entry, as handed to MPI
        return self.others[:self.me] + [own] + self.others[self.me:]

    # submit() broadcasts the control message, x and args before evaluating
    def Bcast(self, buf, root=0):
        self.broadcasts += 1

    def bcast(self, obj, root=0):
        self.broadcasts += 1
        return obj


def _tag(entry):
    """Wrap a test's plain status (None or (bool, str)) the way the exchange
    does; anything else is passed through as foreign data."""
    from odatse._runner import _status_entry
    if entry is None:
        return _status_entry()
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], bool):
        return _status_entry(*entry)
    return entry


def _fake_group(monkeypatch, solrank, others, nsolve=2, global_rank=None):
    """Make odatse.mpi report a solver group of ``nsolve`` ranks in which this
    process is ``solrank`` and the other ranks' statuses are ``others``
    (plain None / (bool, str), tagged here as the exchange does)."""
    comm = _FakeSolcomm([_tag(o) for o in others], solrank)
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
    # the all-succeeded path costs one flag reduction and no object exchange
    assert comm.reductions == 1 and comm.calls == 0


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
    # wording differs from the MPI_Abort path ("solver worker failed") so a
    # log tells the two apart
    assert ("[rank 3] ERROR: solver worker raised in evaluate(), reported to the controller: "
            "RuntimeError: worker boom") in err
    assert "solver worker failed" not in err
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


# --------------------------------------------------------------------------- #
#  Robustness of the status exchange itself
# --------------------------------------------------------------------------- #

class _Unprintable(RuntimeError):
    """An exception whose message cannot be rendered."""
    def __str__(self):
        raise ValueError("no message for you")


def test_group_unprintable_exception_still_joins_the_exchange(monkeypatch):
    """Building the status entry must not raise: a rank that failed here
    would skip the allgather the other ranks are entering and hang them. The
    failure is still reported, with a placeholder message, and is ignorable
    because it is a RuntimeError."""
    comm = _fake_group(monkeypatch, solrank=0, others=[OK])
    result, error = _runner(_Unprintable())._evaluate_group(X, ())
    assert comm.calls == 1
    assert isinstance(error, _Unprintable)
    assert np.isnan(result)
    assert np.isnan(_runner(_Unprintable(), ignore_error=True).submit(X))


def test_group_worker_unprintable_exception_is_described(monkeypatch):
    """The entry exchanged for a failing rank is a plain (bool, str) pair even
    when the exception cannot be rendered."""
    comm = _fake_group(monkeypatch, solrank=1, others=[OK], global_rank=3)
    _runner(_Unprintable())._evaluate_group(X, ())
    assert comm.calls == 1
    assert comm.sent == _tag((True, "[rank 3] _Unprintable: <unprintable exception>"))


def test_is_ignorable_is_the_single_policy():
    from odatse.exception import is_ignorable
    assert is_ignorable(RuntimeError("x"))
    assert is_ignorable(_Unprintable())
    assert not is_ignorable(ValueError("x"))
    assert not is_ignorable(SolverError("x"))


def test_describe_error_never_raises():
    from odatse.exception import describe_error
    assert describe_error(ValueError("boom")) == "ValueError: boom"
    assert describe_error(_Unprintable()) == "_Unprintable: <unprintable exception>"


def test_error_base_class_constructor_compatibility():
    import pickle
    from odatse.exception import Error, SolverError
    e = Error("msg", 42)
    assert e.message == "msg" and e.args == ("msg", 42)
    e0 = Error()                        # argument-less construction still works ...
    assert e0.message == "" and e0.args == () and repr(e0) == "Error()"
    assert Error("").args == ("",)      # ... and an explicit empty message is kept
    for exc in (Error(), Error(""), Error("m"), SolverError("s"), Error("m", 1)):
        back = pickle.loads(pickle.dumps(exc))
        assert type(back) is type(exc) and back.args == exc.args and back.message == exc.message


class _ExitingStr(RuntimeError):
    """__str__ raises something that is not an Exception."""
    def __str__(self):
        raise SystemExit(5)


def test_describe_error_survives_base_exception_in_str():
    from odatse.exception import describe_error
    assert describe_error(_ExitingStr()) == "_ExitingStr: <unprintable exception>"


def test_group_controller_joins_exchange_even_if_str_exits(monkeypatch):
    comm = _fake_group(monkeypatch, solrank=0, others=[OK])
    assert np.isnan(_runner(_ExitingStr(), ignore_error=True).submit(X))
    assert comm.calls == 1


@pytest.mark.parametrize("foreign", [
    (np.array(["a", "b"]), None),          # == on the tag would be elementwise
    (np.array([1.0, 2.0]), np.zeros(2)),
    ("odatse.runner.status", np.zeros(2)),
    object(),
])
def test_is_status_entry_never_raises_on_foreign_data(foreign):
    from odatse._runner import _is_status_entry
    assert _is_status_entry(foreign) is False


# --------------------------------------------------------------------------- #
#  SystemExit / KeyboardInterrupt inside evaluate()
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("exc", [SystemExit(3), KeyboardInterrupt()])
def test_group_controller_base_exception_joins_exchange_then_is_solver_error(monkeypatch, exc):
    """Not an Exception, but it must still enter the status exchange (the
    workers are waiting in it). It is then handed over as a SolverError,
    which the phase wrappers and the algorithm-layer consensus (Exception
    only) can terminate the job with; ignore_error never applies."""
    comm = _fake_group(monkeypatch, solrank=0, others=[OK])
    with pytest.raises(SolverError, match=type(exc).__name__ + ".*controller") as excinfo:
        _runner(exc, ignore_error=True).submit(X)
    assert comm.calls == 1
    assert type(excinfo.value.__cause__) is type(exc)


@pytest.mark.parametrize("exc", [SystemExit(3), KeyboardInterrupt()])
def test_single_rank_base_exception_keeps_its_meaning(exc):
    """With no solver group there is nobody to agree with: sys.exit() and
    Ctrl-C inside evaluate() behave as they always did."""
    with pytest.raises(type(exc)):
        _runner(exc, ignore_error=True).submit(X)


@pytest.mark.parametrize("foreign", [1.5, None, (True, "looks like a status")])
def test_group_garbage_in_exchange_aborts_instead_of_hanging(monkeypatch, capsys, foreign):
    """If evaluate() raised on some ranks before a solcomm collective the
    others entered, this rank's status allgather pairs with the solver's own
    collective and receives its data. Entries are tagged, so even a solver's
    None or a status-shaped tuple is recognised as foreign; the job is
    aborted instead of hanging."""
    from odatse._runner import _is_status_entry
    assert not _is_status_entry(foreign)
    _fake_group(monkeypatch, solrank=0, others=[])
    comm = mpi.solcomm()
    comm.others = [foreign]            # bypass _tag: raw data from the solver
    aborted = []
    monkeypatch.setattr(mpi, "comm", lambda: type("C", (), {"Abort": lambda self, code: aborted.append(code)})())
    with pytest.raises(RuntimeError, match="mismatched collectives"):
        _runner()._evaluate_group(X, ())
    assert aborted == [1]
    assert "mismatched collectives inside solver.evaluate()" in capsys.readouterr().err


def test_group_worker_system_exit_is_reported_not_raised(monkeypatch, capsys):
    comm = _fake_group(monkeypatch, solrank=1, others=[OK], global_rank=3)
    _runner(SystemExit(3)).serve(X, ())   # must not raise, must not exit
    assert comm.calls == 1
    assert comm.sent == _tag((False, "[rank 3] SystemExit: 3"))


def test_group_worker_system_exit_becomes_solver_error_on_controller(monkeypatch):
    _fake_group(monkeypatch, solrank=0, others=[(False, "[rank 3] SystemExit: 3")])
    with pytest.raises(SolverError, match=r"\[rank 3\] SystemExit: 3"):
        _runner(ignore_error=True).submit(X)


def test_group_failure_path_costs_one_reduction_and_one_exchange(monkeypatch):
    comm = _fake_group(monkeypatch, solrank=0, others=[WORKER_RTE])
    _runner()._evaluate_group(X, ())
    assert comm.reductions == 1 and comm.calls == 1


def test_group_worker_success_entry_is_sent_only_on_failure(monkeypatch):
    """A succeeding worker takes part in the details exchange only when some
    other rank failed (the flag told it so)."""
    comm = _fake_group(monkeypatch, solrank=1, others=[OK])
    _runner().serve(X, ())
    assert comm.reductions == 1 and comm.calls == 0
    comm = _fake_group(monkeypatch, solrank=1, others=[(True, "[rank 2] RuntimeError: controller boom")])
    _runner().serve(X, ())
    assert comm.reductions == 1 and comm.calls == 1
