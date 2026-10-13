"""odatse.mpi.fail_together(): a block that raises on some processes only
must make every process raise when it leaves the block, instead of letting
the others go on into the next collective and wait for the failed one.
odatse.main() runs the set-up (solver and runner) inside such a block."""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse
import odatse.mpi as mpi
from odatse import exception
from odatse._main import main


def _last_rank() -> bool:
    return mpi.algrank() == mpi.algsize() - 1


def test_block_that_completes_everywhere_is_transparent():
    with mpi.fail_together():
        value = 42
    assert value == 42


def test_failure_on_every_rank_propagates_the_own_exception():
    with pytest.raises(RuntimeError, match=f"on rank {mpi.algrank()}"):
        with mpi.fail_together():
            raise RuntimeError(f"failure on rank {mpi.algrank()}")


def test_failure_marks_a_framework_error_rank_local():
    with pytest.raises(exception.InputError) as excinfo:
        with mpi.fail_together():
            raise exception.InputError("failure")
    assert excinfo.value.rank_local is True


@pytest.mark.parametrize("exc", [
    RuntimeError("failure"), SystemExit(3), KeyboardInterrupt(), StopIteration(),
], ids=lambda e: type(e).__name__)
def test_the_own_exception_object_propagates_unchanged(exc):
    """The block's exception itself comes out (not a wrapper, nor the
    RuntimeError a generator turns a StopIteration into)."""
    with pytest.raises(type(exc)) as excinfo:
        with mpi.fail_together():
            raise exc
    assert excinfo.value is exc


def test_before_setup_the_exception_only_propagates(monkeypatch):
    """Without setup() there is no layout to agree on: no collective is
    issued, and an odatse error is not marked rank-local."""
    monkeypatch.setattr(mpi._ctx, "_ready", False, raising=False)
    err = exception.InputError("failure")
    with pytest.raises(exception.InputError) as excinfo:
        with mpi.fail_together():
            raise err
    assert excinfo.value is err
    assert err.rank_local is False
    # on success nothing is raised, whatever the other processes do
    with mpi.fail_together():
        pass


def test_failure_on_one_rank_releases_the_others():
    """Serially this is the symmetric case (rank 0 fails)."""
    if _last_rank():
        with pytest.raises(RuntimeError, match="injected"):
            with mpi.fail_together():
                raise RuntimeError("injected failure")
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            with mpi.fail_together():
                pass


def test_system_exit_in_the_block_releases_the_others():
    if _last_rank():
        with pytest.raises(SystemExit):
            with mpi.fail_together():
                raise SystemExit(3)
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            with mpi.fail_together():
                pass


def test_nested_blocks_stay_balanced():
    """The inner block raises on every process (its own error on the failing
    one, OtherAlgorithmProcessError on the others); the outer block then
    sees a failure everywhere and lets each exception through unchanged."""
    if _last_rank():
        with pytest.raises(RuntimeError, match="injected"):
            with mpi.fail_together():
                with mpi.fail_together():
                    raise RuntimeError("injected failure")
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            with mpi.fail_together():
                with mpi.fail_together():
                    pass


def _stub_initialize(monkeypatch):
    # initialize() would call mpi.setup() a second time in the test session
    info = odatse.Info({
        "base": {"dimension": 2, "output_dir": "output"},
        "algorithm": {"name": "mapper"},
        "solver": {"name": "analytical", "function_name": "himmelblau"},
    })
    monkeypatch.setattr(odatse, "initialize", lambda argv: (info, "initial"))


def _fail_on_last_rank(monkeypatch, cls):
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        if _last_rank():
            raise exception.InputError("injected set-up failure")
        original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", __init__)


@pytest.mark.parametrize("where", ["solver", "runner"])
def test_main_releases_every_rank_when_the_set_up_fails_on_one(monkeypatch, capsys, where):
    """The failing rank reports its error and exits with status 1; the
    others leave quietly with status 0 instead of waiting in the
    construction of the algorithm (issue #112)."""
    from odatse.solver.analytical import Solver
    _stub_initialize(monkeypatch)
    _fail_on_last_rank(monkeypatch, Solver if where == "solver" else odatse.Runner)

    with pytest.raises(SystemExit) as excinfo:
        main([])

    err = capsys.readouterr().err
    if _last_rank():
        assert excinfo.value.code == 1
        # reported by the rank that failed, which need not be rank 0
        assert "ERROR: injected set-up failure" in err
    else:
        assert excinfo.value.code == 0
        assert "ERROR" not in err
