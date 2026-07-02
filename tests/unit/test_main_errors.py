import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse
from odatse._main import choose_solver, main
from odatse.algorithm import choose_algorithm
from odatse.exception import InputError


def test_choose_algorithm_unknown_raises():
    """Unknown algorithm name raises InputError instead of calling sys.exit."""
    with pytest.raises(InputError):
        choose_algorithm("no_such_algorithm_xyz")


def test_choose_solver_unknown_raises():
    info = odatse.Info({
        "base": {"dimension": 2},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "no_such_solver"},
    })
    with pytest.raises(InputError, match="Unknown solver"):
        choose_solver(info)


def test_choose_solver_analytical_ok():
    info = odatse.Info({
        "base": {"dimension": 2},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical"},
    })
    Solver = choose_solver(info)
    assert Solver.__name__ == "Solver"


def test_main_converts_input_error_to_exit(monkeypatch):
    """At the CLI boundary, a domain error (odatse.exception.Error) is reported
    and turned into a non-zero exit status rather than propagating a raw
    exception. (initialize() is stubbed because it would otherwise call
    mpi.setup() a second time within the test session.)"""
    def boom(argv):
        raise InputError("simulated input error")
    monkeypatch.setattr(odatse, "initialize", boom)

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1


def test_main_reports_rank_local_error_from_owning_rank(monkeypatch, capsys):
    """A rank-local error (e.g. a CheckpointError re-raised through the
    consensus protocol on the failing rank) must be reported by the rank that
    owns it — previously only rank 0 printed, so a failure on any other rank
    killed the job with no diagnostic text anywhere (issue #60)."""
    from odatse.exception import CheckpointError

    err = CheckpointError("simulated per-rank failure")
    err.rank_local = True

    def boom(argv):
        raise err
    monkeypatch.setattr(odatse, "initialize", boom)
    # pretend to be a non-zero rank of a 4-process run
    monkeypatch.setattr(odatse.mpi, "rank", lambda: 2)
    monkeypatch.setattr(odatse.mpi, "size", lambda: 4)

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "[rank 2]" in captured.err
    assert "simulated per-rank failure" in captured.err


def test_main_global_error_prints_only_on_rank0(monkeypatch, capsys):
    """Errors raised identically on all ranks (e.g. config errors) keep the
    rank-0 gate so a bad input file is not reported once per process."""
    def boom(argv):
        raise InputError("global config error")
    monkeypatch.setattr(odatse, "initialize", boom)
    monkeypatch.setattr(odatse.mpi, "rank", lambda: 2)
    monkeypatch.setattr(odatse.mpi, "size", lambda: 4)

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1
    assert capsys.readouterr().err == ""
