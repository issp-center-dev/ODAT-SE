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


def test_missing_section_message_has_no_error_prefix():
    """The exception text must not carry its own ``ERROR: `` prefix: the CLI
    boundary adds one, so a prefixed message printed ``ERROR: ERROR: ...``."""
    with pytest.raises(InputError) as excinfo:
        odatse.Info({"base": {"dimension": 2}})
    msg = str(excinfo.value)
    assert not msg.startswith("ERROR:")
    assert msg == "section algorithm does not appear in input"


def test_cli_prints_exactly_one_error_prefix(monkeypatch, capsys):
    """Regression for the duplicated prefix: the CLI must emit ``ERROR: ``
    exactly once for a domain error."""
    def boom(argv):
        odatse.Info({"base": {"dimension": 2}})
    monkeypatch.setattr(odatse, "initialize", boom)

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    if odatse.mpi.rank() != 0:
        # a config error is raised identically on every rank, so only rank 0
        # reports it (see test_main_global_error_prints_only_on_rank0)
        assert err == ""
        return
    assert err.count("ERROR:") == 1
    assert err.strip() == "ERROR: section algorithm does not appear in input"


def test_pamc_step_config_error_has_no_error_prefix():
    """Same contract for the other message that used to carry the prefix."""
    from odatse.algorithm.pamc import Algorithm as PAMCAlgorithm

    # bare instance: _find_scheduling is exercised in isolation, as in
    # tests/unit/test_pamc_scheduling.py
    alg = PAMCAlgorithm.__new__(PAMCAlgorithm)
    with pytest.raises(InputError) as excinfo:
        alg._find_scheduling({"numsteps": 0, "numsteps_annealing": 0, "Tnum": 0})
    assert not str(excinfo.value).startswith("ERROR:")
    assert str(excinfo.value).startswith("Two of 'numsteps'")
