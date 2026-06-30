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
