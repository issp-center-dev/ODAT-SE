import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse
from odatse.util.read_ts import read_Ts
from odatse.solver.analytical import Solver


def test_read_ts_bmax_error_names_bmax():
    """The invalid-bmax branch used to report 'bmin' in its message."""
    with pytest.raises(ValueError, match="bmax"):
        read_Ts({"bmin": 1.0, "bmax": -1.0}, numT=4)


def test_linear_regression_dimension_error_says_three():
    """linear_regression_test needs dimension==3; the message used to say
    dimension=2."""
    info = odatse.Info({
        "base": {"dimension": 2},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical",
                   "function_name": "linear_regression_test"},
    })
    with pytest.raises(RuntimeError, match="dimension=3"):
        Solver(info)
