import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
# insert, not append: an installed odatse package must not shadow src/
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.mpi
import odatse.solver.function

try:
    odatse.mpi.setup()
except RuntimeError:
    pass  # already set up by another test module in this session

import odatse.algorithm.min_search as min_search


def _run_minsearch(workdir, unit_list, record):
    def fn(x):
        record.append(np.array(x, copy=True))
        return float(np.sum(x * x))

    inp = {
        "base": {"dimension": 2, "output_dir": str(workdir / "output")},
        "algorithm": {
            "name": "minsearch",
            "seed": 1,
            "param": {
                "min_list": [-5.0, -5.0],
                "max_list": [5.0, 5.0],
                "initial_list": [2.0, 2.0],
                "unit_list": unit_list,
            },
            "minimize": {"maxiter": 3, "maxfev": 10},
        },
        "solver": {"name": "function"},
        "runner": {},
    }
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(fn)
    runner = odatse.Runner(solver, info)
    alg = min_search.Algorithm(info, runner)
    alg.main()


def test_initial_evaluation_uses_unit_scaling(tmp_path, monkeypatch):
    """The initial evaluation f0 must submit initial_list / unit_list,
    consistently with every later evaluation through _f_calc."""
    monkeypatch.chdir(tmp_path)
    record = []
    _run_minsearch(tmp_path, unit_list=[2.0, 2.0], record=record)
    # initial point [2, 2] with unit_list [2, 2] must arrive at the solver
    # as [1, 1]; the unfixed code submitted the unscaled [2, 2]
    np.testing.assert_allclose(record[0], [1.0, 1.0])


def test_run_with_prerelease_scipy_version(tmp_path, monkeypatch):
    """The scipy version gate must not crash on pre-release versions
    like '1.16.0rc1' (int('0rc1') raised ValueError)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(min_search.scipy, "__version__", "1.16.0rc1")
    record = []
    _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record)
    assert len(record) > 0
