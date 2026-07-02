# sys.path and odatse.mpi.setup() are handled by conftest.py
import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.solver.function
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
    # under mpirun only walker 0 gets the configured initial_list; the other
    # ranks draw a random initial point, so capture this rank's actual one
    x0 = np.array(alg.initial_list, dtype=float, copy=True)
    alg.main()
    return x0


def test_initial_evaluation_uses_unit_scaling(tmp_path, monkeypatch):
    """The initial evaluation f0 must submit initial_list / unit_list,
    consistently with every later evaluation through _f_calc."""
    monkeypatch.chdir(tmp_path)
    record = []
    unit_list = [2.0, 2.0]
    x0 = _run_minsearch(tmp_path, unit_list=unit_list, record=record)
    # the initial point must arrive at the solver divided by unit_list
    # (e.g. [2, 2] -> [1, 1]); the unfixed code submitted it unscaled
    np.testing.assert_allclose(record[0], x0 / np.array(unit_list))


def test_run_with_prerelease_scipy_version(tmp_path, monkeypatch):
    """The scipy version gate must not crash on pre-release versions
    like '1.16.0rc1' (int('0rc1') raised ValueError)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(min_search.scipy, "__version__", "1.16.0rc1")
    record = []
    _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record)
    assert len(record) > 0
