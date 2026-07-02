import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

import odatse
import odatse.algorithm.min_search as min_search_mod
from odatse.solver.analytical import Solver


UNIT = [2.0, 4.0]


def _build_algorithm():
    info = odatse.Info({
        "base": {"dimension": 2, "output_dir": "."},
        "algorithm": {
            "name": "minsearch",
            "seed": 12345,
            "param": {
                "min_list": [-5.0, -5.0],
                "max_list": [5.0, 5.0],
                "unit_list": UNIT,
            },
        },
        "solver": {"name": "analytical", "function_name": "quadratics"},
    })
    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    return min_search_mod.Algorithm(info, runner, run_mode="initial")


class _Result:
    pass


def test_objective_does_not_mutate_optimizer_array(monkeypatch, tmp_path):
    """The objective passed to scipy.optimize.minimize must not modify the
    array it receives (which scipy owns), and must submit the *scaled*
    coordinates x / unit_list to the solver."""
    monkeypatch.chdir(tmp_path)
    alg = _build_algorithm()

    submitted = []
    orig_submit = alg.runner.submit

    def spy_submit(x, args=()):
        submitted.append(np.array(x, copy=True))
        return orig_submit(x, args)

    monkeypatch.setattr(alg.runner, "submit", spy_submit)

    probe = np.array([1.5, -2.0])
    captured = {}

    def fake_minimize(func, x0, **kwargs):
        before = probe.copy()
        val = func(probe, *kwargs.get("args", ()))
        captured["mutated"] = not np.array_equal(probe, before)
        x0 = np.asarray(x0, dtype=float)
        r = _Result()
        r.x, r.fun, r.nit, r.nfev, r.allvecs = x0, val, 0, 1, [x0]
        return r

    monkeypatch.setattr(min_search_mod, "minimize", fake_minimize)

    alg._prepare()
    alg._run()

    # scipy's array must be untouched by the objective
    assert captured["mutated"] is False
    np.testing.assert_array_equal(probe, np.array([1.5, -2.0]))

    # the value handed to the solver is the scaled probe
    np.testing.assert_allclose(submitted[-1], probe / np.array(UNIT))


def test_full_run_recovers_scaled_optimum(monkeypatch, tmp_path):
    """End-to-end: the solver minimum of quadratics is at the origin, so for
    any unit_list the recovered raw optimum is the origin. This exercises the
    real scipy optimizer and would diverge if the objective corrupted scipy's
    simplex via in-place scaling."""
    monkeypatch.chdir(tmp_path)
    alg = _build_algorithm()
    alg._prepare()
    alg._run()

    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    assert alg.fopt < 1e-6
