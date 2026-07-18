# sys.path and odatse.mpi.setup() are handled by conftest.py
import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.solver.function
import odatse.algorithm.global_search as global_search


def _run_global_search(workdir, unit_list, record, global_search_params=None,
                       run=True, fn=None):
    if fn is None:
        def fn(x):
            record.append(np.array(x, copy=True))
            return float(np.sum(x * x))

    if global_search_params is None:
        global_search_params = {"maxiter": 20, "popsize": 8, "tol": 0.01}

    inp = {
        "base": {"dimension": 2, "output_dir": str(workdir / "output")},
        "algorithm": {
            "name": "global_search",
            "seed": 1,
            "param": {
                "min_list": [-5.0, -5.0],
                "max_list": [5.0, 5.0],
                "unit_list": unit_list,
            },
            "global_search": global_search_params,
        },
        "solver": {"name": "function"},
        "runner": {},
    }
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(fn)
    runner = odatse.Runner(solver, info)
    alg = global_search.Algorithm(info, runner)
    if run:
        alg.main()
    return alg


def test_de_converges(tmp_path, monkeypatch):
    """Differential evolution finds the minimum of a quadratic function;
    result and output files are produced."""
    monkeypatch.chdir(tmp_path)
    record = []
    alg = _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=record)
    assert alg.method == "differential_evolution"
    # the final result is broadcast, so every algorithm rank can check it
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    assert alg.fopt < 1e-4
    # every rank that evaluated points has its own function-call history
    if len(alg.fev_history) > 0:
        assert list(tmp_path.glob("**/History_FunctionCall.txt"))
    # iteration history and result summary exist on rank 0 only
    if odatse.mpi.algrank() == 0:
        assert len(alg.iter_history) > 0
        assert list(tmp_path.glob("**/GenerationData.txt"))
        content = (tmp_path / "output" / "res.txt").read_text()
        assert content.startswith("fx = ")
        assert "None" not in content


def test_de_multimodal(tmp_path, monkeypatch):
    """DE reaches the global minimum of a double-well function whose two
    wells have different depths (global one at x1 = -2)."""
    monkeypatch.chdir(tmp_path)
    record = []

    def double_well(x):
        record.append(np.array(x, copy=True))
        return float(((x[0] ** 2 - 4) ** 2) / 8.0 + 0.5 * x[0] - 2.0 + x[1] ** 2)

    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record, fn=double_well,
        global_search_params={"maxiter": 40, "popsize": 10, "tol": 0.001})
    assert alg.fopt < -2.0
    assert alg.xopt[0] < 0.0


def test_unit_scaling(tmp_path, monkeypatch):
    """Points must arrive at the solver divided by unit_list: with
    unit_list = [2, 2] and bounds [-5, 5], every submitted point lies in
    [-2.5, 2.5]."""
    monkeypatch.chdir(tmp_path)
    record = []
    _run_global_search(tmp_path, unit_list=[2.0, 2.0], record=record,
                       global_search_params={"maxiter": 3, "popsize": 6})
    assert len(record) > 0
    pts = np.array(record)
    assert np.all(np.abs(pts) <= 2.5 + 1e-12)


def test_method_aliases(tmp_path, monkeypatch):
    """"DE" (default), "de" and "differential_evolution" all select the DE
    routine."""
    monkeypatch.chdir(tmp_path)
    for params in ({}, {"method": "de"}, {"method": "differential_evolution"}):
        alg = _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                                 global_search_params=params, run=False)
        assert alg.method == "differential_evolution"
        assert "method" not in alg.opt_params


def test_unknown_method_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"method": "no-such-method"},
                           run=False)


def test_not_implemented_method_raises(tmp_path, monkeypatch):
    """shgo / direct are recognized but not implemented yet."""
    monkeypatch.chdir(tmp_path)
    for m in ("shgo", "direct"):
        with pytest.raises(NotImplementedError):
            _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                               global_search_params={"method": m}, run=False)


def test_unknown_param_raises(tmp_path, monkeypatch):
    """An argument scipy.optimize.differential_evolution does not accept
    must abort with a message pointing at the input file section.

    Under MPI, rank 0 raises RuntimeError while the other ranks exit
    silently with SystemExit(0): main() catches OtherAlgorithmProcessError
    and leaves the error reporting to the failing rank."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises((RuntimeError, SystemExit)):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"no_such_param": 1})


def test_reserved_param_raises(tmp_path, monkeypatch):
    """Arguments managed by ODAT-SE (workers, seed, ...) cannot be set from
    the input file."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="managed by ODAT-SE"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"workers": 4}, run=False)
