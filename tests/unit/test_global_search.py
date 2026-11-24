# sys.path and odatse.mpi.setup() are handled by conftest.py
import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.solver.function
import odatse.algorithm.global_search as global_search

# scipy.optimize.direct only exists in scipy >= 1.9; the algorithm module
# guards its import and keeps DE / shgo usable without it
requires_direct = pytest.mark.skipif(
    global_search.direct is None,
    reason="scipy.optimize.direct requires scipy >= 1.9")


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


def test_shgo_converges(tmp_path, monkeypatch):
    """shgo finds the minimum of a quadratic function and reports the list
    of local minima."""
    monkeypatch.chdir(tmp_path)
    record = []
    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record,
        global_search_params={"method": "shgo", "n": 32})
    assert alg.method == "shgo"
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    assert alg.fopt < 1e-4
    assert alg.xl is not None and len(alg.xl) >= 1
    if odatse.mpi.algrank() == 0:
        assert list(tmp_path.glob("**/LocalMinimaData.txt"))
        assert list(tmp_path.glob("**/IterationData.txt"))
        content = (tmp_path / "output" / "res.txt").read_text()
        assert content.startswith("fx = ")


def test_shgo_multimodal(tmp_path, monkeypatch):
    """shgo reaches the global minimum of the double-well function and
    enumerates both wells as local minima."""
    monkeypatch.chdir(tmp_path)
    record = []

    def double_well(x):
        record.append(np.array(x, copy=True))
        return float(((x[0] ** 2 - 4) ** 2) / 8.0 + 0.5 * x[0] - 2.0 + x[1] ** 2)

    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record, fn=double_well,
        global_search_params={"method": "shgo", "n": 64})
    assert alg.fopt < -2.0
    assert alg.xopt[0] < 0.0
    # both wells (x1 ~ -2 and x1 ~ +2) should be found as local minima
    assert len(alg.xl) >= 2
    signs = {np.sign(x[0]) for x in alg.xl}
    assert signs == {-1.0, 1.0}
    # the local minima list is sorted, funl[0] is the global minimum
    np.testing.assert_allclose(alg.funl[0], alg.fopt)


def test_shgo_unknown_param_raises(tmp_path, monkeypatch):
    """fail-fast also applies to the shgo argument list."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="shgo"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"method": "shgo",
                                                 "no_such_param": 1},
                           run=False)


@requires_direct
def test_direct_converges(tmp_path, monkeypatch):
    """direct finds the minimum of a quadratic function. It runs entirely
    on rank 0; under MPI the other ranks stay idle but still receive the
    broadcast result."""
    monkeypatch.chdir(tmp_path)
    record = []
    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record,
        global_search_params={"method": "direct", "maxfun": 2000})
    assert alg.method == "direct"
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-2)
    assert alg.fopt < 1e-3
    if odatse.mpi.algrank() == 0:
        assert list(tmp_path.glob("**/IterationData.txt"))
        content = (tmp_path / "output" / "res.txt").read_text()
        assert content.startswith("fx = ")
        assert "None" not in content


@requires_direct
def test_direct_multimodal(tmp_path, monkeypatch):
    """direct reaches the global minimum of the double-well function."""
    monkeypatch.chdir(tmp_path)
    record = []

    def double_well(x):
        record.append(np.array(x, copy=True))
        return float(((x[0] ** 2 - 4) ** 2) / 8.0 + 0.5 * x[0] - 2.0 + x[1] ** 2)

    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record, fn=double_well,
        global_search_params={"method": "direct", "maxfun": 2000})
    assert alg.fopt < -2.0
    assert alg.xopt[0] < 0.0


@requires_direct
def test_direct_unknown_param_raises(tmp_path, monkeypatch):
    """fail-fast also applies to the direct argument list."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="direct"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"method": "direct",
                                                 "no_such_param": 1},
                           run=False)


def test_dual_annealing_converges(tmp_path, monkeypatch):
    """dual_annealing finds the minimum of a quadratic function. Like
    direct it runs entirely on rank 0; under MPI the other ranks stay idle
    but still receive the broadcast result."""
    monkeypatch.chdir(tmp_path)
    record = []
    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record,
        global_search_params={"method": "dual_annealing", "maxiter": 20,
                              "maxfun": 2000})
    assert alg.method == "dual_annealing"
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    assert alg.fopt < 1e-4
    if odatse.mpi.algrank() == 0:
        assert len(alg.iter_history) > 0
        files = list(tmp_path.glob("**/MinimumData.txt"))
        assert len(files) == 1
        # exact writer contract: header line with the labels, then one
        # space-separated str()-formatted row per recorded minimum
        expected = "#no {} R-factor context\n".format(" ".join(alg.label_list))
        expected += "".join(" ".join(map(str, row)) + "\n"
                            for row in alg.iter_history)
        assert files[0].read_text() == expected
        # rows are [index, *x, f, context], numbered from 0, with the
        # context an integer in {0, 1, 2}
        for i, row in enumerate(alg.iter_history):
            assert len(row) == 3 + len(alg.label_list)
            assert row[0] == i
            assert row[-1] in (0, 1, 2)
        content = (tmp_path / "output" / "res.txt").read_text()
        assert content.startswith("fx = ")
        assert "None" not in content


def test_dual_annealing_multimodal(tmp_path, monkeypatch):
    """dual_annealing reaches the global minimum of the double-well
    function."""
    monkeypatch.chdir(tmp_path)
    record = []

    def double_well(x):
        record.append(np.array(x, copy=True))
        return float(((x[0] ** 2 - 4) ** 2) / 8.0 + 0.5 * x[0] - 2.0 + x[1] ** 2)

    alg = _run_global_search(
        tmp_path, unit_list=[1.0, 1.0], record=record, fn=double_well,
        global_search_params={"method": "dual_annealing", "maxiter": 50,
                              "maxfun": 4000})
    assert alg.fopt < -2.0
    assert alg.xopt[0] < 0.0


def test_dual_annealing_unknown_param_raises(tmp_path, monkeypatch):
    """fail-fast also applies to the dual_annealing argument list."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="dual_annealing"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"method": "dual_annealing",
                                                 "no_such_param": 1},
                           run=False)


def test_dual_annealing_runtime_error_propagates(tmp_path, monkeypatch):
    """An exception raised by the objective while dual_annealing runs on
    rank 0 must propagate instead of hanging: the MSG_ABORT broadcast in
    _run releases the other algorithm ranks from the evaluation-server
    loop. Under MPI those ranks exit silently with SystemExit(0), like in
    test_runtime_typeerror_propagates."""
    monkeypatch.chdir(tmp_path)
    calls = [0]

    def broken(x):
        calls[0] += 1
        if calls[0] > 5:
            raise TypeError("broken objective")
        return float(np.sum(x * x))

    if odatse.mpi.algrank() == 0:
        # rank 0 drives the optimizer and evaluates every point itself, so
        # the objective's TypeError must propagate here (possibly wrapped),
        # never a silent exit
        with pytest.raises((TypeError, RuntimeError)) as excinfo:
            _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                               fn=broken,
                               global_search_params={
                                   "method": "dual_annealing",
                                   "maxiter": 5, "maxfun": 500})
        # the original TypeError must be preserved in the exception chain
        chain, e = [], excinfo.value
        while e is not None:
            chain.append(e)
            e = e.__cause__
        assert any(isinstance(c, TypeError) and "broken objective" in str(c)
                   for c in chain)
    else:
        # the idle ranks must be released by the MSG_ABORT broadcast and
        # exit silently with status 0
        with pytest.raises(SystemExit) as excinfo:
            _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                               fn=broken,
                               global_search_params={
                                   "method": "dual_annealing",
                                   "maxiter": 5, "maxfun": 500})
        assert excinfo.value.code == 0


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


def test_all_methods_recognized(tmp_path, monkeypatch):
    """Every documented method name resolves without NotImplementedError."""
    monkeypatch.chdir(tmp_path)
    methods = [("DE", "differential_evolution"), ("shgo", "shgo"),
               ("dual_annealing", "dual_annealing")]
    if global_search.direct is not None:
        methods.append(("direct", "direct"))
    for m, resolved in methods:
        alg = _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                                 global_search_params={"method": m}, run=False)
        assert alg.method == resolved


def test_unknown_param_raises(tmp_path, monkeypatch):
    """An argument scipy.optimize.differential_evolution does not accept
    must abort at construction time, before any solver evaluation, with a
    message pointing at the input file section (issue #76)."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(ValueError, match="global_search"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=record,
                           global_search_params={"no_such_param": 1},
                           run=False)
    assert record == []


def test_runtime_typeerror_propagates(tmp_path, monkeypatch):
    """A TypeError raised by the objective function during the optimization
    must propagate unchanged instead of being misreported as an
    input-configuration error (issue #76).

    Note that scipy's differential_evolution itself wraps exceptions from
    the population evaluation into a RuntimeError about the map-like
    callable, chaining the original via __cause__; what matters here is
    that the original TypeError stays in the chain and that ODAT-SE no
    longer replaces it with a message blaming [algorithm.global_search].
    Under MPI, ranks other than the failing one and rank 0 exit silently
    with SystemExit(0)."""
    monkeypatch.chdir(tmp_path)
    calls = [0]

    def broken(x):
        calls[0] += 1
        if calls[0] > 5:
            raise TypeError("broken objective")
        return float(np.sum(x * x))

    with pytest.raises((TypeError, RuntimeError, SystemExit)) as excinfo:
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           fn=broken,
                           global_search_params={"maxiter": 10, "popsize": 6})
    if not isinstance(excinfo.value, SystemExit):
        # the original TypeError must be preserved in the exception chain
        chain, e = [], excinfo.value
        while e is not None:
            chain.append(e)
            e = e.__cause__
        assert any(isinstance(c, TypeError) and "broken objective" in str(c)
                   for c in chain)
        # and no exception in the chain may misdirect to the input file
        assert all("global_search" not in str(c) for c in chain)


def test_reserved_param_raises(tmp_path, monkeypatch):
    """Arguments managed by ODAT-SE (workers, seed, ...) cannot be set from
    the input file."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="managed by ODAT-SE"):
        _run_global_search(tmp_path, unit_list=[1.0, 1.0], record=[],
                           global_search_params={"workers": 4}, run=False)
