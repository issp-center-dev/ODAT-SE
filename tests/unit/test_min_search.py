# sys.path and odatse.mpi.setup() are handled by conftest.py
import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.solver.function
import odatse.algorithm.min_search as min_search


def _run_minsearch(workdir, unit_list, record, minimize=None, run=True, fn=None):
    if fn is None:
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
            "minimize": {"maxiter": 3, "maxfev": 10} if minimize is None else minimize,
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
    if run:
        alg.main()
    return x0, alg


def test_initial_evaluation_uses_unit_scaling(tmp_path, monkeypatch):
    """The initial evaluation f0 must submit initial_list / unit_list,
    consistently with every later evaluation through _f_calc."""
    monkeypatch.chdir(tmp_path)
    record = []
    unit_list = [2.0, 2.0]
    x0, _ = _run_minsearch(tmp_path, unit_list=unit_list, record=record)
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


def test_method_defaults_to_nelder_mead(tmp_path, monkeypatch):
    """Without [algorithm.minimize] method, the historical Nelder-Mead
    behavior (including its default tolerances) must be preserved."""
    monkeypatch.chdir(tmp_path)
    record = []
    x0, alg = _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                             minimize={"maxiter": 200, "maxfev": 1000})
    assert alg.method == "Nelder-Mead"
    # ODAT-SE-specific keys must not leak into the scipy options
    assert set(alg.minimize_options) == {"maxiter", "maxfev"}
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    assert alg.itera is not None and alg.funcalls is not None
    # return_all is enabled by default for Nelder-Mead
    assert alg.allvecs is not None


def test_method_powell_converges(tmp_path, monkeypatch):
    """method = "Powell" runs through scipy.optimize.minimize and converges;
    bounds are handled by scipy for this method."""
    monkeypatch.chdir(tmp_path)
    record = []
    _, alg = _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                            minimize={"method": "Powell", "maxiter": 100, "maxfev": 1000})
    assert alg.method == "Powell"
    assert "method" not in alg.minimize_options
    np.testing.assert_allclose(alg.xopt, [0.0, 0.0], atol=1e-3)
    # initial_simplex/return_all are Nelder-Mead defaults and must not be
    # injected for other methods
    assert alg.allvecs is None


def test_method_cobyla_without_nit(tmp_path, monkeypatch):
    """COBYLA's OptimizeResult carries no nit/allvecs attribute; the result
    handling and res.txt output must not crash on their absence."""
    monkeypatch.chdir(tmp_path)
    record = []
    x0, alg = _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                             minimize={"method": "COBYLA", "maxiter": 200})
    assert alg.fopt < float(np.sum(x0 * x0))
    # res.txt files are written into output/<rank>/ (and the cwd by _post);
    # none of them may contain a literal "None" from missing nit/nfev
    res_files = list(tmp_path.glob("**/res.txt"))
    assert res_files
    for f in res_files:
        assert "None" not in f.read_text()


def test_unknown_option_raises(tmp_path, monkeypatch):
    """An option name the selected method does not accept must abort before
    the optimization instead of being silently ignored by scipy."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(RuntimeError, match="Unknown solver options"):
        _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                       minimize={"maxiter": 10, "no_such_option": 42})


def test_option_valid_for_other_method_raises(tmp_path, monkeypatch):
    """An option that belongs to a different method (gtol is for gradient
    methods, not Nelder-Mead) must abort as well."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(RuntimeError, match="Unknown solver options"):
        _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                       minimize={"gtol": 1e-5})


def test_invalid_method_raises(tmp_path, monkeypatch):
    """A method name scipy does not know must raise, not run."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(ValueError):
        _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                       minimize={"method": "no-such-method"})


def test_basinhopping_converges(tmp_path, monkeypatch):
    """basinhopping with a Nelder-Mead local minimizer finds the global
    minimum of a double-well function whose nearest local minimum from the
    initial point is not the global one."""
    monkeypatch.chdir(tmp_path)
    record = []

    def double_well(x):
        # wells at x0 = +/-2 with f(+2) = -1 (local) and f(-2) = -3 (global);
        # the initial point [2, 2] sits in the +2 well
        record.append(np.array(x, copy=True))
        return float(((x[0] ** 2 - 4) ** 2) / 8.0 + 0.5 * x[0] - 2.0 + x[1] ** 2)

    x0, alg = _run_minsearch(
        tmp_path, unit_list=[1.0, 1.0], record=record, fn=double_well,
        minimize={"maxiter": 500, "maxfev": 2000,
                  "basinhopping": {"niter": 20, "stepsize": 2.0, "T": 1.0}})
    assert alg.basinhopping_params == {"niter": 20, "stepsize": 2.0, "T": 1.0}
    # basinhopping keys must not leak into the minimize options
    assert set(alg.minimize_options) == {"maxiter", "maxfev"}
    # global minimum is near x = (-2 - eps, 0), f ~ -3; the +2 well only
    # reaches f ~ -1, so f < -2 shows the hop out of the initial well
    assert alg.fopt < -2.0
    # per-hop history is recorded and written out; scipy runs niter + 1
    # local minimizations (the extra one from the initial point)
    assert len(alg.hop_history) == 21
    assert list(tmp_path.glob("**/BasinHoppingData.txt"))


def test_basinhopping_flag_true(tmp_path, monkeypatch):
    """basinhopping = true (bare boolean) enables basinhopping with scipy
    default parameters."""
    monkeypatch.chdir(tmp_path)
    record = []
    _, alg = _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                            minimize={"basinhopping": True}, run=False)
    assert alg.basinhopping_params == {}
    assert "basinhopping" not in alg.minimize_options


def test_basinhopping_hops_stay_in_range(tmp_path, monkeypatch):
    """Even with a stepsize far larger than the search region, hops are
    clipped into [min_list, max_list]."""
    monkeypatch.chdir(tmp_path)
    record = []
    _, alg = _run_minsearch(
        tmp_path, unit_list=[1.0, 1.0], record=record,
        minimize={"maxiter": 200, "maxfev": 1000,
                  "basinhopping": {"niter": 5, "stepsize": 100.0}})
    for hop in alg.hop_history:
        x = np.array(hop[1:3])
        assert np.all((x >= alg.min_list) & (x <= alg.max_list))


def test_basinhopping_unknown_param_raises(tmp_path, monkeypatch):
    """An argument scipy.optimize.basinhopping does not accept must abort
    with a message pointing at the input file section."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(RuntimeError, match="basinhopping"):
        _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                       minimize={"basinhopping": {"no_such_param": 1}})


def test_basinhopping_reserved_param_raises(tmp_path, monkeypatch):
    """Arguments managed by ODAT-SE (take_step, seed, ...) cannot be set
    from the input file."""
    monkeypatch.chdir(tmp_path)
    record = []
    with pytest.raises(ValueError, match="managed by ODAT-SE"):
        _run_minsearch(tmp_path, unit_list=[1.0, 1.0], record=record,
                       minimize={"basinhopping": {"take_step": "foo"}},
                       run=False)
