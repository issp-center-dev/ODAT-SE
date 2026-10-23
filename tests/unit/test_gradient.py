# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Behavior contract of the optional gradient support (C1-C6 of
work/gradient-review/contract.md): Runner.gradient_unavailable_reason(),
Runner.submit_gradient(), use_gradient of min_search, its callback, and the
gradients of the analytical solver."""
import numpy as np
import pytest

pytest.importorskip("scipy")

import odatse
import odatse.mpi
import odatse.solver.function
import odatse.solver.analytical
import odatse.util.mapping
import odatse.util.limitation
import odatse.algorithm.min_search as min_search


# ---------------------------------------------------------------------------
# helpers


class _RecordingSolver(odatse.solver.function.Solver):
    """Function solver that records every evaluate()/gradient() call.

    ``grad`` is the gradient callable in solver units; ``supports`` sets the
    value of the supports_gradient property independently of ``grad`` so that
    a solver can claim no gradient support while still having a working
    gradient() (which must then not be reached).
    """

    def __init__(self, info, fn, grad=None, supports=True):
        super().__init__(info)
        self.set_function(fn)
        self._grad = grad
        self._supports = supports
        self.eval_calls = []
        self.grad_calls = []

    @property
    def supports_gradient(self):
        return self._supports

    def evaluate(self, x, args=()):
        self.eval_calls.append((np.array(x, dtype=float, copy=True), args))
        return super().evaluate(x, args)

    def gradient(self, x, args=()):
        self.grad_calls.append((np.array(x, dtype=float, copy=True), args))
        return self._grad(x)


def _quad_fn(c):
    c = np.asarray(c, dtype=float)

    def fn(x):
        return float(np.sum((x - c) ** 2))

    def grad(x):
        return 2.0 * (np.asarray(x, dtype=float) - c)

    return fn, grad


def _info(tmp_path, minimize=None, unit_list=(1.0, 1.0), runner=None,
          initial_list=(2.0, 2.0), min_list=(-5.0, -5.0), max_list=(5.0, 5.0)):
    inp = {
        "base": {"dimension": 2, "output_dir": str(tmp_path / "output")},
        "algorithm": {
            "name": "minsearch",
            "seed": 1,
            "label_list": ["a", "b"],
            "param": {
                "min_list": list(min_list),
                "max_list": list(max_list),
                "initial_list": list(initial_list),
                "unit_list": list(unit_list),
            },
            "minimize": {} if minimize is None else dict(minimize),
        },
        "solver": {"name": "function"},
        "runner": {} if runner is None else dict(runner),
    }
    return odatse.Info(inp)


def _make_runner(tmp_path, fn=None, grad=None, supports=True, runner=None,
                 mapping=None, limitation=None, monkeypatch=None, solsize=1,
                 **info_kw):
    """Build a Runner around a _RecordingSolver.

    odatse.mpi.solsize is patched (to ``solsize``) only *after* the Runner
    is constructed: the construction-time agreement of FailTogetherMeta
    consults solsize() and would try to talk to a non-existent solver group.
    """
    if fn is None:
        fn, default_grad = _quad_fn([0.0, 0.0])
        if grad is None:
            grad = default_grad
    info = _info(tmp_path, runner=runner, **info_kw)
    solver = _RecordingSolver(info, fn, grad, supports=supports)
    r = odatse.Runner(solver, info, mapping=mapping, limitation=limitation)
    if monkeypatch is not None:
        monkeypatch.setattr(odatse.mpi, "solsize", lambda: solsize)
    return info, solver, r


def _make_alg(tmp_path, minimize, fn=None, grad=None, supports=True,
              unit_list=(1.0, 1.0), runner=None, mapping=None, **info_kw):
    if fn is None:
        fn, default_grad = _quad_fn([0.0, 0.0])
        if grad is None:
            grad = default_grad
    info = _info(tmp_path, minimize=minimize, unit_list=unit_list,
                 runner=runner, **info_kw)
    solver = _RecordingSolver(info, fn, grad, supports=supports)
    r = odatse.Runner(solver, info, mapping=mapping)
    alg = min_search.Algorithm(info, r)
    return alg, solver, r


def _proc_dir(tmp_path):
    return tmp_path / "output" / str(odatse.mpi.algrank())


def _read_table(path):
    lines = path.read_text().splitlines()
    header = lines[0]
    rows = [np.array([float(v) for v in L.split()]) for L in lines[1:] if L.strip()]
    return header, rows


def _assert_no_finite_difference_probe(solver, x0):
    """Without a jac, scipy estimates the gradient at x0 by evaluating
    x0 + h*e_i (one coordinate shifted by h ~ 1e-8); with the solver
    gradient as jac no such axis-aligned probe is made. So no evaluate()
    call (in solver units == algorithm units here) may differ from x0 in
    exactly one coordinate by less than 1e-6. (TNC legitimately evaluates
    at x0 + tiny * d along a search direction d for its Hessian-vector
    products; that moves all coordinates and is not counted.)"""
    def is_probe(x):
        d = np.abs(x - x0)
        return np.count_nonzero(d) == 1 and np.max(d) < 1e-6
    near = [x for x, _ in solver.eval_calls if is_probe(x)]
    assert not near, f"finite-difference probes around x0: {near[:3]}"


def _assert_all_nan(g, shape):
    g = np.asarray(g)
    assert g.shape == shape
    assert np.all(np.isnan(g)), g


# ---------------------------------------------------------------------------
# C1. Runner.gradient_unavailable_reason()


def test_c1_reason_none_when_gradient_available(tmp_path, monkeypatch):
    """C1: supported solver + trivial mapping + solver group of 1 -> None.
    Fails if the method is missing or reports a reason in the valid setup."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch)
    assert r.gradient_unavailable_reason() is None


def test_c1_reason_for_unsupported_solver(tmp_path, monkeypatch):
    """C1: a solver whose supports_gradient is false yields a reason string
    (even though its gradient() would work). Fails if None is returned."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, supports=False)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason


def test_c1_reason_for_affine_mapping(tmp_path, monkeypatch):
    """C1: an Affine mapping yields a reason string even with a
    gradient-capable solver. Fails if the mapping is not taken into account."""
    mapping = odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, mapping=mapping)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason


def test_c1_reason_for_parallel_solver(tmp_path, monkeypatch):
    """C1: a solver group of 2 or more processes yields a reason string.
    Fails if solsize() is not consulted."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, solsize=2)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# C2. Runner.submit_gradient()


@pytest.mark.parametrize("situation", ["unsupported", "affine", "parallel"])
def test_c2_submit_gradient_raises_when_unavailable(tmp_path, monkeypatch, situation):
    """C2: in every C1 situation that yields a reason, submit_gradient raises
    NotImplementedError without reaching the solver's gradient(). The
    unsupported solver has a working gradient(), so the error must come from
    the Runner's own check. Fails if the gradient is computed anyway."""
    kw = {"solsize": 2 if situation == "parallel" else 1}
    if situation == "unsupported":
        kw["supports"] = False
    if situation == "affine":
        kw["mapping"] = odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, **kw)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(np.array([1.0, 1.0]), (1, 0))
    assert solver.grad_calls == []


@pytest.mark.parametrize("situation", ["unsupported", "affine", "parallel"])
def test_c2_unavailable_raises_even_when_limitation_violated(tmp_path, monkeypatch, situation):
    """C2: the NotImplementedError of every C1 situation takes precedence
    over the limitation: with an x that violates the limitation,
    submit_gradient still raises instead of returning the NaN array of the
    limitation path. Fails if the limitation is judged before gradient
    availability (an unusable gradient would then go unnoticed as NaN)."""
    kw = {"solsize": 2 if situation == "parallel" else 1}
    if situation == "unsupported":
        kw["supports"] = False
    if situation == "affine":
        kw["mapping"] = odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
    # x[0] > 0 is required; x below violates it
    kw["limitation"] = odatse.util.limitation.Inequality(
        np.array([[1.0, 0.0]]), np.array([0.0]), True)
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, **kw)
    x = np.array([-1.0, 1.0])
    assert not r.limitation.judge(x)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(x, (1, 0))
    assert solver.grad_calls == []


def test_c2_limitation_violation_returns_nan(tmp_path, monkeypatch):
    """C2: an x violating the limitation gets an all-NaN array of x's shape
    and gradient() is not called. Fails if zeros (or anything finite) are
    returned, or if the solver is consulted."""
    # x[0] > 0 is required
    lim = odatse.util.limitation.Inequality(
        np.array([[1.0, 0.0]]), np.array([0.0]), True)
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, limitation=lim)
    g = r.submit_gradient(np.array([-1.0, 1.0]), (1, 0))
    _assert_all_nan(g, (2,))
    assert solver.grad_calls == []


def test_c2_returns_solver_gradient_and_passes_x_and_args(tmp_path, monkeypatch):
    """C2: the solver's gradient is returned as a float ndarray; gradient()
    receives the (trivially mapped) x and the args unchanged. Fails if the
    value, x or args are altered on the way."""
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=lambda x: np.array([3.5, -1.25]))
    x = np.array([0.5, -0.75])
    g = r.submit_gradient(x, (7, 3))
    assert isinstance(g, np.ndarray)
    assert g.dtype == np.float64
    np.testing.assert_array_equal(g, [3.5, -1.25])
    assert len(solver.grad_calls) == 1
    xs, args = solver.grad_calls[0]
    np.testing.assert_array_equal(xs, x)
    assert args == (7, 3)


def test_c2_converts_non_float_gradient_to_float_ndarray(tmp_path, monkeypatch):
    """C2: a gradient() returning a plain list of ints is still returned as a
    float ndarray. Fails if the raw list (or an int array) is passed through."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=lambda x: [3, -4])
    g = r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    assert isinstance(g, np.ndarray)
    assert np.issubdtype(g.dtype, np.floating)
    np.testing.assert_array_equal(g, [3.0, -4.0])


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("bad", [
    np.array([1.0]),
    np.array([1.0, 2.0, 3.0]),
    1.0,
    np.ones((2, 1)),
], ids=["len1", "len3", "scalar", "2x1"])
def test_c2_wrong_gradient_shape_raises_valueerror(tmp_path, monkeypatch, bad, ignore_error):
    """C2: a gradient() whose shape differs from x's raises ValueError, also
    with ignore_error = true. Fails if the bad value is returned (or
    broadcast) silently or replaced by NaN."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=lambda x: bad,
                           runner={"ignore_error": ignore_error})
    with pytest.raises(ValueError):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))


def _raising(exc):
    def grad(x):
        raise exc
    return grad


def test_c2_runtime_error_propagates_without_ignore_error(tmp_path, monkeypatch):
    """C2: RuntimeError from gradient() propagates when ignore_error is off
    (the default). Fails if it is swallowed."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=_raising(RuntimeError("grad boom")))
    with pytest.raises(RuntimeError, match="grad boom"):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))


def test_c2_runtime_error_with_ignore_error_returns_nan(tmp_path, monkeypatch):
    """C2: RuntimeError from gradient() with ignore_error = true gives an
    all-NaN array of x's shape (not zeros). Fails if zeros are returned or
    the error propagates."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=_raising(RuntimeError("grad boom")),
                           runner={"ignore_error": True})
    g = r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    _assert_all_nan(g, (2,))


@pytest.mark.parametrize("ignore_error", [False, True])
def test_c2_not_implemented_error_always_propagates(tmp_path, monkeypatch, ignore_error):
    """C2: NotImplementedError (a RuntimeError subclass) from gradient()
    propagates regardless of ignore_error. Fails if ignore_error turns it
    into NaN/zeros."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=_raising(NotImplementedError("no grad here")),
                           runner={"ignore_error": ignore_error})
    with pytest.raises(NotImplementedError, match="no grad here"):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("exc_type", [ValueError, TypeError])
def test_c2_non_runtime_error_always_propagates(tmp_path, monkeypatch, exc_type, ignore_error):
    """C2: non-RuntimeError exceptions from gradient() propagate regardless
    of ignore_error. Fails if ignore_error swallows them."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=_raising(exc_type("bad grad")),
                           runner={"ignore_error": ignore_error})
    with pytest.raises(exc_type, match="bad grad"):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))


# ---------------------------------------------------------------------------
# C3. min_search input validation at construction


@pytest.mark.parametrize("value", ["false", "true", 1, 0, ""],
                         ids=["str-false", "str-true", "int1", "int0", "empty"])
def test_c3_non_bool_use_gradient_raises(tmp_path, value):
    """C3: a non-bool use_gradient (including falsy 0 and "") is rejected
    with ValueError at construction, before any solver call. Fails if it is
    coerced with bool() (e.g. "false" -> True) or silently ignored."""
    _solver_untouched_after_failed_construction(
        tmp_path, {"method": "BFGS", "use_gradient": value})


def _solver_untouched_after_failed_construction(tmp_path, minimize, **kw):
    """Build solver+runner, then expect Algorithm() to raise ValueError and
    check the solver was never called."""
    fn, grad = _quad_fn([0.0, 0.0])
    info = _info(tmp_path, minimize=minimize)
    solver = _RecordingSolver(info, fn, grad, supports=kw.pop("supports", True))
    r = odatse.Runner(solver, info, mapping=kw.pop("mapping", None))
    patch = kw.pop("patch", None)
    if patch is not None:
        patch(r)
    with pytest.raises(ValueError):
        min_search.Algorithm(info, r)
    assert solver.eval_calls == []
    assert solver.grad_calls == []


@pytest.mark.parametrize("method", [
    None, "Nelder-Mead", "Powell", "COBYLA", "COBYQA",
    "nelder-mead", "POWELL", "cobyla", "CobyQA",
])
def test_c3_gradient_free_method_with_use_gradient_raises(tmp_path, method):
    """C3: use_gradient = true with a gradient-free method (case-insensitive;
    omitted method means Nelder-Mead) raises ValueError at construction
    without calling the solver. Fails if the gradient is silently unused."""
    minimize = {"use_gradient": True}
    if method is not None:
        minimize["method"] = method
    _solver_untouched_after_failed_construction(tmp_path, minimize)


def test_c3_unsupported_solver_with_use_gradient_raises(tmp_path):
    """C3: use_gradient = true with a solver whose supports_gradient is false
    raises ValueError at construction. Fails if it silently falls back to
    finite differences."""
    _solver_untouched_after_failed_construction(
        tmp_path, {"method": "BFGS", "use_gradient": True}, supports=False)


def test_c3_affine_mapping_with_use_gradient_raises(tmp_path):
    """C3: use_gradient = true with an Affine mapping on the Runner raises
    ValueError at construction (not NotImplementedError later during the
    run). Fails if the check is deferred to submit_gradient."""
    _solver_untouched_after_failed_construction(
        tmp_path, {"method": "BFGS", "use_gradient": True},
        mapping=odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2)))


def test_c3_gradient_unavailable_reason_with_use_gradient_raises(tmp_path, monkeypatch):
    """C3: use_gradient = true raises ValueError at construction when
    runner.gradient_unavailable_reason() returns a string (the path used for
    a parallel solver; solsize itself must not be patched here). Fails if the
    Runner is not asked for a reason."""
    def patch(r):
        monkeypatch.setattr(r, "gradient_unavailable_reason",
                            lambda: "solver group has 2 processes", raising=False)

    _solver_untouched_after_failed_construction(
        tmp_path, {"method": "BFGS", "use_gradient": True}, patch=patch)


@pytest.mark.parametrize("use_gradient", [False, "omitted"])
def test_c3_no_error_without_use_gradient(tmp_path, monkeypatch, use_gradient):
    """C3: with use_gradient false or omitted, none of the gradient
    conditions (gradient-free method, unsupported solver, Affine mapping,
    a reported reason) is an error, and use_gradient is not forwarded to the
    scipy options. Fails if validation runs regardless of use_gradient."""
    minimize = {"method": "Nelder-Mead", "maxiter": 5}
    if use_gradient != "omitted":
        minimize["use_gradient"] = use_gradient
    fn, grad = _quad_fn([0.0, 0.0])
    info = _info(tmp_path, minimize=minimize)
    solver = _RecordingSolver(info, fn, grad, supports=False)
    r = odatse.Runner(solver, info,
                      mapping=odatse.util.mapping.Affine(A=np.eye(2), b=np.zeros(2)))
    monkeypatch.setattr(r, "gradient_unavailable_reason",
                        lambda: "some reason", raising=False)
    alg = min_search.Algorithm(info, r)
    assert "use_gradient" not in alg.minimize_options


def test_c3_use_gradient_not_in_minimize_options(tmp_path):
    """C3: a valid use_gradient = true is consumed by ODAT-SE and not passed
    to scipy. Fails if it leaks into minimize_options."""
    alg, _, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True,
                                     "gtol": 1e-6})
    assert "use_gradient" not in alg.minimize_options
    assert alg.minimize_options == {"gtol": 1e-6}


# ---------------------------------------------------------------------------
# C4. min_search runs with use_gradient = true


_C = np.array([1.0, -0.5])


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "CG", "TNC", "SLSQP"])
def test_c4_gradient_is_used_and_converges(tmp_path, method):
    """C4: with use_gradient = true a gradient method calls the solver's
    gradient() and converges to the minimum of a quadratic, and scipy does
    not fall back to finite differences (no evaluate() probe within 1e-6 of
    the initial point). Fails if gradient() is never called, if the gradient
    is computed but not handed to scipy as jac, or the optimum is missed."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(tmp_path, {"method": method, "use_gradient": True},
                               fn=fn, grad=grad)
    x0 = np.array(alg.initial_list, dtype=float, copy=True)
    alg.main()
    assert len(solver.grad_calls) >= 2
    _assert_no_finite_difference_probe(solver, x0)
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-4)


# a non-separable quadratic in solver units, with an analytic gradient
def _skew_fn(y):
    return float((y[0] - 1.0) ** 2 + 3.0 * (y[1] + 0.5) ** 2 + y[0] * y[1])


def _skew_grad(y):
    return np.array([2.0 * (y[0] - 1.0) + y[1], 6.0 * (y[1] + 0.5) + y[0]])


_UNIT = np.array([2.0, 0.5])


def _run_skew(tmp_path, method="BFGS"):
    alg, solver, _ = _make_alg(
        tmp_path, {"method": method, "use_gradient": True},
        fn=_skew_fn, grad=_skew_grad, unit_list=tuple(_UNIT),
        initial_list=(3.0, 2.0))
    x0 = np.array(alg.initial_list, dtype=float, copy=True)
    alg.main()
    return alg, solver, x0


def test_c4_history_gradient_matches_numerical_derivative(tmp_path):
    """C4 (unit_list chain rule): the first data row of History_Gradient.txt
    holds the initial point and a gradient equal to the central difference of
    F(x) = f(x / unit_list) at that point, with unit_list = [2, 0.5]. A wrong
    scaling (no division, multiplication, ...) changes the row; convergence
    alone would not detect it. Fails on any chain-rule error."""
    alg, _, x0 = _run_skew(tmp_path)
    path = _proc_dir(tmp_path) / "History_Gradient.txt"
    assert path.is_file()
    _, rows = _read_table(path)
    row = rows[0]
    np.testing.assert_allclose(row[1:3], x0, rtol=1e-12)

    def F(x):
        return _skew_fn(np.asarray(x) / _UNIT)

    h = 1e-5
    num = np.array([
        (F(x0 + h * e) - F(x0 - h * e)) / (2 * h) for e in np.eye(2)
    ])
    # the initial point must not be stationary for the check to mean anything
    assert np.linalg.norm(num) > 0.1
    np.testing.assert_allclose(row[4:6], num, rtol=1e-5, atol=1e-7)


def test_c4_history_gradient_file_format(tmp_path):
    """C4: History_Gradient.txt is written to output/<rank>/, with header
    '#No <labels> fx grad_<labels>' and rows (No, x, fx, grad) in algorithm
    units; the first row is the initial point with fx = F(x0) and
    grad = solver gradient / unit_list. Fails on a missing file, wrong
    header, wrong column layout or wrong first row."""
    alg, _, x0 = _run_skew(tmp_path)
    path = _proc_dir(tmp_path) / "History_Gradient.txt"
    assert path.is_file()
    header, rows = _read_table(path)
    assert header.split() == ["#No", "a", "b", "fx", "grad_a", "grad_b"]
    assert len(rows) >= 1
    for row in rows:
        assert row.shape == (6,)
    first = rows[0]
    np.testing.assert_allclose(first[1:3], x0, rtol=1e-12)
    np.testing.assert_allclose(first[3], _skew_fn(x0 / _UNIT), rtol=1e-12)
    np.testing.assert_allclose(first[4:6], _skew_grad(x0 / _UNIT) / _UNIT, rtol=1e-12)


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "SLSQP"])
def test_c4_gradient_not_recomputed_at_same_point(tmp_path, method):
    """C4: gradient() is never called twice at the same point once it has
    returned a finite value (unit_list = 1, so solver x == algorithm x).
    Fails if e.g. the history row and scipy both trigger a computation at
    the initial point, or the callback recomputes scipy's gradient."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(tmp_path, {"method": method, "use_gradient": True},
                               fn=fn, grad=grad)
    alg.main()
    pts = [tuple(x.tolist()) for x, _ in solver.grad_calls]
    assert len(pts) >= 2
    dup = {p for p in pts if pts.count(p) > 1}
    assert not dup, f"gradient() called repeatedly at {sorted(dup)}"


def test_c4_failed_gradient_is_recomputed(tmp_path):
    """C4: with ignore_error = true a RuntimeError from gradient() (NaN) is
    not memoized: the gradient at the first requested point (the initial
    point, needed both by the optimizer and by the History_Gradient row) is
    computed again. Fails if the NaN result is cached."""
    fn, grad = _quad_fn(_C)
    calls = [0]

    def flaky(x):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("transient failure")
        return grad(x)

    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=flaky, runner={"ignore_error": True})
    alg.main()
    assert len(solver.grad_calls) >= 2
    first = solver.grad_calls[0][0]
    again = [x for x, _ in solver.grad_calls[1:] if np.array_equal(x, first)]
    assert again, "gradient() was not called again at the point where it failed"


def test_c4_step_numbers_unique(tmp_path):
    """C4: within one optimization the step numbers of args with iset == 0
    passed to evaluate() and gradient() are pairwise distinct, including the
    initial evaluation (0, 0). Fails if gradient() always gets the same args
    or shares a step number with an evaluation."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=grad)
    alg.main()
    assert len(solver.grad_calls) >= 2
    grad_args = [tuple(a) for _, a in solver.grad_calls]
    assert len(set(grad_args)) > 1
    all_args = [tuple(a) for _, a in solver.eval_calls] + grad_args
    assert (0, 0) in [tuple(a) for _, a in solver.eval_calls]
    steps = [a[0] for a in all_args if a[1] == 0]
    dup = {s for s in steps if steps.count(s) > 1}
    assert not dup, f"duplicated step numbers with iset == 0: {sorted(dup)}"


def test_c4_no_gradient_outside_range(tmp_path):
    """C4: gradient() is never called at a point outside [min_list,
    max_list]. The minimum (10, 0) lies outside the region [-5, 5]^2 and BFGS
    (no bounds) is driven across the boundary, so out-of-range gradient
    requests do occur. Fails if they reach the solver."""
    fn, grad = _quad_fn([10.0, 0.0])
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=grad)
    alg.main()
    assert solver.grad_calls
    for x, _ in solver.grad_calls:
        assert np.all((x >= -5.0) & (x <= 5.0)), x


@pytest.mark.parametrize("use_gradient", [False, "omitted"])
def test_c4_use_gradient_false_never_calls_gradient(tmp_path, use_gradient):
    """C4: with use_gradient false/omitted and a gradient method, gradient()
    is never called and no History_Gradient.txt is written, even though the
    solver supports gradients. Fails if the gradient is used anyway."""
    fn, grad = _quad_fn(_C)
    minimize = {"method": "BFGS"}
    if use_gradient != "omitted":
        minimize["use_gradient"] = use_gradient
    alg, solver, _ = _make_alg(tmp_path, minimize, fn=fn, grad=grad)
    alg.main()
    assert solver.grad_calls == []
    assert list(tmp_path.glob("**/History_Gradient.txt")) == []


def test_c4_basinhopping_uses_gradient(tmp_path):
    """C4: with basinhopping enabled the local minimizer still uses the
    solver gradient (no finite-difference probes around x0) and converges.
    Fails if the jac is not forwarded to the basinhopping minimizer."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "BFGS", "use_gradient": True,
                   "basinhopping": {"niter": 2}},
        fn=fn, grad=grad)
    x0 = np.array(alg.initial_list, dtype=float, copy=True)
    alg.main()
    assert len(solver.grad_calls) >= 2
    _assert_no_finite_difference_probe(solver, x0)
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-4)


# ---------------------------------------------------------------------------
# C5. min_search callback


@pytest.mark.parametrize("use_gradient", [False, True])
def test_c5_trust_constr_runs_to_convergence(tmp_path, use_gradient):
    """C5: the callback does not stop trust-constr, with or without the
    gradient: more than one iteration runs and the optimum is reached.
    Fails if the callback returns a truthy value."""
    fn, grad = _quad_fn(_C)
    alg, _, _ = _make_alg(
        tmp_path, {"method": "trust-constr", "use_gradient": use_gradient},
        fn=fn, grad=grad)
    alg.main()
    assert alg.itera > 1
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-3)


def test_c5_legacy_callback_does_not_reevaluate(tmp_path):
    """C5: with TNC (legacy callback receiving only x; no gradient), the
    callback must (a) actually fire (iter_history longer than the initial
    row) and (b) not call evaluate() with iset == 1 at an x already evaluated
    with iset == 0. Fails if the callback recomputes f at the iterate."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(tmp_path, {"method": "TNC"}, fn=fn, grad=grad)
    alg.main()
    assert len(alg.iter_history) > 1
    seen = set()
    for x, args in solver.eval_calls:
        key = tuple(x.tolist())
        if args[1] == 1:
            assert key not in seen, f"re-evaluated at {key} with args {args}"
        elif args[1] == 0:
            seen.add(key)


@pytest.mark.parametrize("method,use_gradient", [
    ("Nelder-Mead", False),
    ("TNC", False),
    ("BFGS", True),
    ("trust-constr", True),
    ("TNC", True),
])
def test_c5_iter_history_values_match_objective(tmp_path, method, use_gradient):
    """C5: every row of iter_history (SimplexData.txt) has the objective
    F(x) = f(x / unit_list) at that row's x (unit_list = [2, 0.5]). Fails if
    a row pairs an x with the value of another point or the wrong scaling."""
    minimize = {"method": method}
    if use_gradient:
        minimize["use_gradient"] = True
    alg, _, _ = _make_alg(tmp_path, minimize, fn=_skew_fn, grad=_skew_grad,
                          unit_list=tuple(_UNIT), initial_list=(3.0, 2.0))
    alg.main()
    assert len(alg.iter_history) > 1
    for row in alg.iter_history:
        x = np.array(row[:2], dtype=float)
        np.testing.assert_allclose(row[2], _skew_fn(x / _UNIT), rtol=1e-10, atol=1e-12)


def _same_start_for_all_walkers(x0):
    """A 2-D initial_list with one row per walker, so that every MPI rank
    starts from the same point (a 1-D list fixes walker 0 only and the other
    ranks draw a random start). Keeps the trajectories below identical on
    every rank, both for sensitivity and to stay inside the region."""
    return [list(x0) for _ in range(max(1, odatse.mpi.algsize()))]


def _quart_fn(y):
    return float(np.sum((np.asarray(y) - _C) ** 4))


def _quart_grad(y):
    return 4.0 * (np.asarray(y, dtype=float) - _C) ** 3


@pytest.mark.parametrize("case", ["CG-grad", "LBFGSB-grad-basinhopping",
                                  "Powell-basinhopping"])
def test_c5_scipy_evaluations_are_not_skipped(tmp_path, case):
    """C5 (scipy function evaluations are not skipped): every in-range
    function value scipy requests reaches evaluate() and gets a
    History_FunctionCall.txt row, even for a point evaluated before.

    The contract's observable, #evaluate() calls with iset == 0 (excluding
    the initial (0, 0)) == #History_FunctionCall.txt data rows, is checked,
    but on its own it cannot see a cache that skips *both* the call and the
    row. So the row count is also compared with scipy's own count of
    objective requests, alg.funcalls (OptimizeResult.nfev). The settings
    revisit points (CG on a quartic with unit_list = [2, 0.5]; basinhopping
    whose hops are clipped onto the region corners; Powell line searches) and
    never request an out-of-range point (bounds-capable methods, or a fixed
    start for CG), so all three counts must agree. Fails if _f_calc serves an
    already-evaluated x from a cache."""
    if case == "CG-grad":
        minimize = {"method": "CG", "use_gradient": True}
        # CG has no bounds: a wide region keeps its line search in range
        kw = dict(fn=_quart_fn, grad=_quart_grad, unit_list=(2.0, 0.5),
                  min_list=(-50.0, -50.0), max_list=(50.0, 50.0))
    elif case == "LBFGSB-grad-basinhopping":
        minimize = {"method": "L-BFGS-B", "use_gradient": True,
                    "basinhopping": {"niter": 5, "stepsize": 100.0}}
        fn, grad = _quad_fn(_C)
        kw = dict(fn=fn, grad=grad)
    else:
        minimize = {"method": "Powell", "basinhopping": {"niter": 5}}
        fn, grad = _quad_fn(_C)
        kw = dict(fn=fn, grad=grad)
    alg, solver, _ = _make_alg(tmp_path, minimize,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)),
                               **kw)
    alg.main()
    n_eval = sum(1 for _, a in solver.eval_calls
                 if a[1] == 0 and tuple(a) != (0, 0))
    path = _proc_dir(tmp_path) / "History_FunctionCall.txt"
    _, rows = _read_table(path)
    assert n_eval == len(rows)
    assert alg.funcalls is not None
    assert len(rows) == alg.funcalls, (
        f"scipy requested {alg.funcalls} function values, "
        f"but only {len(rows)} reached the solver")


_FAR = np.array([10.0, 0.0])  # unconstrained minimum, outside [-5, 5]^2


@pytest.mark.parametrize("use_gradient", [False, True])
def test_c5_bounds_method_evaluates_boundary(tmp_path, use_gradient):
    """C5 (boundary handling, bounds-capable method): for L-BFGS-B the
    points on min_list/max_list are inside the region. With the
    unconstrained minimum at (10, 0), the constrained optimum is (5, 0) on
    max_list; the solver must be evaluated there, fopt must be finite (25)
    and xopt must sit on the boundary. Fails if the boundary is treated as
    out of range (inf) for bounds-capable methods."""
    fn, grad = _quad_fn(_FAR)
    minimize = {"method": "L-BFGS-B"}
    if use_gradient:
        minimize["use_gradient"] = True
    alg, solver, _ = _make_alg(tmp_path, minimize, fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    on_boundary = [x for x, _ in solver.eval_calls if x[0] == 5.0]
    assert on_boundary, "no point on max_list was passed to evaluate()"
    assert np.isfinite(alg.fopt)
    np.testing.assert_allclose(alg.xopt, [5.0, 0.0], atol=1e-6)
    assert alg.xopt[0] == 5.0
    np.testing.assert_allclose(alg.fopt, 25.0, rtol=1e-8)


def test_c5_non_bounds_method_skips_boundary(tmp_path):
    """C5 (boundary handling, method without bounds): for Nelder-Mead a
    point on min_list/max_list is out of range, so it never reaches
    evaluate(). initial_scale_list = [3, 0.25] from (2, 2) puts the initial
    simplex vertex (5, 2) exactly on max_list, and the unconstrained minimum
    (10, 0) pulls the simplex toward that edge. Fails if the boundary is
    treated as inside for methods without bounds."""
    fn, grad = _quad_fn(_FAR)
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "Nelder-Mead", "initial_scale_list": [3.0, 0.25],
                   "maxiter": 200},
        fn=fn, grad=grad,
        initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    assert solver.eval_calls
    for x, args in solver.eval_calls:
        assert np.all((x > -5.0) & (x < 5.0)), f"boundary/outside point {x} evaluated ({args})"


# ---------------------------------------------------------------------------
# C6. analytical solver


def _analytical(tmp_path, name, dim):
    info = odatse.Info({
        "base": {"dimension": dim, "output_dir": str(tmp_path / "output")},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical", "function_name": name},
        "runner": {},
    })
    # function.Solver.evaluate() runs inside output/<rank>/
    _proc_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    return odatse.solver.analytical.Solver(info)


_GRAD_CASES = [
    ("quadratics", 1), ("quadratics", 2), ("quadratics", 5),
    ("quartics", 1), ("quartics", 2), ("quartics", 5),
    ("ackley", 1), ("ackley", 2), ("ackley", 5),
    ("rosenbrock", 2), ("rosenbrock", 3), ("rosenbrock", 5),
    ("himmelblau", 2),
    ("linear_regression_test", 3),
]


@pytest.mark.parametrize("name,dim", _GRAD_CASES)
def test_c6_gradient_matches_central_difference(tmp_path, name, dim):
    """C6: functions with gradients report supports_gradient = True and
    gradient(x) has shape (d,) and matches the central difference of
    evaluate() (rel. error ~1e-5; points away from ackley's origin). Fails
    on a wrong formula, a missing component or a wrong shape."""
    s = _analytical(tmp_path, name, dim)
    assert s.supports_gradient is True
    rng = np.random.default_rng(12345 + dim)
    h = 1e-6
    for _ in range(3):
        x = rng.uniform(0.3, 1.7, dim) * rng.choice([-1.0, 1.0], dim)
        g = np.asarray(s.gradient(x.copy()))
        assert g.shape == (dim,)
        num = np.array([
            (s.evaluate(x + h * e) - s.evaluate(x - h * e)) / (2 * h)
            for e in np.eye(dim)
        ])
        scale = max(1.0, float(np.max(np.abs(num))))
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("name", [
    "alpine", "exponential", "griewank", "michalewicz",
    "qing", "rastrigin", "schaffer", "schwefel",
])
def test_c6_functions_without_gradient(tmp_path, name):
    """C6: functions without a gradient report supports_gradient = False
    and gradient() raises NotImplementedError. Fails if a gradient is
    claimed or returned."""
    s = _analytical(tmp_path, name, 2)
    assert s.supports_gradient is False
    with pytest.raises(NotImplementedError):
        s.gradient(np.array([0.3, 0.7]))


def test_c6_rosenbrock_dimension_one_gradient_is_zero(tmp_path):
    """C6: rosenbrock with dimension = 1 (constant zero function) has a
    zero gradient of shape (1,) instead of raising. Fails on an exception
    or a wrong shape/value."""
    s = _analytical(tmp_path, "rosenbrock", 1)
    g = np.asarray(s.gradient(np.array([0.3])))
    assert g.shape == (1,)
    np.testing.assert_array_equal(g, [0.0])
