# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Behavior contract of the optional gradient support, items C1-C14:
gradient failures are handled like objective-function failures (under
ignore_error a RuntimeError, NotImplementedError included, gives NaN;
non-finite gradients are returned as they are), the optimizer's result is
kept as is, x containing NaN gives NaN without a solver call, and a solver
must override gradient() to declare supports_gradient. Covers Runner.gradient_unavailable_reason(),
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
              unit_list=(1.0, 1.0), runner=None, mapping=None, limitation=None,
              **info_kw):
    if fn is None:
        fn, default_grad = _quad_fn([0.0, 0.0])
        if grad is None:
            grad = default_grad
    info = _info(tmp_path, minimize=minimize, unit_list=unit_list,
                 runner=runner, **info_kw)
    solver = _RecordingSolver(info, fn, grad, supports=supports)
    r = odatse.Runner(solver, info, mapping=mapping, limitation=limitation)
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
def test_c2_not_implemented_error_from_gradient_like_runtime_error(tmp_path, monkeypatch, ignore_error):
    """C2 (contract3): NotImplementedError raised by gradient() is handled
    like a RuntimeError: with ignore_error = true an all-NaN array of x's
    shape is returned, otherwise it propagates. (The NotImplementedError of
    an unavailable gradient is a separate case and always propagates.)
    Fails if ignore_error does not cover it, or if it is swallowed without
    ignore_error."""
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch,
                                grad=_raising(NotImplementedError("no grad here")),
                                runner={"ignore_error": ignore_error})
    if ignore_error:
        g = r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
        _assert_all_nan(g, (2,))
    else:
        with pytest.raises(NotImplementedError, match="no grad here"):
            r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    # gradient() was really reached: this is not the unavailable-gradient path
    assert len(solver.grad_calls) == 1


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


def test_c6_set_function_drops_builtin_gradient(tmp_path):
    """C6: replacing the objective with set_function() must not keep the
    gradient of the built-in function. Fails if supports_gradient stays
    True or gradient() returns the old function's gradient."""
    s = _analytical(tmp_path, "quadratics", 2)
    s.set_function(lambda x: float(np.sum((x - 1.0) ** 2)))
    assert not s.supports_gradient
    with pytest.raises(NotImplementedError):
        s.gradient(np.array([1.0, 1.0]))


def test_c11_analytical_subclass_can_assign_supports_gradient(tmp_path):
    """C11: a subclass of the analytical solver can declare gradient support
    by assigning self.supports_gradient in __init__, as for any other
    solver. Fails with AttributeError if supports_gradient is a read-only
    property."""
    info = odatse.Info({
        "base": {"dimension": 2, "output_dir": str(tmp_path / "output")},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical", "function_name": "alpine"},
        "runner": {},
    })

    class _WithGradient(odatse.solver.analytical.Solver):
        def __init__(self, info):
            super().__init__(info)
            self.supports_gradient = True

        def gradient(self, x, args=()):
            return np.zeros_like(x, dtype=float)

    s = _WithGradient(info)
    assert s.supports_gradient
    assert odatse.Runner(s, info).gradient_unavailable_reason() is None


# ===========================================================================
# Round 2: C7-C12

import os
import subprocess
import sys

import scipy.optimize
from scipy.optimize import OptimizeResult

import odatse.exception


def _patch_minimize(monkeypatch, factory):
    """Replace scipy.optimize.minimize, both the name min_search calls and
    the one scipy's basinhopping looks up at call time, by factory(real)."""
    real = scipy.optimize.minimize
    wrapped = factory(real)
    monkeypatch.setattr(min_search, "minimize", wrapped)
    monkeypatch.setattr(min_search.scipy.optimize, "minimize", wrapped)
    return real


def _in_box(x, lo=-5.0, hi=5.0):
    x = np.asarray(x)
    return bool(np.all((x >= lo) & (x <= hi)))


def _grad_rows(tmp_path):
    path = _proc_dir(tmp_path) / "History_Gradient.txt"
    assert path.is_file()
    _, rows = _read_table(path)
    return rows


# ---------------------------------------------------------------------------
# C7. non-finite gradients in the Runner


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("bad", [[np.nan, 1.0], [0.5, np.inf], [-np.inf, 0.0],
                                 [np.nan, np.nan]],
                         ids=["partial-nan", "inf", "-inf", "all-nan"])
def test_c7_nonfinite_gradient_returned_as_is(tmp_path, monkeypatch, bad, ignore_error):
    """C7 (contract3): a gradient() that returns, without raising, an array
    containing NaN or +-inf is returned unchanged by submit_gradient (as a
    float ndarray of x's shape), with or without ignore_error, and nothing
    is raised -- the same as a non-finite objective value. Fails if the
    value raises, or is turned into an all-NaN array (partial cases) or
    otherwise altered."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch,
                           grad=lambda x: np.array(bad),
                           runner={"ignore_error": ignore_error})
    g = r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    assert isinstance(g, np.ndarray) and g.shape == (2,)
    np.testing.assert_array_equal(g, np.array(bad, dtype=float))


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf, 0.0, 1.0]), np.nan],
                         ids=["len1-nan", "len3-inf", "scalar-nan"])
def test_c7_shape_check_precedes_finiteness(tmp_path, monkeypatch, bad, ignore_error):
    """C7: a gradient of the wrong shape that is also non-finite raises
    ValueError (shape is checked before finiteness), with or without
    ignore_error. Fails if it becomes a SolverRuntimeError or a NaN array."""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, grad=lambda x: bad,
                           runner={"ignore_error": ignore_error})
    with pytest.raises(ValueError):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))


# ---------------------------------------------------------------------------
# C8. no false "success" after gradient failures


_BAD = np.array([4.0, 4.0])


def _replacing_factory(fn, bad_x, bad_fun):
    """Run the real optimizer, then overwrite its result with (bad_x,
    bad_fun) -- a controlled stand-in for scipy reporting a poor point."""
    def factory(real):
        def wrapped(*a, **kw):
            res = real(*a, **kw)
            res.x = np.array(bad_x, dtype=float)
            res.fun = bad_fun
            return res
        return wrapped
    return factory


def _best_in_range_evaluation(solver, fn, lo=-5.0, hi=5.0):
    pts = [x for x, _ in solver.eval_calls if _in_box(x, lo, hi)]
    assert pts
    fs = [fn(x) for x in pts]
    i = int(np.argmin(fs))
    return pts[i], fs[i]


@pytest.mark.parametrize("bad_fun", ["worse", "inf", "nan"])
def test_c8_result_kept_after_gradient_failure(tmp_path, monkeypatch, bad_fun):
    """C8 (contract3, result kept): min_search reports the optimizer's
    result as is, even after a gradient failure that actually happened. A
    scripted optimizer requests the jac at Q = (1.5, 1.5), where the
    solver's gradient() is called and raises RuntimeError (ignore_error =
    true), and then returns (4, 4) with a fun worse than f(x0), or inf, or
    NaN. xopt/fopt must be exactly that result. Fails if a best evaluated
    point (or anything else) replaces the optimizer's result."""
    fn, grad = _quad_fn(_C)
    x0 = np.array([2.0, 2.0])
    Q = np.array([1.5, 1.5])
    failures = []

    def fails_at_q(x):
        if np.array_equal(x, Q):
            failures.append(np.array(x, copy=True))
            raise RuntimeError("gradient fails at Q")
        return grad(x)

    fun = {"worse": fn(_BAD), "inf": np.inf, "nan": np.nan}[bad_fun]
    assert not fun <= fn(x0)
    _patch_minimize(monkeypatch, _scripted_minimize(
        eval_points=[Q], jac_points=[Q], result_x=_BAD, result_fun=fun))
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "L-BFGS-B", "use_gradient": True},
        fn=fn, grad=fails_at_q, runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers(x0))
    alg.main()
    # the failure really happened inside the solver's gradient()
    assert failures
    assert any(np.array_equal(x, Q) for x, _ in solver.grad_calls)
    np.testing.assert_array_equal(np.asarray(alg.xopt, dtype=float), _BAD)
    if bad_fun == "nan":
        assert np.isnan(alg.fopt)
    else:
        assert alg.fopt == fun


def test_c8_result_kept_without_gradient_failure(tmp_path, monkeypatch):
    """C8: without any gradient failure, a scipy result worse than an
    evaluated point is reported as is. Fails if min_search substitutes
    another point for the optimizer's result."""
    fn, grad = _quad_fn(_C)
    _patch_minimize(monkeypatch, _replacing_factory(fn, _BAD, fn(_BAD)))
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "L-BFGS-B", "use_gradient": True},
        fn=fn, grad=grad, runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    assert solver.grad_calls
    _, best_f = _best_in_range_evaluation(solver, fn)
    assert best_f < fn(_BAD)
    np.testing.assert_array_equal(np.asarray(alg.xopt, dtype=float), _BAD)
    np.testing.assert_allclose(alg.fopt, fn(_BAD), rtol=1e-12)


_W = np.array([10.0, 1.0])
_CW = np.array([0.5, -0.5])


def _weighted_fn(y):
    return float(np.sum(_W * (np.asarray(y) - _CW) ** 2))


def _weighted_grad(y):
    return 2.0 * _W * (np.asarray(y, dtype=float) - _CW)


def _fails_at(point, grad):
    point = np.asarray(point, dtype=float)

    def g(x):
        if np.array_equal(x, point):
            raise RuntimeError("gradient fails at this point")
        return grad(x)
    return g


def test_c8_real_lbfgsb_stays_at_start_and_reports_failure(tmp_path, capsys):
    """C8 (contract3, real scipy): L-BFGS-B with a gradient that always
    fails at the initial point (2, 2) (RuntimeError, ignore_error = true)
    leaves xopt at x0 (no move to a box corner such as (-5, -5)), and the
    success/message line shows success=False with 'Warning'. fopt may be NaN
    and is not checked. Fails if the optimizer walks away from x0 on the
    failed gradient or the failure is reported as a success."""
    x0 = np.array([2.0, 2.0])
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "L-BFGS-B", "use_gradient": True},
        fn=_weighted_fn, grad=_fails_at(x0, _weighted_grad),
        runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers(x0))
    alg.main()
    out = capsys.readouterr().out
    assert any(np.array_equal(x, x0) for x, _ in solver.grad_calls)
    np.testing.assert_allclose(np.asarray(alg.xopt, dtype=float), x0, rtol=0, atol=1e-12)
    lines = [L for L in out.splitlines() if "success=False" in L]
    assert lines, "no success/message line showing success=False"
    assert all("Warning" in L for L in lines), lines


_SENTINEL = "odatse-test-sentinel-message-31415"


@pytest.mark.parametrize("method,use_gradient", [
    ("Nelder-Mead", False), ("Powell", False), ("COBYLA", False),
    ("TNC", False), ("BFGS", True), ("L-BFGS-B", True), ("SLSQP", True),
    ("trust-constr", True),
])
def test_c8_prints_success_and_message(tmp_path, monkeypatch, capsys, method, use_gradient):
    """C8 (display): after the optimization a stdout line contains the word
    'success' together with scipy's message. The message is replaced by a
    unique sentinel through a minimize wrapper, so scipy's own disp output
    ("Optimization terminated successfully.") cannot satisfy the check.
    Fails if success/message are not reported for this method."""
    def factory(real):
        def wrapped(*a, **kw):
            res = real(*a, **kw)
            res.message = _SENTINEL
            return res
        return wrapped
    _patch_minimize(monkeypatch, factory)
    fn, grad = _quad_fn(_C)
    minimize = {"method": method}
    if use_gradient:
        minimize["use_gradient"] = True
    alg, _, _ = _make_alg(tmp_path, minimize, fn=fn, grad=grad,
                          initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    out = capsys.readouterr().out
    lines = [L for L in out.splitlines() if _SENTINEL in L and "success" in L.lower()]
    assert lines, "no line with 'success' and scipy's message"


# ---------------------------------------------------------------------------
# C9. callback at a failed point


_K = 5


def _controlled_callback_factory(xf, fn, k, call_jac=False, record=None):
    """A stand-in optimizer: optionally request the jac at xf (as scipy
    would), then invoke the callback k times at xf, and return xf as the
    result without running scipy."""
    def factory(real):
        def wrapped(fun, x0, *a, **kw):
            if record is not None:
                record["entry"] = record["solver"].grad_calls.copy()
            if call_jac:
                g = kw["jac"](np.array(xf, dtype=float), *kw.get("args", ()))
                if record is not None:
                    record["jac"] = np.asarray(g, dtype=float)
            cb = kw["callback"]
            for _ in range(k):
                cb(OptimizeResult(x=np.array(xf, dtype=float), fun=fn(np.asarray(xf))))
            if record is not None:
                record["after_cb"] = record["solver"].grad_calls.copy()
            return OptimizeResult(x=np.array(xf, dtype=float), fun=fn(np.asarray(xf)),
                                  success=True, status=0, message="controlled",
                                  nit=k, nfev=1)
        return wrapped
    return factory


_FAIL_MODES = ["raise", "partial-nan", "inf"]


def _failing_value(mode):
    """What a failing gradient() does: raise RuntimeError (ignored ->
    NaN), or return a non-finite array directly (contract3 C9)."""
    if mode == "raise":
        raise RuntimeError("gradient failed")
    if mode == "partial-nan":
        return np.array([np.nan, 1.0])
    return np.array([np.inf, 1.0])


def _fails_at_mode(point, grad, mode):
    point = np.asarray(point, dtype=float)

    def g(x):
        if np.array_equal(x, point):
            return _failing_value(mode)
        return grad(x)
    return g


@pytest.mark.parametrize("mode", _FAIL_MODES)
def test_c9_callback_does_not_retry_failed_gradient(tmp_path, monkeypatch, mode):
    """C9: the gradient at x0 always fails -- by an ignored RuntimeError
    (NaN), or by gradient() itself returning [nan, 1] or [inf, 1]. The
    callback is then invoked K = 5 times at x0 by a controlled optimizer: it
    must fire (iter_history grows by K) without calling gradient() again,
    and every History_Gradient.txt row at x0 has a non-finite gradient (all
    NaN for the raised case). Fails if the callback retries the failed
    gradient on each call."""
    fn, grad = _quad_fn(_C)
    x0 = np.array([2.0, 2.0])
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "BFGS", "use_gradient": True},
        fn=fn, grad=_fails_at_mode(x0, grad, mode), runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers(x0))
    record = {"solver": solver}
    _patch_minimize(monkeypatch, _controlled_callback_factory(x0, fn, _K, record=record))
    alg.main()
    assert record["entry"], "the failing gradient was never requested"
    assert len(alg.iter_history) >= 1 + _K
    assert len(record["after_cb"]) == len(record["entry"])
    assert len(solver.grad_calls) == len(record["entry"])
    rows = [r for r in _grad_rows(tmp_path) if np.array_equal(r[1:3], x0)]
    assert len(rows) >= 1 + _K
    for r in rows:
        if mode == "raise":
            assert np.all(np.isnan(r[4:6])), r
        else:
            assert not np.all(np.isfinite(r[4:6])), r


@pytest.mark.parametrize("mode", _FAIL_MODES)
def test_c9_callback_records_finite_gradient_after_recovery(tmp_path, monkeypatch, mode):
    """C9: the first gradient() call (initial point) fails -- raised and
    ignored, or a returned [nan, 1] / [inf, 1]; a later jac request by the
    optimizer at the same point succeeds. Callbacks at that point afterwards
    must record the finite gradient. Fails if the earlier failure sticks to
    the point."""
    fn, grad = _quad_fn(_C)
    x0 = np.array([2.0, 2.0])
    calls = [0]

    def first_fails(x):
        calls[0] += 1
        if calls[0] == 1:
            return _failing_value(mode)
        return grad(x)

    alg, solver, _ = _make_alg(
        tmp_path, {"method": "BFGS", "use_gradient": True},
        fn=fn, grad=first_fails, runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers(x0))
    record = {"solver": solver}
    _patch_minimize(monkeypatch, _controlled_callback_factory(
        x0, fn, _K, call_jac=True, record=record))
    alg.main()
    np.testing.assert_allclose(record["jac"], grad(x0))
    assert len(alg.iter_history) >= 1 + _K
    rows = [r for r in _grad_rows(tmp_path) if np.array_equal(r[1:3], x0)]
    # the first row is the initial point, recorded while it was failing
    later = rows[1:]
    assert len(later) >= _K
    for r in later:
        np.testing.assert_allclose(r[4:6], grad(x0), rtol=1e-12)


@pytest.mark.parametrize("mode", _FAIL_MODES)
def test_c9_real_tnc_few_gradient_calls(tmp_path, mode):
    """C9 (real scipy): TNC with a gradient that always fails at the
    initial point (raised and ignored, or a returned [nan, 1] / [inf, 1])
    calls gradient() fewer than 10 times (about 100 when each callback
    retried it). Fails if failed points are retried from the callback."""
    x0 = np.array([2.0, 2.0])
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "TNC", "use_gradient": True},
        fn=_weighted_fn, grad=_fails_at_mode(x0, _weighted_grad, mode),
        runner={"ignore_error": True},
        initial_list=_same_start_for_all_walkers(x0))
    alg.main()
    assert solver.grad_calls
    assert len(solver.grad_calls) < 10, len(solver.grad_calls)


# ---------------------------------------------------------------------------
# C10. construction-time validation (additions)


@pytest.mark.parametrize("method", ["dogleg", "trust-ncg", "trust-krylov",
                                    "trust-exact", "Dogleg", "TRUST-NCG"])
def test_c10_hessian_methods_with_use_gradient_raise(tmp_path, method):
    """C10: use_gradient = true with a method that needs a Hessian (dogleg,
    trust-ncg, trust-krylov, trust-exact; case-insensitive) is rejected at
    construction with an error that is both ValueError and InputError, and
    the solver is never called. Fails if construction succeeds (scipy would
    only fail later, after evaluations)."""
    fn, grad = _quad_fn([0.0, 0.0])
    info = _info(tmp_path, minimize={"method": method, "use_gradient": True})
    solver = _RecordingSolver(info, fn, grad)
    r = odatse.Runner(solver, info)
    with pytest.raises(ValueError) as ei:
        min_search.Algorithm(info, r)
    assert isinstance(ei.value, odatse.exception.InputError)
    assert solver.eval_calls == [] and solver.grad_calls == []


def test_c10_newton_cg_with_use_gradient_converges(tmp_path):
    """C10: Newton-CG is accepted with use_gradient = true and converges.
    Fails if it is rejected along with the Hessian methods."""
    fn, grad = _quad_fn(_C)
    alg, solver, _ = _make_alg(tmp_path, {"method": "Newton-CG", "use_gradient": True},
                               fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    assert solver.grad_calls
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-4)


_C3_CASES = {
    "non-bool": ({"method": "BFGS", "use_gradient": "true"}, {}),
    "gradient-free": ({"method": "Powell", "use_gradient": True}, {}),
    "omitted-method": ({"use_gradient": True}, {}),
    "unsupported": ({"method": "BFGS", "use_gradient": True}, {"supports": False}),
    "affine": ({"method": "BFGS", "use_gradient": True},
               {"mapping": "affine"}),
    "reason": ({"method": "BFGS", "use_gradient": True}, {"reason": True}),
}


@pytest.mark.parametrize("case", list(_C3_CASES))
def test_c10_c3_errors_are_input_errors(tmp_path, monkeypatch, case):
    """C10: the construction errors of C3 are both ValueError and
    odatse.exception.InputError. Fails if a plain ValueError is raised."""
    minimize, kw = _C3_CASES[case]
    fn, grad = _quad_fn([0.0, 0.0])
    info = _info(tmp_path, minimize=minimize)
    solver = _RecordingSolver(info, fn, grad, supports=kw.get("supports", True))
    mapping = None
    if kw.get("mapping") == "affine":
        mapping = odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
    r = odatse.Runner(solver, info, mapping=mapping)
    if kw.get("reason"):
        monkeypatch.setattr(r, "gradient_unavailable_reason",
                            lambda: "solver group has 2 processes", raising=False)
    with pytest.raises(ValueError) as ei:
        min_search.Algorithm(info, r)
    assert isinstance(ei.value, odatse.exception.InputError)


_CLI_INPUT = """\
[base]
dimension = 2
output_dir = "output"

[algorithm]
name = "minsearch"
seed = 12345

[algorithm.param]
min_list = [-5.0, -5.0]
max_list = [5.0, 5.0]
initial_list = [2.0, 2.0]

[algorithm.minimize]
{minimize}

[solver]
name = "analytical"
function_name = "quadratics"
"""


@pytest.mark.parametrize("minimize", [
    'method = "BFGS"\nuse_gradient = "true"',
    'method = "Nelder-Mead"\nuse_gradient = true',
], ids=["non-bool", "gradient-free"])
def test_c10_cli_reports_use_gradient_error(tmp_path, minimize):
    """C10: through the CLI (odatse_main.py in a subprocess, ODATSE_NOMPI=1)
    a use_gradient input error exits non-zero with an 'ERROR:' line on
    stderr and no Traceback. Fails if the error escapes as a raw exception
    (Traceback) or the run succeeds."""
    (tmp_path / "input.toml").write_text(_CLI_INPUT.format(minimize=minimize))
    main_py = os.path.join(os.path.dirname(__file__), "..", "..", "src", "odatse_main.py")
    env = dict(os.environ)
    env["ODATSE_NOMPI"] = "1"
    proc = subprocess.run([sys.executable, os.path.abspath(main_py), "input.toml"],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode != 0
    assert any("ERROR:" in line for line in proc.stderr.splitlines()), proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


# ---------------------------------------------------------------------------
# C11. declaring supports_gradient


class _InstanceAttrSolver(odatse.solver.SolverBase):
    """Declares gradient support by assigning in __init__."""

    def __init__(self, info):
        super().__init__(info)
        self.supports_gradient = True
        self.grad_calls = []

    def evaluate(self, x, args=()):
        return _quad_fn(_C)[0](x)

    def gradient(self, x, args=()):
        self.grad_calls.append(np.array(x, copy=True))
        return _quad_fn(_C)[1](x)


class _ClassAttrSolver(odatse.solver.SolverBase):
    """Declares gradient support as a class attribute."""

    supports_gradient = True

    def __init__(self, info):
        super().__init__(info)
        self.grad_calls = []

    def evaluate(self, x, args=()):
        return _quad_fn(_C)[0](x)

    def gradient(self, x, args=()):
        self.grad_calls.append(np.array(x, copy=True))
        return _quad_fn(_C)[1](x)


class _PropertySolver(_ClassAttrSolver):
    """Declares gradient support as a property."""

    supports_gradient = property(lambda self: True)


class _PlainSolver(odatse.solver.SolverBase):
    """Declares nothing about gradients."""

    def __init__(self, info):
        super().__init__(info)

    def evaluate(self, x, args=()):
        return _quad_fn(_C)[0](x)


_DECLARING = {"instance": _InstanceAttrSolver, "class": _ClassAttrSolver,
              "property": _PropertySolver}


@pytest.mark.parametrize("how", list(_DECLARING))
def test_c11_declared_support_is_honored_by_runner(tmp_path, monkeypatch, how):
    """C11: a SolverBase subclass may declare supports_gradient = True by
    assignment in __init__, as a class attribute, or as a property; the
    construction succeeds and Runner.gradient_unavailable_reason() is None.
    Fails if the assignment raises (read-only property in the base) or the
    declaration is not seen."""
    info = _info(tmp_path)
    solver = _DECLARING[how](info)
    assert solver.supports_gradient
    r = odatse.Runner(solver, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert r.gradient_unavailable_reason() is None


@pytest.mark.parametrize("how", list(_DECLARING))
def test_c11_declared_support_is_used_by_min_search(tmp_path, how):
    """C11: with each declaration style, min_search with use_gradient =
    true uses the solver's gradient and converges. Fails if the declaration
    is rejected or ignored."""
    info = _info(tmp_path, minimize={"method": "BFGS", "use_gradient": True},
                 initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    solver = _DECLARING[how](info)
    r = odatse.Runner(solver, info)
    alg = min_search.Algorithm(info, r)
    alg.main()
    assert solver.grad_calls
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-4)


def test_c11_undeclared_support_is_false(tmp_path):
    """C11: a SolverBase subclass that declares nothing has a false
    supports_gradient. Fails if the default claims support."""
    solver = _PlainSolver(_info(tmp_path))
    assert not solver.supports_gradient


# ---------------------------------------------------------------------------
# C12. paths not reached otherwise


def _legacy_callback_factory(shift=None):
    """Hand the callback only the raw x array scipy gives (no copy), as
    scipy < 1.11 does; optionally shifted by ``shift`` (a new array) to
    present a point that was never evaluated."""
    fired = {"n": 0}

    def factory(real):
        def wrapped(*a, **kw):
            cb = kw.get("callback")
            if cb is not None:
                def legacy(xk):
                    fired["n"] += 1
                    if shift is None:
                        return cb(xk)
                    return cb(xk + shift)
                kw["callback"] = legacy
            return real(*a, **kw)
        return wrapped
    return factory, fired


# minimum of _skew_fn in solver units, (15/11, -8/11); algorithm units x = y * unit
_SKEW_OPT = np.array([15.0 / 11.0, -8.0 / 11.0]) * _UNIT


_OLD_SCIPY_CASES = [
    ("BFGS", False, False), ("BFGS", True, False),
    ("L-BFGS-B", False, False), ("L-BFGS-B", True, False),
    ("TNC", False, False), ("TNC", True, False),
    ("BFGS", True, True), ("L-BFGS-B", False, True), ("L-BFGS-B", True, True),
]


@pytest.mark.parametrize("method,use_gradient,bh", _OLD_SCIPY_CASES,
                         ids=[f"{m}-{'grad' if g else 'nograd'}{'-bh' if b else ''}"
                              for m, g, b in _OLD_SCIPY_CASES])
def test_c12_simulated_old_scipy_callback(tmp_path, monkeypatch, method, use_gradient, bh):
    """C12 (scipy < 1.11 simulation): scipy.__version__ is set to 1.10.0
    and every minimize call (also those made by basinhopping) hands the
    callback only scipy's raw x array, uncopied. With unit_list = [2, 0.5]
    the run converges, the callback fires, every iter_history row carries
    F(x) = f(x / unit_list) at its x, and with use_gradient the
    History_Gradient.txt gradients are the algorithm-unit gradients. Fails on
    callback incompatibility (e.g. scaling x twice or aliasing scipy's
    array)."""
    monkeypatch.setattr(min_search.scipy, "__version__", "1.10.0")
    factory, fired = _legacy_callback_factory()
    _patch_minimize(monkeypatch, factory)
    minimize = {"method": method}
    if use_gradient:
        minimize["use_gradient"] = True
    if bh:
        minimize["basinhopping"] = {"niter": 2}
    alg, _, _ = _make_alg(tmp_path, minimize, fn=_skew_fn, grad=_skew_grad,
                          unit_list=tuple(_UNIT),
                          initial_list=_same_start_for_all_walkers((3.0, 2.0)))
    alg.main()
    assert fired["n"] > 0
    assert len(alg.iter_history) > 1
    for row in alg.iter_history:
        x = np.array(row[:2], dtype=float)
        np.testing.assert_allclose(row[2], _skew_fn(x / _UNIT), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(alg.xopt, _SKEW_OPT, atol=1e-3)
    if use_gradient:
        rows = _grad_rows(tmp_path)
        assert len(rows) > 1
        for r in rows:
            x = r[1:3]
            np.testing.assert_allclose(r[4:6], _skew_grad(x / _UNIT) / _UNIT,
                                       rtol=1e-9, atol=1e-12)
    else:
        assert list(tmp_path.glob("**/History_Gradient.txt")) == []


def test_c12_legacy_callback_at_unevaluated_point(tmp_path, monkeypatch):
    """C12: when a legacy callback (TNC) receives a point _f_calc has not
    evaluated (scipy's x shifted by 1e-6 through a minimize wrapper), that
    point goes to evaluate() with iset = 1 and its value is recorded in
    iter_history. Fails if the callback records a stale/cached value or
    skips the evaluation."""
    factory, fired = _legacy_callback_factory(shift=np.array([1e-6, -1e-6]))
    _patch_minimize(monkeypatch, factory)
    alg, solver, _ = _make_alg(tmp_path, {"method": "TNC"}, fn=_skew_fn,
                               grad=_skew_grad, unit_list=tuple(_UNIT),
                               initial_list=_same_start_for_all_walkers((3.0, 2.0)))
    alg.main()
    assert fired["n"] > 0
    rows = alg.iter_history[1:]
    assert rows
    iset1 = [x for x, a in solver.eval_calls if a[1] == 1]
    for row in rows:
        x = np.array(row[:2], dtype=float)
        assert any(np.array_equal(y, x / _UNIT) for y in iset1), x
        np.testing.assert_allclose(row[2], _skew_fn(x / _UNIT), rtol=1e-12)


def test_c12_limitation_judged_in_algorithm_units(tmp_path, monkeypatch):
    """C12: with unit_list = [2, 2] and the limitation x1 + x2 < 3, the
    point P = (2, 2) violates it in algorithm units, while P / unit_list =
    (1, 1) satisfies it in solver units (so the Runner's own check would let
    it through). A jac request at P, injected through a minimize wrapper,
    must not reach the solver's gradient(). Fails if min_search leaves the
    limitation check to the Runner."""
    unit = np.array([2.0, 2.0])
    P = np.array([2.0, 2.0])
    lim = odatse.util.limitation.Inequality(
        np.array([[-1.0, -1.0]]), np.array([3.0]), True)
    assert not lim.judge(P) and lim.judge(P / unit)
    requested = {}

    def factory(real):
        def wrapped(*a, **kw):
            if "jac" in kw and callable(kw["jac"]) and not requested:
                requested["g"] = np.asarray(
                    kw["jac"](P.copy(), *kw.get("args", ())), dtype=float)
            return real(*a, **kw)
        return wrapped
    _patch_minimize(monkeypatch, factory)
    fn, grad = _quad_fn([0.25, 0.25])
    alg, solver, _ = _make_alg(
        tmp_path, {"method": "BFGS", "use_gradient": True},
        fn=fn, grad=grad, unit_list=tuple(unit), limitation=lim,
        initial_list=_same_start_for_all_walkers((0.5, 0.5)))
    alg.main()
    assert "g" in requested
    assert not any(np.array_equal(x, P / unit) for x, _ in solver.grad_calls)
    assert solver.grad_calls


# ---------------------------------------------------------------------------
# Scripted optimizer. A stand-in for scipy.optimize.minimize evaluates chosen
# points through the objective, optionally requests the jac at chosen points,
# and returns a chosen result, so stdout and the run are fully determined.


def _scripted_minimize(eval_points=(), jac_points=(), result_x=None,
                       result_fun=None, success=True, message="scripted"):
    def factory(real):
        def wrapped(fun, x0, *a, **kw):
            args = kw.get("args", ())
            for p in eval_points:
                fun(np.array(p, dtype=float), *args)
            for p in jac_points:
                kw["jac"](np.array(p, dtype=float), *args)
            return OptimizeResult(x=np.array(result_x, dtype=float), fun=result_fun,
                                  success=success, status=0 if success else 1,
                                  message=message, nit=1,
                                  nfev=len(eval_points))
        return wrapped
    return factory


def _run_scripted(base, monkeypatch, capsys, factory, grad, minimize=None,
                  unit_list=(1.0, 1.0), x0=(2.0, 2.0), fn=None):
    """One min_search run with the scripted optimizer, in its own output
    directory under ``base``; returns (alg, solver, stdout)."""
    if fn is None:
        fn = _quad_fn(_C)[0]
    if minimize is None:
        minimize = {"method": "L-BFGS-B", "use_gradient": True}
    base.mkdir(parents=True, exist_ok=True)
    _patch_minimize(monkeypatch, factory)
    capsys.readouterr()
    alg, solver, _ = _make_alg(base, minimize, fn=fn, grad=grad,
                               unit_list=unit_list, runner={"ignore_error": True},
                               initial_list=_same_start_for_all_walkers(x0))
    alg.main()
    return alg, solver, capsys.readouterr().out


@pytest.mark.parametrize("use_gradient", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_c8_success_line_warning_prefix(tmp_path, monkeypatch, capsys, success, use_gradient):
    """C8 (display): the line reporting scipy's success and message contains
    'Warning' when success is False and does not when it is True (scripted
    result with a unique message). Fails if a failed optimization is
    reported like a success, or a success is flagged."""
    fn, grad = _quad_fn(_C)
    minimize = {"method": "L-BFGS-B"}
    if use_gradient:
        minimize["use_gradient"] = True
    _, _, out = _run_scripted(
        tmp_path / "run", monkeypatch, capsys,
        _scripted_minimize(result_x=_C, result_fun=fn(_C), success=success,
                           message=_SENTINEL),
        grad, minimize=minimize)
    lines = [L for L in out.splitlines() if _SENTINEL in L and "success" in L.lower()]
    assert lines, "no line with 'success' and scipy's message"
    if success:
        assert not any("Warning" in L for L in lines), lines
    else:
        assert all("Warning" in L for L in lines), lines


# ---------------------------------------------------------------------------
# C13. x containing NaN (contract3)


def _probe_factory(probe, before, after, record):
    """Scripted optimizer: evaluate ``before``, then the probe point, then
    ``after`` through the objective, recording each return value."""
    def factory(real):
        def wrapped(fun, x0, *a, **kw):
            args = kw.get("args", ())
            record["before"] = fun(np.array(before, dtype=float), *args)
            record["probe"] = fun(np.array(probe, dtype=float), *args)
            record["after"] = fun(np.array(after, dtype=float), *args)
            return OptimizeResult(x=np.array(after, dtype=float), fun=record["after"],
                                  success=True, status=0, message="scripted",
                                  nit=1, nfev=3)
        return wrapped
    return factory


_A = np.array([1.0, 1.0])
_B = np.array([0.5, -0.5])


@pytest.mark.parametrize("use_gradient", [False, True])
@pytest.mark.parametrize("probe", [[np.nan, 1.0], [1.0, np.nan], [np.nan, np.nan]],
                         ids=["nan-first", "nan-second", "all-nan"])
def test_c13_nan_x_gives_nan_without_evaluation(tmp_path, monkeypatch, probe, use_gradient):
    """C13: when the optimizer requests the objective at an x containing NaN
    (partly or fully), between two finite in-range requests A and B:
    (a) the returned value is NaN, not the out-of-range inf;
    (b) evaluate() is never called with a NaN x;
    (c) the step numbers of A and B at evaluate() are consecutive (the NaN
        request consumes none);
    (d) History_FunctionCall.txt has exactly one row per finite request.
    Fails if a NaN x is treated as out of range (inf) or reaches the
    solver."""
    fn, grad = _quad_fn(_C)
    record = {}
    _patch_minimize(monkeypatch, _probe_factory(probe, _A, _B, record))
    minimize = {"method": "BFGS"}
    if use_gradient:
        minimize["use_gradient"] = True
    alg, solver, _ = _make_alg(tmp_path, minimize, fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    # (a)
    assert np.isnan(record["probe"]), record["probe"]
    assert record["before"] == fn(_A) and record["after"] == fn(_B)
    # (b)
    assert not any(np.any(np.isnan(x)) for x, _ in solver.eval_calls)
    # (c)
    steps = {}
    for x, a in solver.eval_calls:
        if a[1] == 0 and tuple(a) != (0, 0):
            steps.setdefault(tuple(x.tolist()), a[0])
    sa, sb = steps[tuple(_A.tolist())], steps[tuple(_B.tolist())]
    assert sb == sa + 1, (sa, sb)
    # (d)
    _, rows = _read_table(_proc_dir(tmp_path) / "History_FunctionCall.txt")
    assert len(rows) == 2


@pytest.mark.parametrize("probe", [[np.inf, 1.0], [-np.inf, 1.0], [6.0, 0.0]],
                         ids=["inf", "-inf", "finite-out-of-range"])
def test_c13_inf_or_out_of_range_x_gives_inf(tmp_path, monkeypatch, probe):
    """C13: an x with +-inf components only (no NaN), or a finite x outside
    the range, is out of range as before: the value is inf, evaluate() is not
    called there, and no step number or History_FunctionCall.txt row is
    used. Fails if such points are treated like NaN or evaluated."""
    fn, grad = _quad_fn(_C)
    record = {}
    _patch_minimize(monkeypatch, _probe_factory(probe, _A, _B, record))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS"}, fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    assert record["probe"] == np.inf
    probe = np.array(probe)
    assert not any(np.array_equal(x, probe) for x, _ in solver.eval_calls)
    assert not any(not np.all(np.isfinite(x)) for x, _ in solver.eval_calls)
    steps = [a[0] for _, a in solver.eval_calls if a[1] == 0 and tuple(a) != (0, 0)]
    assert steps == [steps[0], steps[0] + 1]
    _, rows = _read_table(_proc_dir(tmp_path) / "History_FunctionCall.txt")
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# C14. supports_gradient declared, gradient() not overridden (contract3)


class _NoOverrideBaseSolver(odatse.solver.SolverBase):
    """Declares gradient support but keeps SolverBase.gradient."""

    supports_gradient = True

    def __init__(self, info):
        super().__init__(info)
        self.eval_calls = []

    def evaluate(self, x, args=()):
        self.eval_calls.append((np.array(x, copy=True), args))
        return _quad_fn(_C)[0](x)


class _NoOverrideFunctionSolver(odatse.solver.function.Solver):
    """function.Solver subclass declaring support without a gradient()."""

    def __init__(self, info):
        super().__init__(info)
        self.set_function(_quad_fn(_C)[0])
        self.eval_calls = []

    @property
    def supports_gradient(self):
        return True

    def evaluate(self, x, args=()):
        self.eval_calls.append((np.array(x, copy=True), args))
        return super().evaluate(x, args)


class _PartialGradientSolver(_NoOverrideBaseSolver):
    """Overrides gradient(), but raises NotImplementedError for x[0] > 0."""

    def __init__(self, info):
        super().__init__(info)
        self.grad_calls = []

    def gradient(self, x, args=()):
        self.grad_calls.append(np.array(x, copy=True))
        if x[0] > 0:
            raise NotImplementedError("no gradient for x[0] > 0")
        return _quad_fn(_C)[1](x)


_NO_OVERRIDE = {"base": _NoOverrideBaseSolver, "function": _NoOverrideFunctionSolver}


@pytest.mark.parametrize("kind", list(_NO_OVERRIDE))
def test_c14_reason_when_gradient_not_overridden(tmp_path, monkeypatch, kind):
    """C14: a solver with a true supports_gradient that does not override
    gradient() (SolverBase.gradient is inherited) gets a reason string from
    Runner.gradient_unavailable_reason(). Fails if the declaration alone is
    taken as gradient support."""
    info = _info(tmp_path)
    solver = _NO_OVERRIDE[kind](info)
    assert solver.supports_gradient
    r = odatse.Runner(solver, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason


def test_c14_no_reason_for_override_raising_at_some_points(tmp_path, monkeypatch):
    """C14: a solver that overrides gradient() but raises
    NotImplementedError at some points is available (reason None): only the
    presence of the override is judged. Fails if such a solver is rejected."""
    info = _info(tmp_path)
    r = odatse.Runner(_PartialGradientSolver(info), info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert r.gradient_unavailable_reason() is None


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("kind", list(_NO_OVERRIDE))
def test_c14_submit_gradient_raises_when_not_overridden(tmp_path, monkeypatch, kind, ignore_error):
    """C14: submit_gradient raises NotImplementedError for a solver that does
    not override gradient(), with ignore_error true as well as false (it must
    not silently become an all-NaN gradient), and evaluate() is not called.
    Fails if ignore_error turns the missing implementation into NaN."""
    info = _info(tmp_path, runner={"ignore_error": ignore_error})
    solver = _NO_OVERRIDE[kind](info)
    r = odatse.Runner(solver, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    assert solver.eval_calls == []


@pytest.mark.parametrize("ignore_error", [False, True])
def test_c14_override_raising_not_implemented_is_like_runtime_error(tmp_path, monkeypatch, ignore_error):
    """C14 / C2: for a solver that overrides gradient(), a
    NotImplementedError at some point is handled like a RuntimeError: NaN
    with ignore_error = true, raised otherwise; elsewhere the gradient is
    returned. Fails if the override is treated as a missing gradient()."""
    info = _info(tmp_path, runner={"ignore_error": ignore_error})
    solver = _PartialGradientSolver(info)
    r = odatse.Runner(solver, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    good = np.array([-0.5, 0.25])
    np.testing.assert_allclose(r.submit_gradient(good, (1, 0)), _quad_fn(_C)[1](good))
    bad = np.array([0.5, 0.25])
    if ignore_error:
        _assert_all_nan(r.submit_gradient(bad, (2, 0)), (2,))
    else:
        with pytest.raises(NotImplementedError):
            r.submit_gradient(bad, (2, 0))
    assert len(solver.grad_calls) == 2


@pytest.mark.parametrize("kind", list(_NO_OVERRIDE))
def test_c14_min_search_rejects_not_overridden_gradient(tmp_path, kind):
    """C14: min_search with use_gradient = true and a solver that does not
    override gradient() raises InputValueError at construction; it is both
    a ValueError and an InputError, and the solver is never called. Fails if
    the run starts and proceeds with all-NaN gradients."""
    info = _info(tmp_path, minimize={"method": "BFGS", "use_gradient": True},
                 initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    solver = _NO_OVERRIDE[kind](info)
    r = odatse.Runner(solver, info)
    with pytest.raises(odatse.exception.InputValueError) as ei:
        min_search.Algorithm(info, r)
    assert isinstance(ei.value, ValueError)
    assert isinstance(ei.value, odatse.exception.InputError)
    assert solver.eval_calls == []
