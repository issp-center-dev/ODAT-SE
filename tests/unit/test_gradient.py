# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Behavior contract of the optional gradient support, items C1-C18:
gradient failures are handled like objective-function failures (under
ignore_error a RuntimeError, NotImplementedError included, gives NaN;
non-finite gradients are returned as they are), the optimizer's result is
kept as is, x containing NaN gives NaN without a solver call, and a solver
must override gradient() or, for a function solver, set one with
set_gradient() to declare supports_gradient (C14, C17). Covers Runner.gradient_unavailable_reason(),
Runner.submit_gradient(), use_gradient of min_search, its callback, the
memory of function values and gradients cut back at each callback (C15),
the gradients of the analytical solver, which exist for every built-in
function (C6, C16), set_gradient() of the function solver (C17), and
gradients with a parallel solver (--nsolve > 1), which no longer make the
gradient unavailable: the controller sends MSG_GRADIENT to the workers,
which serve it in Runner.serve_gradient() (C18, C1-C3 revised)."""
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


# A solver group of 2 or more processes is no longer a reason (contract7):
# see test_c1r7_reason_none_for_parallel_solver and
# test_c1r7_other_reasons_remain_in_parallel in Round 7.


# ---------------------------------------------------------------------------
# C2. Runner.submit_gradient()


@pytest.mark.parametrize("situation", ["unsupported", "affine"])
def test_c2_submit_gradient_raises_when_unavailable(tmp_path, monkeypatch, situation):
    """C2: in every C1 situation that yields a reason, submit_gradient raises
    NotImplementedError without reaching the solver's gradient(). The
    unsupported solver has a working gradient(), so the error must come from
    the Runner's own check. Fails if the gradient is computed anyway. (The
    same in a solver group of 2: test_c18_unavailable_sends_nothing.)"""
    kw = {"solsize": 1}
    if situation == "unsupported":
        kw["supports"] = False
    if situation == "affine":
        kw["mapping"] = odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, **kw)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(np.array([1.0, 1.0]), (1, 0))
    assert solver.grad_calls == []


@pytest.mark.parametrize("situation", ["unsupported", "affine"])
def test_c2_unavailable_raises_even_when_limitation_violated(tmp_path, monkeypatch, situation):
    """C2: the NotImplementedError of every C1 situation takes precedence
    over the limitation: with an x that violates the limitation,
    submit_gradient still raises instead of returning the NaN array of the
    limitation path. Fails if the limitation is judged before gradient
    availability (an unusable gradient would then go unnoticed as NaN).
    (The same in a solver group of 2:
    test_c2r7_unavailable_precedes_limitation_in_parallel.)"""
    kw = {"solsize": 1}
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
    runner.gradient_unavailable_reason() returns a string, whatever the
    reason (here a patched one, with an otherwise valid setup). Fails if the
    Runner is not asked for a reason."""
    def patch(r):
        monkeypatch.setattr(r, "gradient_unavailable_reason",
                            lambda: "some reason reported by the Runner", raising=False)

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


# every built-in function of the analytical solver, with a valid dimension
_ALL_BUILTIN = [
    ("quadratics", 2), ("quartics", 2), ("ackley", 2), ("alpine", 2),
    ("exponential", 2), ("griewank", 2), ("himmelblau", 2),
    ("michalewicz", 2), ("qing", 2), ("rastrigin", 2), ("rosenbrock", 2),
    ("schaffer", 2), ("schwefel", 2), ("linear_regression_test", 3),
]


@pytest.mark.parametrize("name,dim", _ALL_BUILTIN)
def test_c6_every_builtin_function_supports_gradient(tmp_path, name, dim):
    """C6 (revised in contract5): every built-in function of the analytical
    solver reports a true supports_gradient, and gradient() returns
    instead of raising NotImplementedError. Fails if any built-in function
    is left without a gradient."""
    s = _analytical(tmp_path, name, dim)
    assert s.supports_gradient
    g = np.asarray(s.gradient(np.full(dim, 0.3)))
    assert g.shape == (dim,)


@pytest.mark.parametrize("name,dim", _ALL_BUILTIN)
def test_c6_set_function_drops_gradient_of_every_builtin(tmp_path, name, dim):
    """C6 (unchanged in contract5): for every built-in function, replacing
    the objective with set_function() makes supports_gradient false and
    gradient() raise NotImplementedError. Fails if a newly added gradient
    is attached so that set_function() does not drop it."""
    s = _analytical(tmp_path, name, dim)
    s.set_function(lambda x: float(np.sum((x - 1.0) ** 2)))
    assert not s.supports_gradient
    with pytest.raises(NotImplementedError):
        s.gradient(np.full(dim, 0.3))


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
    solver. The subclass replaces the objective with set_function() (which
    makes supports_gradient false, C6) and overrides gradient() (C14), so
    only the assignment can make it true; a sibling subclass without the
    assignment checks that. Fails with AttributeError if supports_gradient
    is a read-only property, or if the assigned value is not honored by
    the Runner."""
    info = odatse.Info({
        "base": {"dimension": 2, "output_dir": str(tmp_path / "output")},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical", "function_name": "quadratics"},
        "runner": {},
    })
    fn, grad = _quad_fn([1.0, -0.5])

    class _Replaced(odatse.solver.analytical.Solver):
        def __init__(self, info):
            super().__init__(info)
            self.set_function(fn)

        def gradient(self, x, args=()):
            return grad(x)

    class _WithGradient(_Replaced):
        def __init__(self, info):
            super().__init__(info)
            self.supports_gradient = True

    # without the assignment the subclass has no gradient support
    s = _Replaced(info)
    assert not s.supports_gradient
    assert odatse.Runner(s, info).gradient_unavailable_reason() is not None

    s = _WithGradient(info)
    assert s.supports_gradient
    assert odatse.Runner(s, info).gradient_unavailable_reason() is None
    np.testing.assert_array_equal(s.gradient(np.array([2.0, 0.5])), [2.0, 2.0])


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


# ===========================================================================
# Round 4: C15
#
# The memory of function values, finite gradients and failed gradients is
# cut back at every callback to the callback's iterate; points computed
# after that callback are kept until the next one.


_P_A = np.array([1.0, 1.0])
_P_B = np.array([0.5, -0.5])
_P_C = np.array([-1.0, 0.25])


def _c15_script_factory(steps, calls, form, fn, log):
    """Scripted optimizer for C15: run ``steps`` -- (op, point, measured)
    with op "fun", "jac" or "cb" -- in order instead of scipy. For each
    measured step, the solver calls made by that step alone (the part of
    the list ``calls()`` appended during the step) go to ``log``. ``form``
    is how the callback receives the iterate: "result" (an OptimizeResult,
    new style) or "raw" (the x array itself)."""
    def factory(real):
        def wrapped(fun, x0, *a, **kw):
            args = kw.get("args", ())
            cb = kw["callback"]
            for op, p, measured in steps:
                x = np.array(p, dtype=float)
                before = len(calls())
                if op == "fun":
                    fun(x, *args)
                elif op == "jac":
                    kw["jac"](x, *args)
                elif form == "result":
                    cb(OptimizeResult(x=x, fun=fn(x)))
                else:
                    cb(x)
                if measured:
                    log.append(calls()[before:])
            x0 = np.array(x0, dtype=float)
            return OptimizeResult(x=x0, fun=fn(x0), success=True, status=0,
                                  message="scripted", nit=1, nfev=len(steps))
        return wrapped
    return factory


# "raw" also under a simulated scipy < 1.11, where the legacy callback of
# min_search is installed instead of the new-style one fed a bare array
_C15_FORMS = ["result", "raw", "raw-old-scipy"]


def _c15_setup_form(monkeypatch, form):
    if form == "raw-old-scipy":
        monkeypatch.setattr(min_search.scipy, "__version__", "1.10.0")
        return "raw"
    return form


@pytest.mark.parametrize("form", ["raw", "raw-old-scipy"])
def test_c15_function_values_forgotten_at_callback(tmp_path, monkeypatch, form):
    """C15 (a), function values (use_gradient = false, callback given x
    itself): after fun(a), fun(b), the steps callback(a), callback(a),
    callback(b), then fun(c) unmeasured, callback(c) make 0, 0, 1, 0
    evaluate() calls, and the one call is at b with iset = 1. Nothing is
    forgotten before the first callback, the last iterate is kept, b
    (evaluated before the last callback and not its iterate) is forgotten,
    c (evaluated after it) is kept. Fails if callbacks never forget (0 at
    the third step), forget the iterate or forget points computed after
    the callback."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)
    log = []
    steps = [("fun", _P_A, False), ("fun", _P_B, False),
             ("cb", _P_A, True), ("cb", _P_A, True), ("cb", _P_B, True),
             ("fun", _P_C, False), ("cb", _P_C, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].eval_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS"}, fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    assert [len(c) for c in log] == [0, 0, 1, 0], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (x, args), = log[2]
    np.testing.assert_array_equal(x, _P_B)
    assert args[1] == 1, args
    assert len(alg.iter_history) == 1 + 4
    np.testing.assert_allclose(alg.iter_history[3][2], fn(_P_B), rtol=1e-12)


@pytest.mark.parametrize("form", _C15_FORMS)
def test_c15_gradients_forgotten_at_callback(tmp_path, monkeypatch, form):
    """C15 (a), finite gradients (use_gradient = true): after jac(a),
    jac(b), the steps callback(a), callback(a), jac(a), jac(b), callback(b)
    make 0, 0, 0, 1, 0 gradient() calls, and the one call is at b. The
    callback reuses a's gradient (nothing forgotten before the first
    callback) and keeps it as the last iterate; b's gradient was forgotten
    at the callback, so even scipy's jac request recomputes it; that new
    gradient, computed after the last callback, is reused by callback(b).
    Run with a new-style OptimizeResult and with a bare x (also under a
    simulated scipy < 1.11). Fails if callbacks never forget (0 at the
    fourth step), forget the iterate, or forget gradients computed after
    the callback."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)
    log = []
    steps = [("jac", _P_A, False), ("jac", _P_B, False),
             ("cb", _P_A, True), ("cb", _P_A, True), ("jac", _P_A, True),
             ("jac", _P_B, True), ("cb", _P_B, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].grad_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    assert [len(c) for c in log] == [0, 0, 0, 1, 0], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (x, _), = log[3]
    np.testing.assert_array_equal(x, _P_B)
    # every callback recorded the finite gradient of its iterate
    rows = _grad_rows(tmp_path)
    assert len(rows) == 1 + 3
    for r, p in zip(rows[1:], [_P_A, _P_A, _P_B]):
        np.testing.assert_array_equal(r[1:3], p)
        np.testing.assert_allclose(r[4:6], grad(p), rtol=1e-12)


@pytest.mark.parametrize("mode", _FAIL_MODES)
@pytest.mark.parametrize("form", _C15_FORMS)
def test_c15_failed_gradients_forgotten_at_callback(tmp_path, monkeypatch, form, mode):
    """C15 (a), failed gradients (ignore_error = true; gradient() fails at
    a and b by an ignored RuntimeError, or returns [nan, 1] / [inf, 1]):
    after jac(a), jac(b), callback(a) makes no gradient() call (C9: the
    failure at a is still remembered), and callback(b) makes exactly one, at
    b (the record of b's failure was forgotten at callback(a)). Fails if
    the failure records are never forgotten (0 at callback(b)) or are
    forgotten before the first callback (1 at callback(a))."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)

    def g(x):
        if np.array_equal(x, _P_A) or np.array_equal(x, _P_B):
            return _failing_value(mode)
        return grad(x)

    log = []
    steps = [("jac", _P_A, False), ("jac", _P_B, False),
             ("cb", _P_A, True), ("cb", _P_B, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].grad_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=g, runner={"ignore_error": True},
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    # the scripted jac requests did reach the failing gradient()
    assert sum(1 for x, _ in solver.grad_calls if np.array_equal(x, _P_A)) >= 1
    assert [len(c) for c in log] == [0, 1], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (x, _), = log[1]
    np.testing.assert_array_equal(x, _P_B)
    rows = _grad_rows(tmp_path)
    assert len(rows) == 1 + 2
    for r in rows[1:]:
        assert not np.all(np.isfinite(r[4:6])), r


class _RecordingAnalytical(odatse.solver.analytical.Solver):
    """Analytical solver recording every evaluate()/gradient() call."""

    def __init__(self, info):
        super().__init__(info)
        self.eval_calls = []
        self.grad_calls = []

    def evaluate(self, x, args=()):
        self.eval_calls.append((np.array(x, dtype=float, copy=True), args))
        return super().evaluate(x, args)

    def gradient(self, x, args=()):
        self.grad_calls.append((np.array(x, dtype=float, copy=True), args))
        return super().gradient(x, args)


# (initial point, unit_list) in solver units; the region is [-w, w] per
# component in solver units
_C15_ROSEN = {
    "2d": ([-1.2, 1.0], [1.0, 1.0]),
    "3d": ([1.5, -0.5, 0.3], [1.0, 1.0, 1.0]),
    "3d-unit": ([1.5, -0.5, 0.3], [2.0, 0.5, 1.5]),
}


def _c15_rosenbrock_alg(tmp_path, minimize, case, w=2.0):
    y0, unit = (np.array(v, dtype=float) for v in _C15_ROSEN[case])
    dim = len(y0)
    info = odatse.Info({
        "base": {"dimension": dim, "output_dir": str(tmp_path / "output")},
        "algorithm": {
            "name": "minsearch",
            "seed": 1,
            "label_list": [f"x{i}" for i in range(dim)],
            "param": {
                "min_list": list(-w * unit),
                "max_list": list(w * unit),
                "initial_list": _same_start_for_all_walkers(y0 * unit),
                "unit_list": list(unit),
            },
            "minimize": dict(minimize),
        },
        "solver": {"name": "analytical", "function_name": "rosenbrock"},
        "runner": {},
    })
    solver = _RecordingAnalytical(info)
    r = odatse.Runner(solver, info)
    return min_search.Algorithm(info, r), solver, unit


def _c15_observer_factory(seen):
    """Wrap the jac and the callback scipy receives, recording the points
    (algorithm units) in ``seen["jac"]`` and ``seen["cb"]``. The callback
    wrapper keeps the parameter name intermediate_result, so that scipy
    picks the same callback form as for min_search's own callback."""
    def factory(real):
        def wrapped(*a, **kw):
            jac = kw.get("jac")
            if callable(jac):
                def observed_jac(x, *args):
                    seen["jac"].append(np.array(x, dtype=float, copy=True).flatten())
                    return jac(x, *args)
                kw["jac"] = observed_jac
            cb = kw["callback"]

            def observed_cb(intermediate_result):
                if isinstance(intermediate_result, OptimizeResult):
                    x = intermediate_result.x
                else:
                    x = intermediate_result
                seen["cb"].append(np.array(x, dtype=float, copy=True).flatten())
                return cb(intermediate_result)
            kw["callback"] = observed_cb
            return real(*a, **kw)
        return wrapped
    return factory


def _c15_key(x):
    return np.asarray(x, dtype=float).flatten().tobytes()


@pytest.mark.parametrize("case", list(_C15_ROSEN))
def test_c15_real_tnc_without_gradient_no_callback_evaluation(tmp_path, monkeypatch, case):
    """C15 (b): a single TNC run without gradients (rosenbrock, the callback
    gets x only) makes no evaluate() call with iset = 1: every iterate the
    callback sees was evaluated in the same iteration, after the previous
    callback. The callback fires several times at two or more distinct
    iterates. Fails if forgetting at the callback drops the value of the
    iterate it is called with, or of points evaluated since the previous
    callback."""
    seen = {"jac": [], "cb": []}
    _patch_minimize(monkeypatch, _c15_observer_factory(seen))
    alg, solver, _ = _c15_rosenbrock_alg(tmp_path, {"method": "TNC"}, case)
    alg.main()
    assert len(seen["cb"]) > 1
    assert len({_c15_key(x) for x in seen["cb"]}) >= 2
    iset1 = [(x.tolist(), a) for x, a in solver.eval_calls if a[1] == 1]
    assert iset1 == []
    assert solver.grad_calls == []


@pytest.mark.parametrize("case", list(_C15_ROSEN))
@pytest.mark.parametrize("method", ["TNC", "L-BFGS-B", "BFGS", "trust-constr"])
def test_c15_real_run_gradient_calls_equal_distinct_points(tmp_path, monkeypatch, method, case):
    """C15 (b): in a single real scipy run with use_gradient = true
    (rosenbrock, all gradients finite, one case with unit_list != 1), the
    number of gradient() calls equals the number of distinct points
    (algorithm-unit x compared as bytes) in the union of the initial point,
    the points of scipy's jac requests and the callback iterates -- i.e. no
    point has its gradient computed twice. The callback fires several times
    at two or more distinct iterates, and every point stays inside the
    region (so each distinct point needs exactly one call). Fails if the
    memory is cut back so that the iterate, or a point computed after the
    previous callback, is recomputed."""
    seen = {"jac": [], "cb": []}
    _patch_minimize(monkeypatch, _c15_observer_factory(seen))
    # BFGS has no bounds: its line search reaches |y| ~ 15 on the 3-D start,
    # where an out-of-range jac request would not call gradient() at all
    w = 20.0 if method == "BFGS" else 2.0
    alg, solver, unit = _c15_rosenbrock_alg(
        tmp_path, {"method": method, "use_gradient": True}, case, w=w)
    alg.main()
    assert seen["jac"], "scipy never requested the jac"
    assert len(seen["cb"]) > 1
    assert len({_c15_key(x) for x in seen["cb"]}) >= 2
    points = [np.asarray(alg.initial_list, dtype=float).flatten()] + seen["jac"] + seen["cb"]
    for x in points:
        assert np.all((-w * unit <= x) & (x <= w * unit)), x
    distinct = {_c15_key(x) for x in points}
    assert len(solver.grad_calls) == len(distinct), (
        f"{len(solver.grad_calls)} gradient() calls for {len(distinct)} distinct points")


# An iterate whose entry is in no cache: out of range, so its value is inf
# without a solver call and is not remembered.
_P_Z_OUT = np.array([6.0, 0.0])


@pytest.mark.parametrize("form", ["raw", "raw-old-scipy"])
def test_c15_function_values_forgotten_at_uncached_iterate(tmp_path, monkeypatch, form):
    """C15 (a), function values, iterate without a cached value
    (use_gradient = false, callback given x itself): after fun(a), the steps
    callback(a), callback(z), callback(a) make 0, 0, 1 evaluate() calls, with
    z out of range (inf, never evaluated nor remembered) and the one call at
    a with iset = 1. Since z has no entry, nothing is left after
    callback(z), so a must be evaluated again. Fails if a callback whose
    iterate is not cached keeps the whole memory instead of emptying it."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)
    log = []
    steps = [("fun", _P_A, False),
             ("cb", _P_A, True), ("cb", _P_Z_OUT, True), ("cb", _P_A, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].eval_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS"}, fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    assert [len(c) for c in log] == [0, 0, 1], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (x, args), = log[2]
    np.testing.assert_array_equal(x, _P_A)
    assert args[1] == 1, args
    assert len(alg.iter_history) == 1 + 3
    assert alg.iter_history[2][2] == np.inf


@pytest.mark.parametrize("form", _C15_FORMS)
def test_c15_gradients_forgotten_at_uncached_iterate(tmp_path, monkeypatch, form):
    """C15 (a), finite gradients, iterate without a cached gradient
    (use_gradient = true): after jac(a), jac(b) and callback(c) at a point c
    whose gradient was never requested (its gradient may be computed inside
    that callback; not measured), the steps jac(a), callback(b) make 1, 1
    gradient() calls, at a and at b. The gradients of a and b were computed
    before callback(c) and c is not either of them, so both were forgotten.
    Fails if a callback whose iterate is not in the gradient memory keeps
    the whole memory instead of emptying it."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)
    log = []
    steps = [("jac", _P_A, False), ("jac", _P_B, False), ("cb", _P_C, False),
             ("jac", _P_A, True), ("cb", _P_B, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].grad_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=grad,
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    assert [len(c) for c in log] == [1, 1], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (xa, _), = log[0]
    (xb, _), = log[1]
    np.testing.assert_array_equal(xa, _P_A)
    np.testing.assert_array_equal(xb, _P_B)


@pytest.mark.parametrize("mode", _FAIL_MODES)
@pytest.mark.parametrize("form", _C15_FORMS)
def test_c15_failed_gradients_forgotten_at_uncached_iterate(tmp_path, monkeypatch, form, mode):
    """C15 (a), failed gradients, iterate without a failure record
    (ignore_error = true; gradient() fails at a and b by an ignored
    RuntimeError, or returns [nan, 1] / [inf, 1]; finite elsewhere): after
    jac(a), jac(b) and callback(c) at a point c with a finite gradient that
    was never requested (computed inside that callback; not measured),
    callback(b) makes exactly one gradient() call, at b. c has no failure
    record, so after callback(c) none is left and b's failure is retried.
    Fails if a callback whose iterate is not among the failed points keeps
    all failure records instead of emptying them."""
    form = _c15_setup_form(monkeypatch, form)
    fn, grad = _quad_fn(_C)

    def g(x):
        if np.array_equal(x, _P_A) or np.array_equal(x, _P_B):
            return _failing_value(mode)
        return grad(x)

    log = []
    steps = [("jac", _P_A, False), ("jac", _P_B, False), ("cb", _P_C, False),
             ("cb", _P_B, True)]
    holder = {}
    _patch_minimize(monkeypatch, _c15_script_factory(
        steps, lambda: holder["solver"].grad_calls, form, fn, log))
    alg, solver, _ = _make_alg(tmp_path, {"method": "BFGS", "use_gradient": True},
                               fn=fn, grad=g, runner={"ignore_error": True},
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    holder["solver"] = solver
    alg.main()
    # the scripted jac requests did reach the failing gradient()
    assert sum(1 for x, _ in solver.grad_calls if np.array_equal(x, _P_B)) >= 1
    assert [len(c) for c in log] == [1], [
        [(x.tolist(), a) for x, a in c] for c in log]
    (x, _), = log[0]
    np.testing.assert_array_equal(x, _P_B)
    rows = _grad_rows(tmp_path)
    assert len(rows) == 1 + 2
    np.testing.assert_allclose(rows[1][4:6], grad(_P_C), rtol=1e-12)
    assert not np.all(np.isfinite(rows[2][4:6])), rows[2]


# ===========================================================================
# Round 5: C16
#
# Gradients of the eight analytical functions added in contract5. They are
# checked against the central difference of evaluate() and by their values
# at special points, never against a copy of the formula.

import math
import warnings


_C16_NAMES = ["alpine", "exponential", "griewank", "michalewicz",
              "qing", "rastrigin", "schaffer", "schwefel"]
_C16_DIMS = [1, 2, 5]

# step h of the central difference and rtol of the comparison per function;
# atol is always 1e-6 * max(1, max|difference|). schwefel takes a larger h
# because its points reach |x| ~ 500 and |f| ~ 2000.
_C16_FD = {
    "alpine": (1e-6, 1e-5),
    "exponential": (1e-6, 1e-5),
    "griewank": (1e-6, 1e-5),
    "michalewicz": (1e-6, 1e-5),
    "qing": (1e-6, 1e-5),
    "rastrigin": (1e-6, 1e-5),
    "schaffer": (1e-6, 1e-5),
    "schwefel": (1e-4, 1e-5),
}


def _c16_central_difference(s, x, h, components=None):
    if components is None:
        components = range(len(x))
    e = np.eye(len(x))
    return np.array([(s.evaluate(x + h * e[i]) - s.evaluate(x - h * e[i])) / (2 * h)
                     for i in components])


def _c16_assert_matches(s, name, x, g, components=None):
    h, rtol = _C16_FD[name]
    if components is None:
        components = list(range(len(x)))
    num = _c16_central_difference(s, x, h, components)
    scale = max(1.0, float(np.max(np.abs(num))))
    np.testing.assert_allclose(np.asarray(g)[components], num, rtol=rtol,
                               atol=1e-6 * scale, err_msg=f"{name} at {x}")
    return num


def _c16_michalewicz_point(rng, dim, jitter):
    """A point where every sin(i x_i^2 / pi)^20 is near 1 (a ridge of the
    function), so that its gradient is far from zero; random ridges for
    rng, the first ridge otherwise."""
    i = np.arange(dim) + 1.0
    if rng is None:
        k = np.zeros(dim)
        dx = np.full(dim, jitter)
    else:
        k = np.array([rng.integers(0, int(j)) for j in i], dtype=float)
        dx = rng.uniform(-jitter, jitter, dim)
    return np.sqrt(np.pi * (np.pi / 2 + k * np.pi) / i) + dx


def _c16_random_point(name, rng, dim):
    """A random point in a meaningful range of ``name``, away from the
    points where the function is not differentiable."""
    if name == "alpine":
        # away from the roots of x (sin x + 0.1): x = 0 and sin x = -0.1
        while True:
            x = rng.uniform(-8.0, 8.0, dim)
            if np.all(np.abs(x) > 0.2) and np.all(np.abs(np.sin(x) + 0.1) > 0.05):
                return x
    if name == "exponential":
        return rng.uniform(-1.5, 1.5, dim)
    if name == "griewank":
        return rng.uniform(-10.0, 10.0, dim)
    if name == "michalewicz":
        return _c16_michalewicz_point(rng, dim, 0.03)
    if name == "qing":
        return rng.uniform(-3.0, 3.0, dim)
    if name == "rastrigin":
        return rng.uniform(-5.12, 5.12, dim)
    if name == "schaffer":
        return rng.uniform(-5.0, 5.0, dim)
    if name == "schwefel":
        # away from x = 0, where sqrt(|x|) is not differentiable
        return rng.uniform(1.0, 500.0, dim) * rng.choice([-1.0, 1.0], dim)
    raise AssertionError(name)


def _c16_fixed_point(name, dim):
    """A fixed point where the largest component of the gradient is of
    order 1e-1 or larger, so that a zero vector cannot pass."""
    if name == "michalewicz":
        return _c16_michalewicz_point(None, dim, 0.02)
    if name == "schwefel":
        return np.array([100.0, -200.0, 300.0, -400.0, 450.0])[:dim]
    if name == "exponential":
        return np.full(dim, 0.5)
    if name == "rastrigin":
        return np.full(dim, 0.25)
    # alpine, griewank, qing, schaffer: sin x > 0 there, so no alpine root
    return np.linspace(1.0, 2.5, dim)


def _c16_cases():
    # schaffer with d = 1 is the zero function (see the special points)
    return [(n, d) for n in _C16_NAMES for d in _C16_DIMS
            if not (n == "schaffer" and d == 1)]


# ---------------------------------------------------------------------------
# C16.1 return type, shape, input untouched


@pytest.mark.parametrize("dim", _C16_DIMS)
@pytest.mark.parametrize("name", _C16_NAMES)
def test_c16_gradient_type_shape_and_input_untouched(tmp_path, name, dim):
    """C16.1: gradient(x) returns a float np.ndarray of shape (d,) and does
    not write to x (x is made read-only and compared afterwards). Fails if
    gradient() raises NotImplementedError, returns a list/scalar/wrong
    shape, or updates x in place."""
    s = _analytical(tmp_path, name, dim)
    rng = np.random.default_rng(1600 + dim)
    for x in [_c16_fixed_point(name, dim), _c16_random_point(name, rng, dim)]:
        x = np.array(x, dtype=float)
        x0 = x.copy()
        x.flags.writeable = False
        g = s.gradient(x)
        assert isinstance(g, np.ndarray), type(g)
        assert g.shape == (dim,)
        assert np.issubdtype(g.dtype, np.floating), g.dtype
        np.testing.assert_array_equal(x, x0)


# ---------------------------------------------------------------------------
# C16.2 agreement with the central difference


@pytest.mark.parametrize("name,dim", _c16_cases())
def test_c16_gradient_matches_central_difference(tmp_path, name, dim):
    """C16.2: at a fixed point with a large gradient and at five random
    points (fixed seed, away from non-differentiable points) gradient(x)
    matches the central difference of evaluate() (h and rtol from _C16_FD,
    atol = 1e-6 * max(1, max|difference|)). At least one point has a
    difference whose largest component is >= 1e-2, so a zero vector
    cannot pass. Fails on a wrong sign or factor (e.g. michalewicz without
    the factor i, schaffer without one of the two terms sharing x_i,
    schwefel without sign(x), alpine without the sign of the inner term),
    a missing component or a zero gradient."""
    s = _analytical(tmp_path, name, dim)
    rng = np.random.default_rng(16000 + 10 * dim + _C16_NAMES.index(name))
    points = [_c16_fixed_point(name, dim)]
    points += [_c16_random_point(name, rng, dim) for _ in range(5)]
    largest = 0.0
    for x in points:
        g = s.gradient(x.copy())
        num = _c16_assert_matches(s, name, x, g)
        largest = max(largest, float(np.max(np.abs(num))))
    assert largest >= 1e-2, f"all test points of {name} have a tiny gradient"


# ---------------------------------------------------------------------------
# C16.3 non-differentiable and special points


@pytest.mark.parametrize("dim", _C16_DIMS)
def test_c16_alpine_zero_components(tmp_path, dim):
    """C16.3 (alpine): a component x_i = 0 has gradient exactly 0, while
    the other components match the central difference; at x = 0 the
    gradient is the zero vector. Fails if the sign of the inner term is
    dropped (that gives 0.1 at x_i = 0) or if a subgradient other than 0
    is chosen."""
    s = _analytical(tmp_path, "alpine", dim)
    np.testing.assert_array_equal(s.gradient(np.zeros(dim)), np.zeros(dim))
    x = _c16_fixed_point("alpine", dim)
    zero = [0] if dim < 5 else [0, 3]
    x[zero] = 0.0
    g = s.gradient(x.copy())
    np.testing.assert_array_equal(g[zero], 0.0)
    others = [i for i in range(dim) if i not in zero]
    if others:
        _c16_assert_matches(s, "alpine", x, g, others)


# roots of x (sin x + 0.1) other than 0, i.e. sin x = -0.1. (The sign flip
# of a root is not a root: sin(-x) = 0.1 there.)
_C16_ALPINE_ROOTS = [
    math.pi + math.asin(0.1), 2 * math.pi - math.asin(0.1),
    -math.asin(0.1), -math.pi + math.asin(0.1),
]


@pytest.mark.parametrize("root", _C16_ALPINE_ROOTS)
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_c16_alpine_both_sides_of_nonzero_root(tmp_path, dim, root):
    """C16.3 (alpine): 1e-3 on either side of a nonzero root of
    x sin x + 0.1 x (sin x = -0.1) the component at the root matches the
    central difference and its sign is opposite on the two sides (the
    value on the root is not checked). Fails if the sign of the inner
    term x sin x + 0.1 x is replaced by the sign of x or dropped."""
    s = _analytical(tmp_path, "alpine", dim)
    j = dim // 2
    base = _c16_fixed_point("alpine", dim)
    sides = []
    for off in (-1e-3, 1e-3):
        x = base.copy()
        x[j] = root + off
        g = s.gradient(x.copy())
        _c16_assert_matches(s, "alpine", x, g)
        sides.append(g[j])
    assert sides[0] * sides[1] < 0, sides


@pytest.mark.parametrize("dim", _C16_DIMS)
def test_c16_schwefel_zero_components(tmp_path, dim):
    """C16.3 (schwefel): a component x_i = 0 has gradient exactly 0, with
    no NaN and no RuntimeWarning (divide by zero / invalid value); the
    other components match the central difference. Fails if the
    derivative is written as x / sqrt(|x|) (0/0 at x_i = 0) without a
    guard, or if a guard leaves a warning or a nonzero value."""
    s = _analytical(tmp_path, "schwefel", dim)
    x = _c16_fixed_point("schwefel", dim)
    zero = [0] if dim < 5 else [0, 3]
    x[zero] = 0.0
    for p in [np.zeros(dim), x]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            g = s.gradient(p.copy())
        assert np.all(np.isfinite(g)), g
        np.testing.assert_array_equal(g[p == 0.0], 0.0)
    others = [i for i in range(dim) if i not in zero]
    if others:
        _c16_assert_matches(s, "schwefel", x, g, others)


@pytest.mark.parametrize("dim,n_zero_cos", [(1, 1), (2, 1), (5, 1), (2, 2), (5, 2)])
def test_c16_griewank_cos_near_zero(tmp_path, dim, n_zero_cos):
    """C16.3 (griewank): where cos(x_j / sqrt(j)) is near 0 for one
    component (x_1 = pi/2) or two (also x_4 = pi, i.e. x_4 / 2 = pi/2;
    for d < 4 x_2 = sqrt(2) pi/2), the gradient is finite and matches the
    central difference. Fails if the product over the other components
    is lost or blows up there."""
    s = _analytical(tmp_path, "griewank", dim)
    x = _c16_fixed_point("griewank", dim)
    x[0] = math.pi / 2
    if n_zero_cos == 2:
        j = 3 if dim >= 4 else 1
        x[j] = math.sqrt(j + 1) * math.pi / 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        g = s.gradient(x.copy())
    assert np.all(np.isfinite(g)), g
    _c16_assert_matches(s, "griewank", x, g)


def test_c16_schaffer_dimension_one_gradient_is_zero(tmp_path):
    """C16.3 (schaffer): with d = 1 (no pair of neighbors, the zero
    function) the gradient is the zero vector of shape (1,). Fails on an
    exception (e.g. from indexing an empty pair list) or a nonzero
    value."""
    s = _analytical(tmp_path, "schaffer", 1)
    for v in [0.0, 1.3, -2.7, 500.0]:
        g = np.asarray(s.gradient(np.array([v])))
        assert g.shape == (1,)
        np.testing.assert_array_equal(g, [0.0])


# ---------------------------------------------------------------------------
# C16.4 finite, without warnings, for |x_i| <= 500


def _c16_finiteness_points(dim, rng):
    pts = [np.zeros(dim), np.full(dim, 500.0), np.full(dim, -500.0),
           np.where(np.arange(dim) % 2 == 0, 500.0, -500.0),
           np.full(dim, 1e-300), np.full(dim, -5e-324)]
    # special points of C16.3
    p = np.linspace(1.0, 2.5, dim)
    p[0] = 0.0
    pts.append(p)
    p = np.linspace(1.0, 2.5, dim)
    p[0] = math.pi / 2
    if dim >= 4:
        p[3] = math.pi
    pts.append(p)
    pts.append(np.full(dim, math.pi + math.asin(0.1)))
    pts.append(np.full(dim, -math.asin(0.1)))
    pts += [rng.uniform(-500.0, 500.0, dim) for _ in range(5)]
    return pts


@pytest.mark.parametrize("dim", _C16_DIMS)
@pytest.mark.parametrize("name", _C16_NAMES)
def test_c16_gradient_finite_without_warnings(tmp_path, name, dim):
    """C16.4: for x with |x_i| <= 500 -- the origin, |x_i| = 500 (all
    signs), tiny |x_i| (1e-300, a subnormal), the special points of C16.3
    and random points -- the gradient is finite and no RuntimeWarning
    (divide by zero, overflow, invalid value) is emitted. Fails on NaN/inf
    from 0/0 or 0*inf (e.g. schwefel at 0, griewank dividing the product)
    or on an unguarded operation that warns."""
    s = _analytical(tmp_path, name, dim)
    rng = np.random.default_rng(1640 + dim)
    for x in _c16_finiteness_points(dim, rng):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            g = s.gradient(x.copy())
        assert np.all(np.isfinite(g)), f"{name} at {x}: {g}"


# ---------------------------------------------------------------------------
# C16.5 min_search with use_gradient = true


def _c16_alg(tmp_path, name, x0, method="L-BFGS-B", w=5.0):
    dim = len(x0)
    info = odatse.Info({
        "base": {"dimension": dim, "output_dir": str(tmp_path / "output")},
        "algorithm": {
            "name": "minsearch",
            "seed": 1,
            "label_list": [f"x{i}" for i in range(dim)],
            "param": {
                "min_list": [-w] * dim,
                "max_list": [w] * dim,
                "initial_list": _same_start_for_all_walkers(x0),
                "unit_list": [1.0] * dim,
            },
            "minimize": {"method": method, "use_gradient": True},
        },
        "solver": {"name": "analytical", "function_name": name},
        "runner": {},
    })
    solver = _RecordingAnalytical(info)
    r = odatse.Runner(solver, info)
    return min_search.Algorithm(info, r), solver


@pytest.mark.parametrize("name", _C16_NAMES)
def test_c16_min_search_accepts_use_gradient(tmp_path, name):
    """C16.5: min_search with use_gradient = true and L-BFGS-B is built for
    every added function without an InputValueError. Fails if a function
    still lacks a gradient (supports_gradient false)."""
    _c16_alg(tmp_path, name, [1.0, 1.2])


# (start, minimum point, minimum value); each start is 0.2-0.3 away from
# the minimum in every component, inside the same valley
_C16_MIN_CASES = {
    "qing-2d": ("qing", [1.25, math.sqrt(2) - 0.3], [1.0, math.sqrt(2)], 0.0),
    "qing-3d-neg": ("qing", [-0.75, math.sqrt(2) + 0.2, -math.sqrt(3) + 0.25],
                    [-1.0, math.sqrt(2), -math.sqrt(3)], 0.0),
    "rastrigin-2d": ("rastrigin", [0.2, -0.25], [0.0, 0.0], 0.0),
    "rastrigin-3d": ("rastrigin", [-0.25, 0.2, 0.3], [0.0, 0.0, 0.0], 0.0),
    "exponential-2d": ("exponential", [0.3, -0.2], [0.0, 0.0], -1.0),
    "exponential-3d": ("exponential", [-0.25, 0.3, 0.2], [0.0, 0.0, 0.0], -1.0),
    "griewank-2d": ("griewank", [0.3, -0.25], [0.0, 0.0], 0.0),
    "griewank-3d": ("griewank", [-0.2, 0.3, -0.25], [0.0, 0.0, 0.0], 0.0),
}


@pytest.mark.parametrize("case", list(_C16_MIN_CASES))
def test_c16_lbfgsb_with_gradient_converges(tmp_path, case):
    """C16.5: L-BFGS-B with use_gradient = true, started 0.2-0.3 from the
    minimum in every component (same start on every rank), calls the
    solver's gradient(), does not fall back to finite differences (no
    evaluate() probe within 1e-6 of the start along one axis), moves away
    from the start and ends within 1e-4 of the minimum with fopt within
    1e-8 of the minimum value. Fails if the function is rejected at
    construction, if gradient() is never called, or if a wrong gradient
    stops the optimizer early or sends it elsewhere."""
    name, x0, xmin, fmin = _C16_MIN_CASES[case]
    x0 = np.array(x0, dtype=float)
    alg, solver = _c16_alg(tmp_path, name, x0)
    alg.main()
    assert len(solver.grad_calls) >= 2
    _assert_no_finite_difference_probe(solver, x0)
    xopt = np.asarray(alg.xopt, dtype=float)
    assert np.max(np.abs(xopt - x0)) > 0.1, xopt
    np.testing.assert_allclose(xopt, xmin, rtol=0, atol=1e-4)
    assert abs(alg.fopt - fmin) < 1e-8, alg.fopt


# ===========================================================================
# Round 6: C17 / C6 / C14 revisions
#
# function.Solver.set_gradient(g) (C17); the analytical solver regains a
# gradient by set_gradient() after set_function() (C6 revised); a function
# solver subclass that declares supports_gradient without overriding
# gradient() and without a gradient set is rejected (C14 revised).

import pickle
import types


class _C17Recorder:
    """Callable that records each call: the original argument objects (for
    identity checks) and a copy of the first argument (for values)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []
        self.xs = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args:
            self.xs.append(np.array(args[0], dtype=float, copy=True))
        return self.fn(*args, **kwargs)


def _c17_f(x):
    return float(np.sum(np.asarray(x) ** 2))


def _c17_f2(x):
    return float(np.sum((np.asarray(x) - 3.0) ** 2))


# gradients unrelated to the objectives above, so a value can only come
# from the g that was set
def _c17_g1(x):
    return np.array([10.0 * x[0] + 1.0, 7.0 - x[1]])


def _c17_g2(x):
    return np.array([3.0 * x[1], -2.0 * x[0]])


# module-level, so that a solver holding them can be pickled (C17-9)
def _c17_pickle_f(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def _c17_pickle_g(x):
    return np.array([5.0 * x[0], x[0] - x[1]])


_C17_X = np.array([0.3, -1.2])


def _c17_plain(tmp_path, f=None, g=None, **info_kw):
    """A plain odatse.solver.function.Solver (no subclass), with set_function
    and set_gradient called when f / g are given."""
    info = _info(tmp_path, **info_kw)
    s = odatse.solver.function.Solver(info)
    if f is not None:
        s.set_function(f)
    if g is not None:
        s.set_gradient(g)
    return info, s


class _C17Assigned(odatse.solver.function.Solver):
    """function.Solver subclass assigning supports_gradient in __init__,
    after calling set_function / set_gradient when f / g are given."""

    def __init__(self, info, value, f=None, g=None):
        super().__init__(info)
        if f is not None:
            self.set_function(f)
        if g is not None:
            self.set_gradient(g)
        self.supports_gradient = value


class _C17ReadOnlyTrue(odatse.solver.function.Solver):
    """function.Solver subclass with a read-only supports_gradient property."""

    @property
    def supports_gradient(self):
        return True


class _C17ReadOnlyFalse(odatse.solver.function.Solver):
    """function.Solver subclass with a read-only supports_gradient property."""

    @property
    def supports_gradient(self):
        return False


_C17_READ_ONLY = {True: _C17ReadOnlyTrue, False: _C17ReadOnlyFalse}


# ---------------------------------------------------------------------------
# C17-1 no gradient before set_gradient


@pytest.mark.parametrize("state", ["fresh", "function-only"])
def test_c17_no_gradient_before_set_gradient(tmp_path, state):
    """C17-1: a function solver just constructed, or with only set_function()
    called, has a false supports_gradient and gradient(x) raises
    NotImplementedError without calling the objective. Fails if the solver
    claims gradient support before set_gradient(), or if gradient() falls
    back to something computed from the objective."""
    f = _C17Recorder(_c17_f)
    _, s = _c17_plain(tmp_path, f=None if state == "fresh" else f)
    assert not s.supports_gradient
    with pytest.raises(NotImplementedError):
        s.gradient(_C17_X.copy())
    with pytest.raises(NotImplementedError):
        s.gradient(_C17_X.copy(), (1, 0))
    assert f.calls == []


# ---------------------------------------------------------------------------
# C17-2 set_gradient after set_function


def test_c17_set_gradient_enables_gradient(tmp_path):
    """C17-2: after set_function(f) and set_gradient(g), supports_gradient is
    true and gradient(x, args) returns the value of g(x); g receives the very
    x object given to gradient() as its only argument (no args, with or
    without args given to gradient()). Fails if set_gradient does not turn
    supports_gradient on, if the value does not come from g, if x is copied
    or transformed, or if args are forwarded to g."""
    g = _C17Recorder(_c17_g1)
    _, s = _c17_plain(tmp_path, f=_c17_f, g=g)
    assert s.supports_gradient

    x = _C17_X.copy()
    np.testing.assert_array_equal(np.asarray(s.gradient(x, (3, 0))), _c17_g1(_C17_X))
    y = np.array([-2.0, 0.5])
    np.testing.assert_array_equal(np.asarray(s.gradient(y)), _c17_g1(y))

    assert len(g.calls) == 2
    for (args, kwargs), arr in zip(g.calls, (x, y)):
        assert kwargs == {}
        assert len(args) == 1
        assert args[0] is arr


# ---------------------------------------------------------------------------
# C17-3 set_function after set_gradient drops the gradient


@pytest.mark.parametrize("same_function", [True, False],
                         ids=["same-function", "other-function"])
def test_c17_set_function_after_set_gradient_drops_it(tmp_path, same_function):
    """C17-3: set_function() after set_gradient() drops the gradient (back to
    the C17-1 state: supports_gradient false, gradient() raises
    NotImplementedError, neither g nor the objective is called), also when
    the same objective is set again; a later set_gradient(g2) makes the
    gradient available again with g2. Fails if set_function keeps the old
    gradient (which would be wrong for a new objective), or only drops it
    when the objective changes, or if the gradient cannot be set again."""
    f = _C17Recorder(_c17_f)
    f_new = f if same_function else _C17Recorder(_c17_f2)
    g1 = _C17Recorder(_c17_g1)
    g2 = _C17Recorder(_c17_g2)
    _, s = _c17_plain(tmp_path, f=f, g=g1)
    assert s.supports_gradient

    s.set_function(f_new)
    assert not s.supports_gradient
    with pytest.raises(NotImplementedError):
        s.gradient(_C17_X.copy())
    assert g1.calls == []
    assert f.calls == [] and f_new.calls == []

    s.set_gradient(g2)
    assert s.supports_gradient
    np.testing.assert_array_equal(np.asarray(s.gradient(_C17_X.copy())), _c17_g2(_C17_X))
    assert len(g2.calls) == 1
    assert g1.calls == []


# ---------------------------------------------------------------------------
# C17-4 the last set_gradient wins


def test_c17_second_set_gradient_wins(tmp_path):
    """C17-4: calling set_gradient twice makes gradient() use the later
    function only. Fails if the first g is kept or both are called."""
    g1 = _C17Recorder(_c17_g1)
    g2 = _C17Recorder(_c17_g2)
    _, s = _c17_plain(tmp_path, f=_c17_f, g=g1)
    s.set_gradient(g2)
    assert s.supports_gradient
    np.testing.assert_array_equal(np.asarray(s.gradient(_C17_X.copy())), _c17_g2(_C17_X))
    assert g1.calls == []
    assert len(g2.calls) == 1


# ---------------------------------------------------------------------------
# C17-5 exceptions and the current directory


class _C17Error(Exception):
    pass


@pytest.mark.parametrize("make_error", [
    lambda: RuntimeError("g failed"),
    lambda: NotImplementedError("g has no value here"),
    lambda: ValueError("bad x"),
    lambda: _C17Error("custom"),
], ids=["RuntimeError", "NotImplementedError", "ValueError", "custom"])
def test_c17_gradient_exception_propagates_unchanged(tmp_path, monkeypatch, make_error):
    """C17-5: an exception raised by g leaves gradient() as the same object
    (not wrapped or replaced, whatever its type), and the current directory
    is the same afterwards. The solver's work_dir does not exist, so an
    implementation that changes into it fails with FileNotFoundError instead.
    Fails if the exception is wrapped, converted (e.g. to NaN or another
    type), or if the directory is left changed."""
    monkeypatch.chdir(tmp_path)
    err = make_error()

    def g(x):
        raise err

    _, s = _c17_plain(tmp_path, f=_c17_f, g=g)
    assert not s.work_dir.exists()
    before = os.getcwd()
    with pytest.raises(type(err)) as ei:
        s.gradient(_C17_X.copy())
    assert ei.value is err
    assert os.getcwd() == before


def test_c17_gradient_keeps_cwd_and_needs_no_work_dir(tmp_path, monkeypatch):
    """C17-5: gradient() works while the solver's work_dir does not exist,
    calls g in the caller's current directory, and leaves the current
    directory unchanged. Fails if gradient() changes into work_dir as
    evaluate() does (FileNotFoundError here, or g seeing work_dir), or does
    not change back."""
    monkeypatch.chdir(tmp_path)
    seen = []

    def g(x):
        seen.append(os.getcwd())
        return _c17_g1(x)

    _, s = _c17_plain(tmp_path, f=_c17_f, g=g)
    assert not s.work_dir.exists()
    before = os.getcwd()
    np.testing.assert_array_equal(np.asarray(s.gradient(_C17_X.copy())), _c17_g1(_C17_X))
    assert seen == [before]
    assert os.getcwd() == before


# ---------------------------------------------------------------------------
# C17-6 Runner and min_search with a plain function solver


def test_c17_runner_uses_set_gradient(tmp_path, monkeypatch):
    """C17-6: for a plain function.Solver with set_function and set_gradient,
    Runner.gradient_unavailable_reason() is None (trivial mapping, solver
    group of 1) and submit_gradient returns the value of g at x without
    evaluating the objective. Fails if the Runner does not see the gradient
    set by set_gradient (e.g. still judges only by the class)."""
    f = _C17Recorder(_c17_f)
    g = _C17Recorder(_c17_g1)
    info, s = _c17_plain(tmp_path, f=f, g=g)
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert r.gradient_unavailable_reason() is None
    x = np.array([0.5, -0.75])
    np.testing.assert_array_equal(r.submit_gradient(x, (1, 0)), _c17_g1(x))
    assert len(g.calls) == 1
    np.testing.assert_array_equal(g.xs[0], x)
    assert f.calls == []


def _c17_plain_without_gradient(tmp_path, state, f, g, **info_kw):
    """A plain function solver with set_function only ("function-only") or
    with the gradient dropped by a later set_function ("dropped")."""
    info, s = _c17_plain(tmp_path, f=f, **info_kw)
    if state == "dropped":
        s.set_gradient(g)
        s.set_function(f)
    return info, s


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("state", ["function-only", "dropped"])
def test_c17_runner_reason_without_set_gradient(tmp_path, monkeypatch, state, ignore_error):
    """C17-6: a plain function solver with set_function only (or whose
    gradient a later set_function dropped) gets a reason string, and
    submit_gradient raises NotImplementedError with ignore_error true as
    well as false, calling neither the objective nor g. Fails if the Runner
    takes the solver as gradient-capable or turns the missing gradient into
    NaN under ignore_error."""
    f = _C17Recorder(_c17_f)
    g = _C17Recorder(_c17_g1)
    info, s = _c17_plain_without_gradient(tmp_path, state, f, g,
                                          runner={"ignore_error": ignore_error})
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason
    with pytest.raises(NotImplementedError):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    assert f.calls == []
    assert g.calls == []


@pytest.mark.parametrize("state", ["function-only", "dropped"])
def test_c17_min_search_rejects_use_gradient_without_set_gradient(tmp_path, state):
    """C17-6: min_search with use_gradient = true and a plain function solver
    without a gradient (set_function only, or dropped by a later
    set_function) raises InputValueError at construction, and neither the
    objective nor g is called. Fails if the run starts (with finite
    differences or all-NaN gradients)."""
    f = _C17Recorder(_c17_f)
    g = _C17Recorder(_c17_g1)
    info, s = _c17_plain_without_gradient(
        tmp_path, state, f, g,
        minimize={"method": "L-BFGS-B", "use_gradient": True},
        initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    r = odatse.Runner(s, info)
    with pytest.raises(odatse.exception.InputValueError):
        min_search.Algorithm(info, r)
    assert f.calls == []
    assert g.calls == []


def test_c17_min_search_lbfgsb_uses_set_gradient(tmp_path):
    """C17-6: min_search with use_gradient = true and L-BFGS-B on a plain
    function solver with set_function / set_gradient (a 2-D quadratic with
    minimum _C, started at (2, 2) on every rank) calls g, does not probe
    the objective by finite differences around the start, and converges to
    the minimum. Fails if the run is rejected, if g is never called or not
    handed to scipy as jac, or if the optimum is missed."""
    fn, grad = _quad_fn(_C)
    f = _C17Recorder(fn)
    g = _C17Recorder(grad)
    x0 = np.array([2.0, 2.0])
    info, s = _c17_plain(
        tmp_path, f=f, g=g,
        minimize={"method": "L-BFGS-B", "use_gradient": True},
        initial_list=_same_start_for_all_walkers(x0))
    r = odatse.Runner(s, info)
    alg = min_search.Algorithm(info, r)
    alg.main()
    assert len(g.calls) >= 2
    _assert_no_finite_difference_probe(
        types.SimpleNamespace(eval_calls=[(x, ()) for x in f.xs]), x0)
    xopt = np.asarray(alg.xopt, dtype=float)
    assert np.max(np.abs(xopt - x0)) > 0.1, xopt
    np.testing.assert_allclose(xopt, _C, atol=1e-4)


# ---------------------------------------------------------------------------
# C17-7 an assigned supports_gradient


@pytest.mark.parametrize("in_init", ["assign-only", "after-set-function",
                                     "after-set-gradient"])
@pytest.mark.parametrize("value", [True, False])
def test_c17_assigned_supports_gradient_is_kept(tmp_path, monkeypatch, value, in_init):
    """C17-7: a function solver subclass that assigns self.supports_gradient
    in __init__ (alone, or after set_function / set_gradient there) has
    that value, and later set_function / set_gradient calls in any order do
    not change it; the Runner follows the assigned value (False gives a
    reason even with a gradient set; True with a gradient set gives None).
    Fails if set_function forces it to False, if set_gradient forces it to
    True, if the assignment raises, or if the Runner judges by the gradient
    alone."""
    info = _info(tmp_path)
    if in_init == "assign-only":
        s = _C17Assigned(info, value)
    elif in_init == "after-set-function":
        s = _C17Assigned(info, value, f=_c17_f)
    else:
        s = _C17Assigned(info, value, f=_c17_f, g=_c17_g1)
    assert bool(s.supports_gradient) is value

    s.set_function(_c17_f)
    assert bool(s.supports_gradient) is value
    s.set_gradient(_c17_g1)
    assert bool(s.supports_gradient) is value
    s.set_function(_c17_f2)
    assert bool(s.supports_gradient) is value
    s.set_gradient(_c17_g2)
    assert bool(s.supports_gradient) is value
    s.set_gradient(_c17_g1)
    assert bool(s.supports_gradient) is value

    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    reason = r.gradient_unavailable_reason()
    if value:
        assert reason is None
        np.testing.assert_array_equal(
            r.submit_gradient(_C17_X.copy(), (1, 0)), _c17_g1(_C17_X))
    else:
        assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# C17-8 a read-only supports_gradient property


@pytest.mark.parametrize("value", [True, False])
def test_c17_read_only_property_is_left_alone(tmp_path, value):
    """C17-8: when a function solver subclass declares supports_gradient as
    a read-only property, set_function and set_gradient (in any order,
    repeatedly) raise nothing and the property's value stays the value of
    supports_gradient. Fails with AttributeError if either method assigns
    self.supports_gradient."""
    s = _C17_READ_ONLY[value](_info(tmp_path))
    assert bool(s.supports_gradient) is value
    s.set_function(_c17_f)
    assert bool(s.supports_gradient) is value
    s.set_gradient(_c17_g1)
    assert bool(s.supports_gradient) is value
    s.set_function(_c17_f2)
    assert bool(s.supports_gradient) is value
    s.set_gradient(_c17_g2)
    s.set_gradient(_c17_g1)
    assert bool(s.supports_gradient) is value


# ---------------------------------------------------------------------------
# C17-9 pickling


def _c17_pickle_case(info, case):
    if case == "gradient":
        s = odatse.solver.function.Solver(info)
        s.set_function(_c17_pickle_f)
        s.set_gradient(_c17_pickle_g)
    elif case == "function-only":
        s = odatse.solver.function.Solver(info)
        s.set_function(_c17_pickle_f)
    elif case == "dropped":
        s = odatse.solver.function.Solver(info)
        s.set_function(_c17_pickle_f)
        s.set_gradient(_c17_pickle_g)
        s.set_function(_c17_pickle_f)
    elif case == "assigned-true":
        s = _C17Assigned(info, True, f=_c17_pickle_f, g=_c17_pickle_g)
    elif case == "assigned-true-without-gradient":
        s = _C17Assigned(info, True, f=_c17_pickle_f)
    else:  # assigned-false, with a gradient set
        s = _C17Assigned(info, False, f=_c17_pickle_f, g=_c17_pickle_g)
    return s


# whether gradient() is specified in each case: a value of g, a
# NotImplementedError, or not specified by the contract (None)
_C17_PICKLE_CASES = {
    "gradient": "value",
    "function-only": "raises",
    "dropped": "raises",
    "assigned-true": "value",
    "assigned-true-without-gradient": None,
    "assigned-false": None,
}


@pytest.mark.parametrize("case", list(_C17_PICKLE_CASES))
def test_c17_pickle_round_trip(tmp_path, case):
    """C17-9: a function solver whose objective and g are module-level
    functions survives pickle.dumps / pickle.loads: the restored
    supports_gradient equals the original (including a value assigned by a
    subclass, C17-7), and gradient() gives the same value (or raises
    NotImplementedError when there is no gradient). Fails if pickling
    raises (e.g. a lambda or bound helper stored for g), if the gradient or
    its removal is lost, or if an assigned value is replaced on restore."""
    s = _c17_pickle_case(_info(tmp_path), case)
    t = pickle.loads(pickle.dumps(s))
    assert type(t) is type(s)
    assert bool(t.supports_gradient) is bool(s.supports_gradient)
    expected = case in ("gradient", "assigned-true", "assigned-true-without-gradient")
    assert bool(t.supports_gradient) is expected

    kind = _C17_PICKLE_CASES[case]
    if kind == "value":
        np.testing.assert_array_equal(np.asarray(t.gradient(_C17_X.copy())),
                                      np.asarray(s.gradient(_C17_X.copy())))
        np.testing.assert_array_equal(np.asarray(t.gradient(_C17_X.copy())),
                                      _c17_pickle_g(_C17_X))
    elif kind == "raises":
        with pytest.raises(NotImplementedError):
            t.gradient(_C17_X.copy())


# ---------------------------------------------------------------------------
# C6 revised: the analytical solver after set_function + set_gradient


@pytest.mark.parametrize("name,dim", _ALL_BUILTIN)
def test_c6r6_set_gradient_after_set_function(tmp_path, monkeypatch, name, dim):
    """C6 (revised in contract6): for every built-in function, after
    set_function() drops the built-in gradient, set_gradient(g) makes
    supports_gradient true again and gradient() returns g(x) (with x passed
    to g), not the built-in gradient; the Runner then reports no reason.
    Fails if analytical's gradient() keeps answering from its own (dropped)
    gradient, or if set_gradient is not honored for analytical."""
    s = _analytical(tmp_path, name, dim)
    s.set_function(lambda x: float(np.sum((x - 1.0) ** 2)))
    assert not s.supports_gradient

    g = _C17Recorder(lambda x: np.arange(1.0, x.size + 1.0) * x + 0.5)
    s.set_gradient(g)
    assert s.supports_gradient
    x = np.linspace(0.2, 0.9, dim)
    np.testing.assert_array_equal(np.asarray(s.gradient(x.copy())), g.fn(x))
    assert len(g.calls) == 1
    np.testing.assert_array_equal(g.xs[0], x)

    info = odatse.Info({
        "base": {"dimension": dim, "output_dir": str(tmp_path / "output")},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "analytical", "function_name": name},
        "runner": {},
    })
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert r.gradient_unavailable_reason() is None


def test_c6r6_set_function_again_drops_set_gradient(tmp_path):
    """C6 (revised) / C17-3 for analytical: set_function, set_gradient(g1),
    then set_function again drops g1 (supports_gradient false, gradient()
    raises NotImplementedError, g1 not called); set_gradient(g2) then gives
    g2. Fails if analytical's set_function only drops its built-in
    gradient but keeps one set by set_gradient."""
    s = _analytical(tmp_path, "quadratics", 2)
    g1 = _C17Recorder(_c17_g1)
    g2 = _C17Recorder(_c17_g2)
    s.set_function(_c17_f)
    s.set_gradient(g1)
    assert s.supports_gradient
    s.set_function(_c17_f2)
    assert not s.supports_gradient
    with pytest.raises(NotImplementedError):
        s.gradient(_C17_X.copy())
    assert g1.calls == []
    s.set_gradient(g2)
    assert s.supports_gradient
    np.testing.assert_array_equal(np.asarray(s.gradient(_C17_X.copy())), _c17_g2(_C17_X))
    assert g1.calls == []


def test_c6r6_builtin_gradient_unchanged(tmp_path):
    """C6 (unchanged in contract6): a built-in function keeps its gradient
    without set_gradient: quadratics reports supports_gradient true and its
    gradient 2x. Fails if the built-in gradients now need set_gradient or
    are replaced by something else."""
    s = _analytical(tmp_path, "quadratics", 2)
    assert s.supports_gradient
    np.testing.assert_allclose(np.asarray(s.gradient(_C17_X.copy())), 2.0 * _C17_X)


def test_c6r6_min_search_with_replaced_function_and_set_gradient(tmp_path):
    """C6 (revised): min_search with use_gradient = true accepts an
    analytical solver whose objective was replaced by set_function and whose
    gradient was given by set_gradient, calls that g and converges to the
    replaced function's minimum _C (not the built-in quadratics' origin).
    Fails if the replaced solver is rejected or the built-in gradient is
    used."""
    fn, grad = _quad_fn(_C)
    g = _C17Recorder(grad)
    info = _info(tmp_path, minimize={"method": "L-BFGS-B", "use_gradient": True},
                 initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    s = odatse.solver.analytical.Solver(info)
    s.set_function(fn)
    s.set_gradient(g)
    alg = min_search.Algorithm(info, odatse.Runner(s, info))
    alg.main()
    assert len(g.calls) >= 2
    np.testing.assert_allclose(np.asarray(alg.xopt, dtype=float), _C, atol=1e-4)


# ---------------------------------------------------------------------------
# C14 revised: declared support without gradient() and without a gradient


class _C14r6Property(odatse.solver.function.Solver):
    """Declares supports_gradient as a true property; no gradient()."""

    @property
    def supports_gradient(self):
        return True


class _C14r6ClassAttr(odatse.solver.function.Solver):
    """Declares supports_gradient as a true class attribute; no gradient()."""

    supports_gradient = True


class _C14r6Analytical(odatse.solver.analytical.Solver):
    """analytical subclass replacing the objective, then assigning
    supports_gradient = True; no gradient()."""

    def __init__(self, info, f):
        super().__init__(info)
        self.set_function(f)
        self.supports_gradient = True


_C14R6_KINDS = ["property", "class-attr", "property-dropped",
                "class-attr-dropped", "analytical", "analytical-dropped"]


def _c14r6_solver(kind, info, f, g):
    """A function solver subclass with a true supports_gradient, without a
    gradient() override and without a gradient: set_gradient never called,
    or ("-dropped") called and then dropped by set_function."""
    if kind.startswith("property"):
        s = _C14r6Property(info)
        s.set_function(f)
    elif kind.startswith("class-attr"):
        s = _C14r6ClassAttr(info)
        s.set_function(f)
    else:
        s = _C14r6Analytical(info, f)
    if kind.endswith("-dropped"):
        s.set_gradient(g)
        s.set_function(f)
    return s


@pytest.mark.parametrize("kind", _C14R6_KINDS)
def test_c14r6_reason_without_gradient(tmp_path, monkeypatch, kind):
    """C14 (revised): a function solver subclass (analytical subclasses
    included) with a true supports_gradient, no gradient() override and no
    gradient (never set, or dropped by set_function) gets a reason string
    from Runner.gradient_unavailable_reason(). Fails if the Runner only
    checks whether the class differs from SolverBase.gradient (function or
    analytical then count as an override) and ignores the missing g."""
    f = _C17Recorder(_c17_f)
    info = _info(tmp_path)
    s = _c14r6_solver(kind, info, f, _c17_g1)
    assert s.supports_gradient
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason
    assert f.calls == []


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("kind", _C14R6_KINDS)
def test_c14r6_submit_gradient_raises_without_gradient(tmp_path, monkeypatch, kind, ignore_error):
    """C14 (revised): for those solvers submit_gradient raises
    NotImplementedError regardless of ignore_error (not an all-NaN
    gradient) because a reason is reported, and neither the objective nor
    a dropped g is called. Fails if ignore_error turns the missing gradient
    into NaN, or if the check happens only inside solver.gradient()."""
    f = _C17Recorder(_c17_f)
    g = _C17Recorder(_c17_g1)
    info = _info(tmp_path, runner={"ignore_error": ignore_error})
    s = _c14r6_solver(kind, info, f, g)
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert isinstance(r.gradient_unavailable_reason(), str)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(np.array([0.5, -0.75]), (1, 0))
    assert f.calls == []
    assert g.calls == []


@pytest.mark.parametrize("kind", _C14R6_KINDS)
def test_c14r6_min_search_rejects_without_gradient(tmp_path, kind):
    """C14 (revised): min_search with use_gradient = true raises
    InputValueError (a ValueError and an InputError) at construction for
    those solvers, without calling the objective or g. Fails if the run
    starts and proceeds with all-NaN gradients."""
    f = _C17Recorder(_c17_f)
    g = _C17Recorder(_c17_g1)
    info = _info(tmp_path, minimize={"method": "L-BFGS-B", "use_gradient": True},
                 initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    s = _c14r6_solver(kind, info, f, g)
    r = odatse.Runner(s, info)
    with pytest.raises(odatse.exception.InputValueError) as ei:
        min_search.Algorithm(info, r)
    assert isinstance(ei.value, ValueError)
    assert isinstance(ei.value, odatse.exception.InputError)
    assert f.calls == []
    assert g.calls == []


@pytest.mark.parametrize("kind", ["property", "class-attr", "analytical"])
def test_c14r6_no_reason_once_gradient_is_set(tmp_path, monkeypatch, kind):
    """C14 (revised) / C17: the same subclasses, without a gradient()
    override, become available once set_gradient(g) is called: the reason is
    None and submit_gradient returns g(x). Fails if every subclass without
    an override is rejected regardless of the gradient set on it."""
    g = _C17Recorder(_c17_g1)
    info = _info(tmp_path)
    s = _c14r6_solver(kind, info, _c17_f, None)
    s.set_gradient(g)
    assert s.supports_gradient
    r = odatse.Runner(s, info)
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: 1)
    assert r.gradient_unavailable_reason() is None
    np.testing.assert_array_equal(r.submit_gradient(_C17_X.copy(), (1, 0)), _c17_g1(_C17_X))
    assert len(g.calls) == 1


# ===========================================================================
# Round 7: C18 / C1-C3 revisions
#
# Gradients with a parallel solver (--nsolve > 1). A solver group of
# several ranks no longer makes gradients unavailable (C1-C3 revised); the
# controller sends MSG_GRADIENT, xp and args to the workers before its own
# gradient(), every rank joins one status exchange per gradient, and the
# workers serve the request in Runner.serve_gradient() from the worker loop
# of AlgorithmBase.main() (C18).
#
# The solver group is faked on one process, as in test_runner_errors.py:
# odatse.mpi.solsize/solrank/solcomm/rank are patched after the Runner is
# built (see _make_runner), except where a test builds the objects inside
# the fake group on purpose (the fake then answers the construction-time
# agreement). The real multi-rank protocol is covered by
# tests/parallel_solver/minsearch_gradient.

import re

from odatse._runner import _status_entry, _is_status_entry
from odatse.exception import SolverError, SolverRuntimeError, describe_error, is_ignorable


def _c18_tag(entry):
    """A test's plain status (None, or (is_ignorable, summary)) tagged the
    way the status exchange tags it; anything else is foreign data."""
    if entry is None:
        return _status_entry()
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], bool):
        return _status_entry(*entry)
    return entry


class _C18Solcomm:
    """Fake solcomm recording, in order, every call this rank makes.

    ``log`` holds ("Bcast", copy of the buffer after the call),
    ("bcast", the object returned), ("allgather", this rank's entry) and
    ("Allreduce", copy of the send buffer). On the root (``me == 0``) Bcast
    and bcast send, i.e. leave the data as it is. On a worker they receive:
    Bcast copies the next item of ``incoming`` into the buffer and bcast
    returns the next item. allgather returns the entries of the other ranks
    (``others``, tagged by _c18_tag) around this rank's own entry, and
    Allreduce sums identical contributions of every rank (the construction-
    time agreement of FailTogetherMeta, where every rank succeeds)."""

    def __init__(self, me=0, others=(None,), incoming=()):
        self.me = me
        self.others = [_c18_tag(o) for o in others]
        self.incoming = list(incoming)
        self.log = []

    @property
    def size(self):
        return len(self.others) + 1

    def Bcast(self, buf, root=0):
        if self.me != root:
            buf[...] = self.incoming.pop(0)
        self.log.append(("Bcast", np.array(buf, copy=True)))

    def bcast(self, obj, root=0):
        if self.me != root:
            obj = self.incoming.pop(0)
        self.log.append(("bcast", obj))
        return obj

    def allgather(self, own):
        self.log.append(("allgather", own))
        return self.others[:self.me] + [own] + self.others[self.me:]

    def Allreduce(self, sendbuf, recvbuf, op=None):
        self.log.append(("Allreduce", np.array(sendbuf, copy=True)))
        recvbuf[...] = sendbuf * self.size

    def kinds(self):
        return [k for k, _ in self.log]


def _c18_group(monkeypatch, comm, global_rank=None):
    """Make odatse.mpi report a solver group of comm.size ranks in which this
    process is rank comm.me of the group (global rank ``global_rank``)."""
    monkeypatch.setattr(odatse.mpi, "solsize", lambda: comm.size)
    monkeypatch.setattr(odatse.mpi, "solrank", lambda: comm.me)
    monkeypatch.setattr(odatse.mpi, "solcomm", lambda: comm)
    monkeypatch.setattr(odatse.mpi, "rank",
                        lambda: comm.me if global_rank is None else global_rank)
    return comm


def _c18_abort_recorder(monkeypatch):
    """Patch odatse.mpi.comm() so that Abort(code) is recorded instead of
    killing the process; returns the list of recorded codes."""
    aborted = []

    class _Comm:
        def Abort(self, code=0):
            aborted.append(code)

    monkeypatch.setattr(odatse.mpi, "comm", lambda: _Comm())
    return aborted


def _c18_msg_gradient():
    # looked up at run time so that a missing constant fails the test that
    # needs it, not the collection of the module
    return odatse.mpi.MSG_GRADIENT


def _c18_controller(tmp_path, monkeypatch, others=(None,), grad=None,
                    global_rank=None, **kw):
    """Runner around a _RecordingSolver, then a fake group with this process
    as the controller. ``grad`` is wrapped so that each gradient() call also
    records how much of comm.log had been sent at that moment."""
    comm = _C18Solcomm(0, others)
    at_call = []
    if grad is None:
        grad = _quad_fn(_C)[1]

    def recording_grad(x):
        at_call.append(list(comm.log))
        return grad(x)

    _, solver, r = _make_runner(tmp_path, fn=_quad_fn(_C)[0], grad=recording_grad, **kw)
    _c18_group(monkeypatch, comm, global_rank)
    return solver, r, comm, at_call


def _c18_requests(log):
    """[(signal, xp, args)] of the control messages in a controller's log: a
    one-element Bcast followed by the Bcast of xp and the bcast of args (the
    final MSG_FINISHED / MSG_ABORT have no xp)."""
    out = []
    for i, (kind, val) in enumerate(log):
        if kind == "Bcast" and np.asarray(val).shape == (1,):
            sig = int(np.asarray(val)[0])
            if (i + 2 < len(log) and log[i + 1][0] == "Bcast"
                    and np.asarray(log[i + 1][1]).shape != (1,)
                    and log[i + 2][0] == "bcast"):
                out.append((sig, np.array(log[i + 1][1], dtype=float), log[i + 2][1]))
            else:
                out.append((sig, None, None))
    return out


def _c18_assert_gradient_summary(msg, n, lines):
    """The controller's summary of a group failure names solver.gradient()
    (not solver.evaluate()), the number of failing ranks, and one
    "[rank N] Type: message" line per failing rank."""
    assert "solver.gradient()" in msg, msg
    assert "solver.evaluate()" not in msg, msg
    assert re.search(rf"failed on {n} rank\(s\)", msg), msg
    for line in lines:
        assert line in msg, msg


_C18_X = np.array([0.5, -0.75])
_C18_OK = None
_C18_WORKER_RTE = (True, "[rank 3] RuntimeError: worker boom")
_C18_WORKER_NIE = (True, "[rank 3] NotImplementedError: worker has no gradient here")
_C18_WORKER_VAL = (False, "[rank 3] ValueError: worker boom")


# ---------------------------------------------------------------------------
# C1 / C2 / C3 revised: a parallel solver alone is no reason


@pytest.mark.parametrize("solsize", [2, 4])
def test_c1r7_reason_none_for_parallel_solver(tmp_path, monkeypatch, solsize):
    """C1 (revised): with a solver group of 2 or more processes and every
    other condition met (gradient-capable solver, trivial mapping),
    gradient_unavailable_reason() is None. Fails if solsize() > 1 is still
    reported as a reason. (Replaces test_c1_reason_for_parallel_solver.)"""
    _, _, r = _make_runner(tmp_path, monkeypatch=monkeypatch, solsize=solsize)
    assert r.gradient_unavailable_reason() is None


def _c18_unavailable_runner(tmp_path, situation, runner=None, limitation=None):
    """A Runner whose gradient is unavailable for a reason other than the
    solver group, with the solver's evaluate()/gradient() calls recorded in
    solver.eval_calls (and solver.grad_calls where gradient() exists)."""
    if situation in ("unsupported", "affine"):
        kw = {"supports": False} if situation == "unsupported" else {
            "mapping": odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))}
        _, solver, r = _make_runner(tmp_path, runner=runner, limitation=limitation, **kw)
        return solver, r
    info = _info(tmp_path, runner=runner)
    if situation == "base-no-override":
        solver = _NoOverrideBaseSolver(info)
    elif situation == "function-no-override":
        solver = _NoOverrideFunctionSolver(info)
    else:  # "function-no-set-gradient": a plain function solver
        f = _C17Recorder(_quad_fn(_C)[0])
        solver = odatse.solver.function.Solver(info)
        solver.set_function(f)
        solver.eval_calls = f.calls    # the objective's calls
    return solver, odatse.Runner(solver, info, limitation=limitation)


_C18_UNAVAILABLE = ["unsupported", "affine", "base-no-override",
                    "function-no-override", "function-no-set-gradient"]


@pytest.mark.parametrize("situation", _C18_UNAVAILABLE)
def test_c1r7_other_reasons_remain_in_parallel(tmp_path, monkeypatch, situation):
    """C1 (revised, unchanged part): in a solver group of 2, every other
    reason (no gradient support, Affine mapping, gradient() not overridden,
    a function solver without set_gradient) is still reported. Fails if
    a parallel solver group short-circuits the remaining checks to None."""
    _, r = _c18_unavailable_runner(tmp_path, situation)
    _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    reason = r.gradient_unavailable_reason()
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize("situation", ["unsupported", "affine"])
def test_c2r7_unavailable_precedes_limitation_in_parallel(tmp_path, monkeypatch, situation):
    """C2 (revised, unchanged part) / C18(a)2: in a solver group of 2, an
    unavailable gradient raises NotImplementedError even at an x violating
    the limitation, and nothing is sent to the workers. Fails if the
    limitation is judged first (an unusable gradient would pass as NaN) or
    if the control message is sent before the availability check.
    (Replaces the "parallel" cases of test_c2_submit_gradient_raises_when_unavailable
    and test_c2_unavailable_raises_even_when_limitation_violated.)"""
    lim = odatse.util.limitation.Inequality(np.array([[1.0, 0.0]]), np.array([0.0]), True)
    solver, r = _c18_unavailable_runner(tmp_path, situation, limitation=lim)
    comm = _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    x = np.array([-1.0, 1.0])
    assert not r.limitation.judge(x)
    with pytest.raises(NotImplementedError):
        r.submit_gradient(x, (1, 0))
    assert comm.log == []
    assert solver.grad_calls == []


def test_c3r7_min_search_accepts_use_gradient_in_parallel(tmp_path, monkeypatch):
    """C3 (revised): min_search with use_gradient = true is built without
    error in a solver group of 2 when the gradient is otherwise available,
    and the solver is not called at construction. The objects are built
    inside the fake group (the fake answers the construction agreement),
    so that the constructor sees solsize() == 2. Fails if a parallel solver
    is still rejected with InputValueError."""
    comm = _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    alg, solver, r = _make_alg(tmp_path, {"method": "L-BFGS-B", "use_gradient": True},
                               fn=_quad_fn(_C)[0], grad=_quad_fn(_C)[1],
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    assert odatse.mpi.solsize() == 2
    assert alg.use_gradient is True
    assert solver.eval_calls == [] and solver.grad_calls == []
    # nothing but the construction agreement was communicated
    assert set(comm.kinds()) <= {"Allreduce", "bcast"}


@pytest.mark.parametrize("situation", ["unsupported", "affine"])
def test_c3r7_min_search_rejects_other_reasons_in_parallel(tmp_path, monkeypatch, situation):
    """C3 (revised, unchanged part): in a solver group of 2, use_gradient =
    true with an unsupported solver or an Affine mapping still raises
    InputValueError at construction, without calling the solver. Fails if
    the parallel case skips the validation."""
    comm = _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    fn, grad = _quad_fn(_C)
    info = _info(tmp_path, minimize={"method": "L-BFGS-B", "use_gradient": True},
                 initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    solver = _RecordingSolver(info, fn, grad, supports=(situation != "unsupported"))
    mapping = (odatse.util.mapping.Affine(A=2.0 * np.eye(2), b=np.zeros(2))
               if situation == "affine" else None)
    r = odatse.Runner(solver, info, mapping=mapping)
    with pytest.raises(odatse.exception.InputValueError):
        min_search.Algorithm(info, r)
    assert solver.eval_calls == [] and solver.grad_calls == []
    assert "allgather" not in comm.kinds()


# ---------------------------------------------------------------------------
# C18(a) the control message on the controller


def test_c18_msg_gradient_is_a_distinct_integer():
    """C18(a)1: odatse.mpi exposes MSG_GRADIENT, an integer different from
    MSG_EVALUATE, MSG_FINISHED and MSG_ABORT. Fails if the constant is
    missing, not an integer, or collides with another signal (a worker
    would then evaluate instead of computing a gradient, or stop)."""
    msg = _c18_msg_gradient()
    assert isinstance(msg, (int, np.integer)) and not isinstance(msg, bool)
    others = {odatse.mpi.MSG_EVALUATE, odatse.mpi.MSG_FINISHED, odatse.mpi.MSG_ABORT}
    assert int(msg) not in {int(v) for v in others}


def test_c18_controller_sends_signal_xp_args_before_gradient(tmp_path, monkeypatch):
    """C18(a)1, (a)3, (a)5: on the controller of a group of 2,
    submit_gradient sends, before calling its own gradient(), exactly the
    one-element integer array [MSG_GRADIENT], then xp, then args (in that
    order, Bcast / Bcast / bcast); after gradient() it takes part in one
    status exchange (allgather), and returns its own gradient as a float
    ndarray. Fails if nothing is sent (the workers would never compute),
    if MSG_EVALUATE is sent instead, if the order differs, if the exchange
    happens before gradient() or is missing, or if xp / args are altered."""
    solver, r, comm, at_call = _c18_controller(
        tmp_path, monkeypatch, grad=lambda x: np.array([3.5, -1.25]))
    g = r.submit_gradient(_C18_X.copy(), (7, 3))

    assert len(at_call) == 1
    before = at_call[0]
    assert [k for k, _ in before] == ["Bcast", "Bcast", "bcast"]
    sig = before[0][1]
    assert sig.shape == (1,) and np.issubdtype(sig.dtype, np.integer)
    assert int(sig[0]) == int(_c18_msg_gradient())
    np.testing.assert_array_equal(before[1][1], _C18_X)
    assert before[2][1] == (7, 3)
    # exactly one exchange, after gradient(), and nothing else
    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"]

    assert isinstance(g, np.ndarray) and g.dtype == np.float64
    np.testing.assert_array_equal(g, [3.5, -1.25])
    xs, args = solver.grad_calls[0]
    np.testing.assert_array_equal(xs, _C18_X)
    assert args == (7, 3)


def test_c18_one_exchange_per_gradient_and_evaluate_unchanged(tmp_path, monkeypatch):
    """C18(a)3, (d): requests interleaved on the controller keep their own
    signals: submit() sends MSG_EVALUATE and submit_gradient() sends
    MSG_GRADIENT, each followed by its xp and args, and each request has
    exactly one status exchange. Fails if gradient() is exchanged twice (or
    not at all), if a gradient request reuses MSG_EVALUATE, or if submit()
    starts sending MSG_GRADIENT."""
    solver, r, comm, _ = _c18_controller(tmp_path, monkeypatch)
    solver.work_dir.mkdir(parents=True, exist_ok=True)   # evaluate() runs there
    xa, xb, xc = np.array([0.5, 0.25]), np.array([-1.0, 2.0]), np.array([1.5, -0.5])
    r.submit_gradient(xa.copy(), (1, 0))
    r.submit(xb.copy(), (2, 0))
    r.submit_gradient(xc.copy(), (3, 1))

    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"] * 3
    reqs = _c18_requests(comm.log)
    G, E = int(_c18_msg_gradient()), int(odatse.mpi.MSG_EVALUATE)
    assert [s for s, _, _ in reqs] == [G, E, G]
    for (_, xp, args), (x, a) in zip(reqs, [(xa, (1, 0)), (xb, (2, 0)), (xc, (3, 1))]):
        np.testing.assert_array_equal(xp, x)
        assert args == a
    assert len(solver.grad_calls) == 2 and len(solver.eval_calls) == 1


def test_c18_single_rank_gradient_uses_no_collective(tmp_path, monkeypatch):
    """C18(a) applies to solsize() >= 2 only: with a solver group of one,
    submit_gradient neither signals nor exchanges anything. Fails if the
    control message or the exchange is issued unconditionally."""
    _, solver, r = _make_runner(tmp_path, monkeypatch=monkeypatch, solsize=1)

    def forbidden():
        raise AssertionError("solcomm() must not be used when solsize() == 1")
    monkeypatch.setattr(odatse.mpi, "solcomm", forbidden)
    np.testing.assert_allclose(r.submit_gradient(_C18_X.copy(), (1, 0)), 2.0 * _C18_X)
    assert len(solver.grad_calls) == 1


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("situation", _C18_UNAVAILABLE)
def test_c18_unavailable_sends_nothing(tmp_path, monkeypatch, situation, ignore_error):
    """C18(a)2: when the gradient is unavailable, the controller of a group
    of 2 raises NotImplementedError (regardless of ignore_error) without
    sending anything and without calling gradient() or evaluate(). Fails if
    the control message goes out first: the workers would compute (and wait
    in the exchange) for a request the controller abandons."""
    solver, r = _c18_unavailable_runner(tmp_path, situation,
                                        runner={"ignore_error": ignore_error})
    comm = _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    with pytest.raises(NotImplementedError):
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert comm.log == []
    assert solver.eval_calls == []
    assert getattr(solver, "grad_calls", []) == []


def test_c18_limitation_violation_sends_nothing(tmp_path, monkeypatch):
    """C18(a)2: at an x violating the limitation the controller of a group
    of 2 returns an all-NaN gradient without sending anything and without
    calling gradient(). Fails if the request is sent anyway (the workers
    would compute a gradient nobody uses) or if NaN is not returned."""
    lim = odatse.util.limitation.Inequality(np.array([[1.0, 0.0]]), np.array([0.0]), True)
    solver, r, comm, _ = _c18_controller(tmp_path, monkeypatch, limitation=lim)
    x = np.array([-1.0, 1.0])
    assert not r.limitation.judge(x)
    _assert_all_nan(r.submit_gradient(x, (1, 0)), (2,))
    assert comm.log == []
    assert solver.grad_calls == []


@pytest.mark.parametrize("x", [
    np.array([0.5, -0.75, 1.0]),
    np.array([0.5]),
    np.array([[0.5, -0.75]]),
], ids=["len3", "len1", "2d"])
def test_c18_dimension_mismatch_sends_nothing(tmp_path, monkeypatch, x):
    """C18(a)4: an xp that is not one-dimensional or whose length is not
    solver.dimension (2 here) is an error on the controller of a group of
    2, raised before anything is sent and without calling gradient(). The
    error is not the NotImplementedError of an unavailable gradient (the
    gradient is available here). Fails if the request is broadcast (a worker
    would receive xp into a buffer of solver.dimension elements and the
    group would desynchronise) or if the check is missing."""
    solver, r, comm, _ = _c18_controller(tmp_path, monkeypatch)
    assert r.gradient_unavailable_reason() is None
    with pytest.raises(Exception) as ei:
        r.submit_gradient(x, (1, 0))
    assert not isinstance(ei.value, NotImplementedError)
    assert comm.log == []
    assert solver.grad_calls == []


def test_c18_converts_controller_gradient_to_float_ndarray(tmp_path, monkeypatch):
    """C18(a)5: the value returned is the controller's own gradient()
    converted to a float ndarray (a list of ints here). Fails if the raw
    value is passed through or a value from another rank is used."""
    _, r, _, _ = _c18_controller(tmp_path, monkeypatch, grad=lambda x: [3, -4])
    g = r.submit_gradient(_C18_X.copy(), (1, 0))
    assert isinstance(g, np.ndarray) and np.issubdtype(g.dtype, np.floating)
    np.testing.assert_array_equal(g, [3.0, -4.0])


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("bad", [np.array([1.0]), np.array([1.0, 2.0, 3.0]), 1.0],
                         ids=["len1", "len3", "scalar"])
def test_c18_wrong_shape_raises_after_exchange(tmp_path, monkeypatch, bad, ignore_error):
    """C18(a)5: a controller gradient of the wrong shape raises ValueError
    (also with ignore_error = true), and only after the status exchange, so
    that the workers waiting in it are released. Fails if the shape is
    checked before the exchange (the workers would hang) or not at all."""
    _, r, comm, _ = _c18_controller(tmp_path, monkeypatch, grad=lambda x: bad,
                                    runner={"ignore_error": ignore_error})
    with pytest.raises(ValueError):
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"]


# ---------------------------------------------------------------------------
# C18(b) failures in the group, decided on the controller


@pytest.mark.parametrize("worker", [_C18_WORKER_RTE, _C18_WORKER_NIE], ids=["runtime", "not-implemented"])
def test_c18_worker_runtime_error_is_nan_with_ignore_error(tmp_path, monkeypatch, worker):
    """C18(b)1: a worker's RuntimeError (NotImplementedError included) in
    gradient() gives an all-NaN gradient with ignore_error = true, after
    one exchange. Fails if the controller returns its own (finite) gradient
    as if the group had succeeded, or raises."""
    solver, r, comm, _ = _c18_controller(tmp_path, monkeypatch, others=[worker],
                                         runner={"ignore_error": True})
    _assert_all_nan(r.submit_gradient(_C18_X.copy(), (1, 0)), (2,))
    assert comm.kinds().count("allgather") == 1
    assert len(solver.grad_calls) == 1


@pytest.mark.parametrize("worker", [_C18_WORKER_RTE, _C18_WORKER_NIE], ids=["runtime", "not-implemented"])
def test_c18_worker_runtime_error_raises_solver_runtime_error(tmp_path, monkeypatch, worker):
    """C18(b)1, (b)4: without ignore_error a worker's RuntimeError raises
    SolverRuntimeError (a RuntimeError and a SolverError) whose message
    names solver.gradient(), the number of failing ranks and the worker's
    "[rank N] Type: message". Fails if the failure is swallowed, if the
    worker's exception type is lost, or if the message says
    solver.evaluate() (the evaluate wording copied unchanged)."""
    _, r, _, _ = _c18_controller(tmp_path, monkeypatch, others=[worker])
    with pytest.raises(SolverRuntimeError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert isinstance(ei.value, RuntimeError) and isinstance(ei.value, SolverError)
    _c18_assert_gradient_summary(str(ei.value), 1, [worker[1]])


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("exc_type", [RuntimeError, NotImplementedError])
def test_c18_controller_only_runtime_error(tmp_path, monkeypatch, exc_type, ignore_error):
    """C18(b)1: when only the controller's gradient() raises a
    RuntimeError (or NotImplementedError), ignore_error = true gives an
    all-NaN gradient and otherwise that very exception is raised, after
    one exchange (the workers must be released). Fails if the exception is
    wrapped or replaced, if ignore_error does not apply, or if the
    controller raises before the exchange."""
    own = exc_type("controller boom")
    _, r, comm, _ = _c18_controller(tmp_path, monkeypatch, grad=_raising(own),
                                    runner={"ignore_error": ignore_error})
    if ignore_error:
        _assert_all_nan(r.submit_gradient(_C18_X.copy(), (1, 0)), (2,))
    else:
        with pytest.raises(exc_type) as ei:
            r.submit_gradient(_C18_X.copy(), (1, 0))
        assert ei.value is own
    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"]


@pytest.mark.parametrize("ignore_error", [False, True])
def test_c18_controller_and_worker_runtime_error(tmp_path, monkeypatch, ignore_error):
    """C18(b)1: when the controller and a worker both raise RuntimeError,
    ignore_error = true gives NaN, and otherwise a SolverRuntimeError lists
    both ranks. Fails if only the controller's own exception is raised
    (the worker's failure would be hidden) or if ignore_error is refused."""
    _, r, _, _ = _c18_controller(tmp_path, monkeypatch, others=[_C18_WORKER_RTE],
                                 grad=_raising(RuntimeError("controller boom")),
                                 global_rank=2, runner={"ignore_error": ignore_error})
    if ignore_error:
        _assert_all_nan(r.submit_gradient(_C18_X.copy(), (1, 0)), (2,))
        return
    with pytest.raises(SolverRuntimeError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    _c18_assert_gradient_summary(
        str(ei.value), 2, ["[rank 2] RuntimeError: controller boom", _C18_WORKER_RTE[1]])


@pytest.mark.parametrize("ignore_error", [False, True])
def test_c18_worker_non_runtime_error_is_solver_error(tmp_path, monkeypatch, ignore_error):
    """C18(b)2, (b)4: a worker's non-RuntimeError (ValueError) in gradient()
    raises a SolverError that is not a RuntimeError, regardless of
    ignore_error, naming solver.gradient() and the worker's line. Fails if
    ignore_error turns it into NaN or the controller's finite gradient is
    returned."""
    _, r, _, _ = _c18_controller(tmp_path, monkeypatch, others=[_C18_WORKER_VAL],
                                 runner={"ignore_error": ignore_error})
    with pytest.raises(SolverError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert not isinstance(ei.value, RuntimeError)
    _c18_assert_gradient_summary(str(ei.value), 1, [_C18_WORKER_VAL[1]])


@pytest.mark.parametrize("ignore_error", [False, True])
def test_c18_controller_only_non_runtime_error_is_unchanged(tmp_path, monkeypatch, ignore_error):
    """C18(b)2: a non-RuntimeError raised by the controller's gradient()
    alone is raised as it is (the same object), regardless of
    ignore_error, after the exchange. Fails if it is wrapped, ignored, or
    raised before the workers are released."""
    own = ValueError("controller boom")
    _, r, comm, _ = _c18_controller(tmp_path, monkeypatch, grad=_raising(own),
                                    runner={"ignore_error": ignore_error})
    with pytest.raises(ValueError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert ei.value is own
    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"]


@pytest.mark.parametrize("case", ["controller-value", "worker-value"])
def test_c18_mixed_failures_with_a_worker_are_solver_error(tmp_path, monkeypatch, case):
    """C18(b)2: a non-RuntimeError anywhere in the group, together with a
    worker failure, raises SolverError even with ignore_error = true.
    Fails if the RuntimeError of one rank makes the whole failure
    ignorable."""
    if case == "controller-value":
        own, worker = ValueError("controller boom"), _C18_WORKER_RTE
    else:
        own, worker = RuntimeError("controller boom"), _C18_WORKER_VAL
    _, r, _, _ = _c18_controller(tmp_path, monkeypatch, others=[worker], grad=_raising(own),
                                 global_rank=2, runner={"ignore_error": True})
    with pytest.raises(SolverError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert not isinstance(ei.value, RuntimeError)
    assert "solver.gradient()" in str(ei.value)
    assert "solver.evaluate()" not in str(ei.value)


@pytest.mark.parametrize("exc", [SystemExit(3), KeyboardInterrupt()], ids=["exit", "interrupt"])
def test_c18_controller_base_exception_is_solver_error(tmp_path, monkeypatch, exc):
    """C18(b)3, (b)4: SystemExit / KeyboardInterrupt from the controller's
    gradient() takes part in the exchange (the workers wait in it) and is
    then raised as a SolverError, also with ignore_error = true, whose
    message names solver.gradient(). Fails if it propagates as is before the
    exchange (the workers would hang), if ignore_error turns it into NaN, or
    if the message says solver.evaluate()."""
    _, r, comm, _ = _c18_controller(tmp_path, monkeypatch, grad=_raising(exc),
                                    runner={"ignore_error": True})
    with pytest.raises(SolverError) as ei:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    assert comm.kinds() == ["Bcast", "Bcast", "bcast", "allgather"]
    assert "solver.gradient()" in str(ei.value)
    assert "solver.evaluate()" not in str(ei.value)


@pytest.mark.parametrize("foreign", [1.5, None, (True, "looks like a status")])
def test_c18_garbage_in_exchange_aborts(tmp_path, monkeypatch, capsys, foreign):
    """C18(b)5: if the status exchange of a gradient receives an entry that
    is not a tagged status (the exchange was paired with a collective of
    the solver's gradient()), the job is aborted with comm().Abort(1), and
    the report names solver.gradient(), not solver.evaluate(). Fails if
    the foreign entry is taken for a status (e.g. None read as success) and
    the run goes on with a desynchronised group, or if the report does not
    name solver.gradient()."""
    assert not _is_status_entry(foreign)
    aborted = _c18_abort_recorder(monkeypatch)
    _, r, comm, _ = _c18_controller(tmp_path, monkeypatch, others=[])
    comm.others = [foreign]            # raw data from the solver, not tagged
    raised = None
    try:
        r.submit_gradient(_C18_X.copy(), (1, 0))
    except BaseException as e:         # after the (recorded) Abort, anything goes
        raised = e
    assert aborted == [1]
    texts = capsys.readouterr().err + ("" if raised is None else str(raised))
    assert "solver.gradient()" in texts
    assert "solver.evaluate()" not in texts


# ---------------------------------------------------------------------------
# C18(c)1 Runner.serve_gradient() on a worker


def test_c18_serve_gradient_calls_gradient_and_joins_exchange(tmp_path, monkeypatch):
    """C18(c)1: on a worker, serve_gradient(xp, args) calls gradient() with
    exactly xp and args, takes part in one status exchange (reporting
    success), sends nothing else and returns None. Fails if the method is
    missing, if evaluate() is called instead, if the worker broadcasts, or
    if it skips the exchange (the controller would hang in it)."""
    _, solver, r = _make_runner(tmp_path)
    comm = _c18_group(monkeypatch, _C18Solcomm(1, [_C18_OK]), global_rank=3)
    xp = np.array([0.25, -1.5])
    assert r.serve_gradient(xp, (4, 1)) is None
    assert len(solver.grad_calls) == 1 and solver.eval_calls == []
    np.testing.assert_array_equal(solver.grad_calls[0][0], xp)
    assert solver.grad_calls[0][1] == (4, 1)
    assert comm.kinds() == ["allgather"]
    sent = comm.log[0][1]
    assert _is_status_entry(sent) and sent[1] is None


@pytest.mark.parametrize("ignore_error", [False, True])
@pytest.mark.parametrize("exc", [
    RuntimeError("worker boom"), NotImplementedError("no gradient here"),
    ValueError("worker boom"), SystemExit(3), KeyboardInterrupt(),
], ids=["runtime", "not-implemented", "value", "exit", "interrupt"])
def test_c18_serve_gradient_never_raises(tmp_path, monkeypatch, exc, ignore_error):
    """C18(c)1: whatever the worker's gradient() raises (RuntimeError,
    NotImplementedError, ValueError, SystemExit, KeyboardInterrupt),
    serve_gradient raises nothing, returns None and reports the failure in
    its one exchange: a tagged failure status whose ignorable flag is
    is_ignorable(exc) and whose summary carries "[rank 3]" and the
    exception's "Type: message". Fails if the exception escapes (the
    worker would leave the group and the controller hang) or if the
    failure is reported as a success."""
    _, solver, r = _make_runner(tmp_path, grad=_raising(exc),
                                runner={"ignore_error": ignore_error})
    comm = _c18_group(monkeypatch, _C18Solcomm(1, [_C18_OK]), global_rank=3)
    assert r.serve_gradient(_C18_X.copy(), (1, 0)) is None
    assert len(solver.grad_calls) == 1
    assert comm.kinds() == ["allgather"]
    sent = comm.log[0][1]
    assert _is_status_entry(sent) and sent[1] is not None
    flag, summary = sent[1]
    assert flag is is_ignorable(exc)
    assert "[rank 3]" in summary and describe_error(exc) in summary


@pytest.mark.parametrize("exc, ignore_error", [
    (RuntimeError("worker boom"), False),
    (ValueError("worker boom"), False),
    (ValueError("worker boom"), True),
    (SystemExit(3), True),
], ids=["runtime-not-ignored", "value", "value-ignore_error", "exit-ignore_error"])
def test_c18_worker_reports_unignored_gradient_failure(tmp_path, monkeypatch, capsys,
                                                       exc, ignore_error):
    """C18(b)4, (c)1: a worker whose gradient() failure will not be ignored
    (ignore_error = false, or an exception other than RuntimeError) reports
    it on stderr, as for evaluate(): its traceback, and an ERROR line tagged
    with its rank that names gradient() and carries the exception's
    "Type: message". No ERROR line names evaluate(). Fails if the worker's
    report is limited to evaluate() (a failed gradient would then leave no
    trace of the worker's own exception) or reuses the evaluate() wording."""
    _, _, r = _make_runner(tmp_path, grad=_raising(exc),
                           runner={"ignore_error": ignore_error})
    _c18_group(monkeypatch, _C18Solcomm(1, [_C18_OK]), global_rank=3)
    r.serve_gradient(_C18_X.copy(), (1, 0))
    err = capsys.readouterr().err
    err_lines = [L for L in err.splitlines() if "ERROR" in L]
    assert any("[rank 3]" in L and "gradient()" in L and describe_error(exc) in L
               for L in err_lines), err
    for line in err_lines:
        assert "evaluate()" not in line, line
    assert "Traceback" in err


def test_c18_serve_still_evaluates(tmp_path, monkeypatch):
    """C18(d): Runner.serve() on a worker is unchanged: it calls evaluate()
    (not gradient()) with xp and args and joins one exchange. Fails if
    serve() is redirected to the gradient path."""
    _, solver, r = _make_runner(tmp_path)
    comm = _c18_group(monkeypatch, _C18Solcomm(1, [_C18_OK]), global_rank=3)
    r.serve(_C18_X.copy(), (5, 0))
    assert len(solver.eval_calls) == 1 and solver.grad_calls == []
    assert solver.eval_calls[0][1] == (5, 0)
    assert comm.kinds() == ["allgather"]


# ---------------------------------------------------------------------------
# C18(c)2 the worker loop of AlgorithmBase.main()


def _c18_worker_alg(tmp_path, monkeypatch, incoming, grad=None, others=(None,)):
    """A min_search Algorithm built as usual, then turned into a solver
    worker (solrank 1 of a group of 2) whose control messages come from
    ``incoming``; Abort calls are recorded instead of killing the process."""
    alg, solver, r = _make_alg(tmp_path, {"method": "L-BFGS-B"},
                               fn=_quad_fn(_C)[0], grad=grad or _quad_fn(_C)[1])
    comm = _c18_group(monkeypatch, _C18Solcomm(1, others, incoming), global_rank=3)
    monkeypatch.setattr(odatse.mpi, "run_on_algorithm", lambda: False)
    aborted = _c18_abort_recorder(monkeypatch)
    return alg, solver, comm, aborted


def test_c18_worker_loop_serves_gradient_and_evaluate(tmp_path, monkeypatch):
    """C18(c)2: the worker loop of AlgorithmBase.main(), on MSG_GRADIENT,
    receives xp (solver.dimension elements) and args, serves the gradient
    (gradient() with exactly those values, one exchange) and waits for the
    next signal; MSG_EVALUATE still evaluates and MSG_FINISHED still ends
    the loop (main() returns None). Fails if MSG_GRADIENT is an unknown
    signal (the worker aborts the job), if the worker evaluates instead, or
    if it reads the messages out of order."""
    G, E, F = int(_c18_msg_gradient()), int(odatse.mpi.MSG_EVALUATE), int(odatse.mpi.MSG_FINISHED)
    x1, x2, x3 = np.array([0.5, 0.25]), np.array([-1.0, 2.0]), np.array([1.5, -0.5])
    incoming = [np.array([G]), x1, (1, 0),
                np.array([E]), x2, (2, 0),
                np.array([G]), x3, (3, 1),
                np.array([F])]
    alg, solver, comm, aborted = _c18_worker_alg(tmp_path, monkeypatch, incoming)
    assert alg.main() is None
    assert aborted == []
    assert comm.incoming == []
    assert [a for _, a in solver.grad_calls] == [(1, 0), (3, 1)]
    np.testing.assert_array_equal(solver.grad_calls[0][0], x1)
    np.testing.assert_array_equal(solver.grad_calls[1][0], x3)
    assert [a for _, a in solver.eval_calls] == [(2, 0)]
    np.testing.assert_array_equal(solver.eval_calls[0][0], x2)
    assert comm.kinds() == (["Bcast", "Bcast", "bcast", "allgather"] * 3 + ["Bcast"])


@pytest.mark.parametrize("exc", [RuntimeError("worker boom"), ValueError("worker boom"), SystemExit(3)],
                         ids=["runtime", "value", "exit"])
def test_c18_worker_loop_survives_gradient_failure(tmp_path, monkeypatch, exc):
    """C18(c)1-2: a gradient() that raises on the worker (RuntimeError,
    ValueError, even SystemExit) does not end the worker loop or abort the
    job: the failure is reported in the exchange and the loop serves the
    next request, until MSG_FINISHED. Fails if serve_gradient lets the
    exception out (the worker's last-resort handler would Abort the job,
    or the worker would exit and leave the controller hanging)."""
    G, F = int(_c18_msg_gradient()), int(odatse.mpi.MSG_FINISHED)
    calls = []

    def grad(x):
        calls.append(np.array(x, copy=True))
        if len(calls) == 1:
            raise exc
        return _quad_fn(_C)[1](x)

    incoming = [np.array([G]), np.array([0.5, 0.25]), (1, 0),
                np.array([G]), np.array([-1.0, 2.0]), (2, 0),
                np.array([F])]
    alg, solver, comm, aborted = _c18_worker_alg(tmp_path, monkeypatch, incoming, grad=grad)
    assert alg.main() is None
    assert aborted == []
    assert len(calls) == 2
    sent = [v for k, v in comm.log if k == "allgather"]
    assert len(sent) == 2
    assert sent[0][1] is not None and sent[1][1] is None


def test_c18_worker_loop_abort_after_gradient(tmp_path, monkeypatch):
    """C18(c)2: MSG_ABORT after a gradient request still ends the worker
    with sys.exit(0) and without MPI_Abort, as before. Fails if serving a
    gradient changes how the loop reacts to the following signals."""
    G, A = int(_c18_msg_gradient()), int(odatse.mpi.MSG_ABORT)
    incoming = [np.array([G]), np.array([0.5, 0.25]), (1, 0), np.array([A])]
    alg, solver, comm, aborted = _c18_worker_alg(tmp_path, monkeypatch, incoming)
    with pytest.raises(SystemExit) as ei:
        alg.main()
    assert ei.value.code in (0, None)
    assert aborted == []
    assert len(solver.grad_calls) == 1


# ---------------------------------------------------------------------------
# C18 end to end on the controller: min_search in a fake group


def test_c18_min_search_controller_requests_gradients(tmp_path, monkeypatch):
    """C18(a) through min_search: on the controller of a fake group of 2
    (workers always succeed), use_gradient = true with L-BFGS-B converges,
    and every gradient() call of the controller was preceded by an
    MSG_GRADIENT request carrying the same x and args (likewise
    MSG_EVALUATE for evaluate()), in the same order; the run ends with
    MSG_FINISHED. Fails if min_search's gradients bypass the control
    protocol (the real workers would never compute their share)."""
    comm = _c18_group(monkeypatch, _C18Solcomm(0, [_C18_OK]))
    alg, solver, _ = _make_alg(tmp_path, {"method": "L-BFGS-B", "use_gradient": True},
                               fn=_quad_fn(_C)[0], grad=_quad_fn(_C)[1],
                               initial_list=_same_start_for_all_walkers((2.0, 2.0)))
    alg.main()
    np.testing.assert_allclose(alg.xopt, _C, atol=1e-4)

    reqs = _c18_requests(comm.log)
    G, E, F = int(_c18_msg_gradient()), int(odatse.mpi.MSG_EVALUATE), int(odatse.mpi.MSG_FINISHED)
    grads = [(xp, a) for s, xp, a in reqs if s == G]
    evals = [(xp, a) for s, xp, a in reqs if s == E]
    assert len(grads) == len(solver.grad_calls) >= 2
    assert len(evals) == len(solver.eval_calls)
    for (xp, a), (x, b) in zip(grads, solver.grad_calls):
        np.testing.assert_array_equal(xp, x)
        assert a == b
    for (xp, a), (x, b) in zip(evals, solver.eval_calls):
        np.testing.assert_array_equal(xp, x)
        assert a == b
    assert comm.kinds().count("allgather") == len(grads) + len(evals)
    assert reqs[-1][0] == F
