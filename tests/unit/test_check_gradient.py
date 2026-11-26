# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Behavior contract 8 of SolverBase.check_gradient(x, args=(0, 0), step=None),
items K1-K8: it returns (g, d), the solver's gradient and the central
difference of evaluate(), without judging them (K1); rejects malformed x and
step before calling the solver (K2); uses eps**(1/3) * max(1, |x_i|), a scalar
or a per-component step (K3); divides by the difference of the two computed
coordinates, skips components whose two points coincide and passes non-finite
values through silently (K4); calls the solver in a fixed order with fresh
float arrays and is immune to a solver that overwrites them (K5); lets the
solver's exceptions through untouched and ignores supports_gradient (K6);
leaves the solver's state and timer alone (K7); and agrees with the gradients
of the built-in analytical and function solvers (K8)."""
import warnings
from collections import namedtuple

import numpy as np
import pytest

import odatse
import odatse.solver
import odatse.solver.analytical
import odatse.solver.function

EPS = np.finfo(float).eps

# ---------------------------------------------------------------------------
# helpers

_Call = namedtuple("_Call", "name received point args")


def _fn(p):
    return np.float64(np.sum((np.arange(p.size) + 1) * np.sin(p)))


def _grad(p):
    return (np.arange(p.size) + 1) * np.cos(p)


class _NoGradient(odatse.solver.SolverBase):
    """Solver recording every call: its name, the object received, a copy of
    the point taken on entry, and args. gradient() is the base class's.

    ``mutate`` overwrites the received array before returning; ``fail_at``
    makes the call with that index (over evaluate and gradient together)
    raise ``exc``.
    """

    def __init__(self, info, fn=_fn, grad=_grad, mutate=False, fail_at=None, exc=None):
        super().__init__(info)
        self.fn, self.grad, self.mutate = fn, grad, mutate
        self.fail_at, self.exc = fail_at, exc
        self.calls = []

    def _enter(self, name, x, args):
        self.calls.append(_Call(name, x, np.array(x, copy=True), args))
        if self.fail_at == len(self.calls) - 1:
            raise self.exc
        return self.calls[-1].point

    def evaluate(self, x, args=()):
        value = self.fn(self._enter("evaluate", x, args))
        if self.mutate:
            x[:] = 0.0
        return value


class _Recorder(_NoGradient):
    """_NoGradient plus a recorded gradient(); supports_gradient stays False."""

    def gradient(self, x, args=()):
        value = self.grad(self._enter("gradient", x, args))
        if self.mutate:
            x[:] = 0.0
        return value


def _info(tmp_path, dimension=2, **solver):
    return odatse.Info({
        "base": {"dimension": dimension, "output_dir": str(tmp_path / "output")},
        "algorithm": {"name": "minsearch"},
        "solver": {"name": "function", **solver},
    })


def _names(solver):
    return [c.name for c in solver.calls]


def _points(x, h):
    """Points the solver must receive, in order, for steps h (K5)."""
    x = np.asarray(x, dtype=float)
    h = np.broadcast_to(np.asarray(h, dtype=float), x.shape)
    points = [x.copy(), x.copy()]
    for i in range(x.size):
        for sign in (1.0, -1.0):
            p = x.copy()
            p[i] = x[i] + sign * h[i]
            points.append(p)
    return points


def _quotients(solver, n):
    """d rebuilt from the recorded calls: the difference of the two values
    over the difference of the two coordinates the solver was given."""
    d = np.empty(n)
    for i in range(n):
        plus, minus = solver.calls[2 + 2 * i].point, solver.calls[3 + 2 * i].point
        d[i] = (solver.fn(plus) - solver.fn(minus)) / (plus[i] - minus[i])
    return d


# ---------------------------------------------------------------------------
# K1 return value


def test_k1_returns_gradient_and_central_difference_silently(tmp_path, capfd):
    """K1: the result is a tuple (g, d) of float arrays, g being gradient(x)
    and d, of shape (n,), its central-difference estimate; nothing is printed."""
    solver = _Recorder(_info(tmp_path))
    x = np.array([0.3, -1.2, 2.0])
    result = solver.check_gradient(x)
    assert type(result) is tuple and len(result) == 2
    g, d = result
    assert isinstance(g, np.ndarray) and g.dtype == np.float64
    assert isinstance(d, np.ndarray) and d.dtype == np.float64 and d.shape == (3,)
    np.testing.assert_array_equal(g, _grad(x))
    np.testing.assert_allclose(d, _grad(x), rtol=1e-6)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("returned, shape", [
    (3.0, ()),
    ([[1.0], [2.0]], (2, 1)),
    ([1, 2], (2,)),
    (np.arange(3), (3,)),
])
def test_k1_g_keeps_the_shape_gradient_returned(tmp_path, returned, shape):
    """K1: g is gradient()'s return value as a float array in the shape it
    came in, even when that shape differs from x's; d is still (n,)."""
    solver = _Recorder(_info(tmp_path), grad=lambda p: returned)
    g, d = solver.check_gradient([0.3, -1.2])
    assert isinstance(g, np.ndarray) and g.dtype == np.float64 and g.shape == shape
    np.testing.assert_array_equal(g, np.asarray(returned, dtype=float))
    assert d.shape == (2,)


# ---------------------------------------------------------------------------
# K2 argument checks

_BAD_X = [1.0, [[1.0, 2.0]], [], np.array([1.0 + 0.0j, 2.0]), [1.0, np.inf], [np.nan, 2.0]]
_BAD_X.append([10 ** 400, 1])  # an integer that does not fit in a float is not finite
_BAD_STEP = [
    [0.1], [], [0.1, 0.1, 0.1], [[0.1, 0.1]], [[0.1], [0.1]],
    0.0, -0.1, np.inf, np.nan, 0.1 + 0.0j,
    [0.1, 0.0], [0.1, -0.1], [np.inf, 0.1], [0.1, np.nan], np.array([0.1 + 0.0j, 0.1]),
    10 ** 400,
]


@pytest.mark.parametrize("x, step", [(x, None) for x in _BAD_X]
                         + [([1.0, 2.0], step) for step in _BAD_STEP]
                         + [([1e308, 0.0], 1e308), ([0.0, -1e308], [1.0, 1e308])])
def test_k2_invalid_arguments_raise_before_any_solver_call(tmp_path, x, step):
    """K2: x that is not a non-empty, finite, real 1-d sequence, a step that
    is neither a scalar nor (n,) or holds a value that is not positive, finite
    and real, and a difference point that overflows all raise ValueError with
    neither evaluate() nor gradient() called. The checks themselves raise
    nothing else and warn about nothing, even with numpy errors and warnings
    raised."""
    solver = _Recorder(_info(tmp_path))
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            solver.check_gradient(x, step=step)
    assert solver.calls == []


@pytest.mark.parametrize("x", [[1, 2], (1.0, 2.0), np.array([1, 2])])
def test_k2_x_may_be_a_list_a_tuple_or_an_integer_array(tmp_path, x):
    """K2: any of these forms of x gives the result of the float array."""
    g_ref, d_ref = _Recorder(_info(tmp_path)).check_gradient(np.array([1.0, 2.0]))
    g, d = _Recorder(_info(tmp_path)).check_gradient(x)
    np.testing.assert_array_equal(g, g_ref)
    np.testing.assert_array_equal(d, d_ref)


# ---------------------------------------------------------------------------
# K3 step


def test_k3_default_step_scales_with_the_magnitude_of_each_component(tmp_path):
    """K3: step=None uses h_i = eps**(1/3) * max(1, |x_i|)."""
    solver = _Recorder(_info(tmp_path))
    x = np.array([0.0, 0.5, -300.0, 4.0e5])
    solver.check_gradient(x)
    h = EPS ** (1.0 / 3.0) * np.array([1.0, 1.0, 300.0, 4.0e5])
    expected = _points(x, h)
    assert len(solver.calls) == len(expected)
    for call, point in zip(solver.calls, expected):
        # not exact: eps**(1/3) may be evaluated as a cube root
        np.testing.assert_allclose(call.point, point, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("step, h", [
    (0.25, [0.25, 0.25, 0.25]),
    (1, [1.0, 1.0, 1.0]),
    (np.float64(1e-4), [1e-4, 1e-4, 1e-4]),
    ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
    ((0.1, 0.2, 0.3), [0.1, 0.2, 0.3]),
    (np.array([1, 2, 3]), [1.0, 2.0, 3.0]),
])
def test_k3_scalar_step_is_shared_and_sequence_step_is_per_component(tmp_path, step, h):
    """K3 (and K2's accepted forms of step): a scalar is every component's
    h_i, a sequence of shape (n,) gives h_i = step[i]."""
    solver = _Recorder(_info(tmp_path))
    x = [0.3, -1.2, 2.0]
    solver.check_gradient(x, step=step)
    expected = _points(x, h)
    assert len(solver.calls) == len(expected)
    for call, point in zip(solver.calls, expected):
        np.testing.assert_array_equal(call.point, point)


# ---------------------------------------------------------------------------
# K4 difference


def test_k4_divides_by_the_difference_of_the_computed_coordinates(tmp_path):
    """K4: d_i is the difference of the two values over x+_i - x-_i as
    computed in double precision, which here is not 2 * h_i."""
    x, h = np.array([1.0, 0.3]), np.array([1e-9, 1e-7])
    assert np.all((x + h) - (x - h) != 2.0 * h)  # otherwise the test shows nothing
    solver = _Recorder(_info(tmp_path))
    _, d = solver.check_gradient(x, step=h)
    np.testing.assert_array_equal(d, _quotients(solver, 2))
    np.testing.assert_allclose(d, _grad(x), rtol=1e-5)


@pytest.mark.parametrize("big", [0, 1])
def test_k4_component_whose_two_points_coincide_is_nan_and_not_evaluated(tmp_path, big):
    """K4: where x_i + h_i == x_i - h_i no evaluate() is spent on component i
    and d_i is NaN; the other component is computed as usual."""
    small = 1 - big
    x = np.array([2.0, 2.0])
    x[big] = 1e16
    solver = _Recorder(_info(tmp_path))
    _, d = solver.check_gradient(x, step=1e-3)
    assert _names(solver) == ["evaluate", "gradient", "evaluate", "evaluate"]
    plus, minus = solver.calls[2].point, solver.calls[3].point
    assert plus[small] == 2.0 + 1e-3 and minus[small] == 2.0 - 1e-3
    assert plus[big] == 1e16 and minus[big] == 1e16
    assert np.isnan(d[big])
    assert d[small] == (_fn(plus) - _fn(minus)) / (plus[small] - minus[small])


@pytest.mark.parametrize("plus, minus, expected", [
    (np.inf, 0.0, np.inf),
    (0.0, np.inf, -np.inf),
    (np.inf, np.inf, np.nan),
    (np.inf, -np.inf, np.inf),
    (np.nan, 1.0, np.nan),
    (1e308, -1e308, np.inf),  # finite values whose difference overflows
    (-1e308, 1e308, -np.inf),
])
def test_k4_non_finite_values_pass_into_d_without_error_or_warning(tmp_path, plus, minus, expected):
    """K4: inf and NaN returned by evaluate(), and an overflow in the
    difference of finite values, go into d as they are; no
    exception and no warning even with numpy errors and warnings raised."""

    def fn(p):
        if p[0] != 1.0:
            return np.float64(plus if p[0] > 1.0 else minus)
        return np.float64(p[1] ** 2)

    solver = _Recorder(_info(tmp_path), fn=fn)
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        _, d = solver.check_gradient([1.0, 2.0], step=0.5)
    np.testing.assert_array_equal(d, [expected, 4.0])


# ---------------------------------------------------------------------------
# K5 calls to the solver


@pytest.mark.parametrize("x", [[2.0], [0.3, -1.2, 2.0]])
def test_k5_calls_evaluate_gradient_then_plus_and_minus_per_component(tmp_path, x):
    """K5: evaluate(x), gradient(x), then evaluate at x + h_i e_i and
    x - h_i e_i for i = 0..n-1, i.e. 2n + 1 evaluations and one gradient,
    every call with the args object that was passed in."""
    args = (3, 7)
    solver = _Recorder(_info(tmp_path))
    solver.check_gradient(x, args, step=0.125)
    assert _names(solver) == ["evaluate", "gradient"] + ["evaluate"] * (2 * len(x))
    for call, point in zip(solver.calls, _points(x, 0.125)):
        np.testing.assert_array_equal(call.point, point)
        assert call.args is args


def test_k5_args_default_to_0_0(tmp_path):
    """K5: without args every call receives (0, 0)."""
    solver = _Recorder(_info(tmp_path))
    solver.check_gradient([0.3, -1.2])
    assert len(solver.calls) == 6
    assert all(call.args == (0, 0) for call in solver.calls)


@pytest.mark.parametrize("x", [np.array([0.3, -1.2]), np.array([1, 2]), [0.3, -1.2]])
def test_k5_solver_receives_a_fresh_float_array_on_every_call(tmp_path, x):
    """K5: each call gets its own float ndarray of shape (n,), distinct from
    the other calls' and from the caller's x."""
    solver = _Recorder(_info(tmp_path))
    solver.check_gradient(x)
    received = [call.received for call in solver.calls]
    for r in received:
        assert isinstance(r, np.ndarray) and r.dtype == np.float64 and r.shape == (2,)
        assert r is not x
    assert len({id(r) for r in received}) == len(received) == 6


def test_k5_solver_overwriting_its_input_changes_nothing(tmp_path):
    """K5: a solver that zeroes the array it receives alters neither the
    caller's x, nor the points of later calls, nor d."""
    x = np.array([0.3, -1.2, 2.0])
    clean = _Recorder(_info(tmp_path))
    g_ref, d_ref = clean.check_gradient(x.copy(), step=1e-4)
    solver = _Recorder(_info(tmp_path), mutate=True)
    g, d = solver.check_gradient(x, step=1e-4)
    np.testing.assert_array_equal(x, [0.3, -1.2, 2.0])
    assert len(solver.calls) == len(clean.calls)
    for call, ref in zip(solver.calls, clean.calls):
        np.testing.assert_array_equal(call.point, ref.point)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(g, g_ref)


def test_k5_g_is_a_copy_taken_right_after_gradient(tmp_path):
    """K5: when gradient() returns the solver's own buffer and later
    evaluate() calls overwrite it, g holds the values at return time, and
    writing to g leaves the buffer alone."""
    buffer = np.zeros(2)

    def fn(p):
        buffer[:] = -99.0
        return _fn(p)

    def grad(p):
        buffer[:] = _grad(p)
        return buffer

    solver = _Recorder(_info(tmp_path), fn=fn, grad=grad)
    x = np.array([0.3, -1.2])
    g, _ = solver.check_gradient(x)
    np.testing.assert_array_equal(g, _grad(x))
    g[:] = 5.0
    np.testing.assert_array_equal(buffer, [-99.0, -99.0])


# ---------------------------------------------------------------------------
# K6 exceptions


@pytest.mark.parametrize("fail_at", [0, 1, 2, 3, 5])
def test_k6_solver_exception_propagates_unchanged_and_stops_the_calls(tmp_path, fail_at):
    """K6: an exception raised by evaluate() (calls 0, 2, 3, 5) or gradient()
    (call 1) comes out as the same object, and no call follows it."""
    exc = KeyError("raised by the solver")
    solver = _Recorder(_info(tmp_path), fail_at=fail_at, exc=exc)
    with pytest.raises(KeyError) as excinfo:
        solver.check_gradient([0.3, -1.2])
    assert excinfo.value is exc
    assert len(solver.calls) == fail_at + 1


def test_k6_solver_without_gradient_raises_not_implemented_after_one_evaluate(tmp_path):
    """K6: with the base class's gradient() NotImplementedError propagates,
    after the single leading evaluate(x, args) of K5's order."""
    solver = _NoGradient(_info(tmp_path))
    with pytest.raises(NotImplementedError):
        solver.check_gradient([0.3, -1.2], (1, 2))
    assert _names(solver) == ["evaluate"]
    np.testing.assert_array_equal(solver.calls[0].point, [0.3, -1.2])
    assert solver.calls[0].args == (1, 2)


def test_k6_supports_gradient_false_does_not_prevent_the_check(tmp_path):
    """K6: supports_gradient is not consulted; a working gradient() is enough."""
    solver = _Recorder(_info(tmp_path))
    assert solver.supports_gradient is False
    x = np.array([0.3, -1.2])
    g, d = solver.check_gradient(x)
    np.testing.assert_array_equal(g, _grad(x))
    np.testing.assert_allclose(d, _grad(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# K7 what it stays away from


def test_k7_solver_state_is_not_restored_and_timer_is_untouched(tmp_path):
    """K7: the function solver is left at the last difference point rather
    than put back at x, and solver.timer is as it was."""
    solver = odatse.solver.function.Solver(_info(tmp_path))
    solver.work_dir.mkdir(parents=True, exist_ok=True)
    solver.set_function(lambda p: float(np.sum(p ** 2)))
    solver.set_gradient(lambda p: 2.0 * p)
    solver.check_gradient([0.3, -1.2], step=0.5)
    np.testing.assert_array_equal(solver.x, [0.3, -1.7])
    assert solver.timer == {"prepare": {}, "run": {}, "post": {}}


# ---------------------------------------------------------------------------
# K8 built-in solvers


@pytest.mark.parametrize("function_name, x", [
    ("quadratics", [0.3, -1.2, 2.0]),
    ("rosenbrock", [-1.2, 1.0, 0.7]),
    ("himmelblau", [1.0, -2.0]),
])
def test_k8_analytical_solver_gradient_matches_the_difference(tmp_path, function_name, x):
    """K8: for built-in functions at a differentiable point, g and d agree to
    a relative error of about 1e-6 with the default step."""
    info = _info(tmp_path, dimension=len(x), name="analytical", function_name=function_name)
    solver = odatse.solver.analytical.Solver(info)
    solver.work_dir.mkdir(parents=True, exist_ok=True)
    g, d = solver.check_gradient(x)
    assert g.shape == d.shape == (len(x),)
    np.testing.assert_allclose(d, g, rtol=1e-6)


@pytest.mark.parametrize("factor", [1.0, -1.0, 2.0])
def test_k8_function_solver_shows_a_wrong_gradient_as_a_mismatch(tmp_path, factor):
    """K8: with set_function()/set_gradient(), d is close to the true
    gradient whatever was set, and g is what was set: equal to d for the
    right gradient, visibly different for a negated or doubled one."""
    c = np.array([1.0, -0.5])
    true_grad = lambda p: 4.0 * (p - c) ** 3 + np.cos(p)
    solver = odatse.solver.function.Solver(_info(tmp_path))
    solver.work_dir.mkdir(parents=True, exist_ok=True)
    solver.set_function(lambda p: float(np.sum((p - c) ** 4 + np.sin(p))))
    solver.set_gradient(lambda p: factor * true_grad(p))
    x = np.array([0.3, 1.2])
    g, d = solver.check_gradient(x)
    np.testing.assert_allclose(d, true_grad(x), rtol=1e-6)
    np.testing.assert_array_equal(g, factor * true_grad(x))
    assert np.allclose(g, d, rtol=1e-6, atol=0.0) == (factor == 1.0)
