# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Check the benchmark functions in odatse.solver.analytical against their
standard textbook definitions, evaluated directly with the math module."""
import math

import numpy as np
import pytest

import odatse.solver.analytical as A


# ---------------------------------------------------------------------------
# griewank


def _griewank_direct(x):
    prod = 1.0
    for i, v in enumerate(x, start=1):
        prod *= math.cos(v / math.sqrt(i))
    return 1.0 + sum(v * v for v in x) / 4000.0 - prod


@pytest.mark.parametrize("d", [2, 10])
def test_griewank_zero_at_origin(d):
    assert A.griewank(np.zeros(d)) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "x",
    [
        [1.0, 2.0],
        [math.pi, 0.0],
        [-3.5, 0.7, 12.0, -0.25, 100.0],
        [0.3 * (k + 1) for k in range(10)],
    ],
)
def test_griewank_matches_standard_formula(x):
    xs = np.array(x, dtype=float)
    assert A.griewank(xs) == pytest.approx(_griewank_direct(x), rel=1e-12, abs=1e-14)


def test_griewank_origin_is_below_pi_point():
    # With the historical "+" sign f(0) = 2 and (pi, 0) was the minimum;
    # with the standard "-" sign the origin is the global minimum.
    assert A.griewank(np.zeros(2)) < A.griewank(np.array([math.pi, 0.0]))


# ---------------------------------------------------------------------------
# other functions: direct evaluation of the standard definitions


def _ackley(x):
    d = len(x)
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(sum(v * v for v in x) / d))
        - math.exp(sum(math.cos(2 * math.pi * v) for v in x) / d)
        + 20.0
        + math.e
    )


def _alpine(x):
    return sum(abs(v * math.sin(v) + 0.1 * v) for v in x)


def _exponential(x):
    return -math.exp(-0.5 * sum(v * v for v in x))


def _michalewicz(x, m=10):
    return -sum(
        math.sin(v) * math.sin(i * v * v / math.pi) ** (2 * m)
        for i, v in enumerate(x, start=1)
    )


def _qing(x):
    return sum((v * v - i) ** 2 for i, v in enumerate(x, start=1))


def _rastrigin(x):
    return 10.0 * len(x) + sum(v * v - 10.0 * math.cos(2 * math.pi * v) for v in x)


def _rosenbrock(x):
    return sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def _schaffer(x):
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i] ** 2 + x[i + 1] ** 2
        s += 0.5 + (math.sin(a) ** 2 - 0.5) / (1.0 + 0.001 * a) ** 2
    return s


def _schwefel(x):
    return 418.9829 * len(x) - sum(v * math.sin(math.sqrt(abs(v))) for v in x)


def _himmelblau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2


_REFERENCE = {
    "ackley": _ackley,
    "alpine": _alpine,
    "exponential": _exponential,
    "griewank": _griewank_direct,
    "michalewicz": _michalewicz,
    "qing": _qing,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "schaffer": _schaffer,
    "schwefel": _schwefel,
}


@pytest.mark.parametrize("name", sorted(_REFERENCE))
@pytest.mark.parametrize("d", [2, 5, 10])
def test_matches_standard_formula(name, d):
    rng = np.random.default_rng(12345)
    f = getattr(A, name)
    ref = _REFERENCE[name]
    for _ in range(10):
        xs = rng.uniform(-5.0, 5.0, d)
        assert f(xs) == pytest.approx(ref(list(xs)), rel=1e-10, abs=1e-12), name


def test_himmelblau_matches_standard_formula():
    rng = np.random.default_rng(6789)
    for _ in range(10):
        xs = rng.uniform(-5.0, 5.0, 2)
        assert A.himmelblau(xs) == pytest.approx(_himmelblau(list(xs)), rel=1e-12)


# known global minima (values as documented in the docstrings)
@pytest.mark.parametrize(
    "name, x, fmin, tol",
    [
        ("ackley", np.zeros(3), 0.0, 1e-14),
        ("alpine", np.zeros(3), 0.0, 1e-14),
        ("exponential", np.zeros(3), -1.0, 1e-14),
        ("griewank", np.zeros(3), 0.0, 1e-14),
        ("qing", np.sqrt(np.arange(1, 6)), 0.0, 1e-12),
        ("rastrigin", np.zeros(3), 0.0, 1e-14),
        ("rosenbrock", np.ones(4), 0.0, 1e-14),
        ("schaffer", np.zeros(3), 0.0, 1e-14),
        # 418.9829 is the conventional rounded constant, so f is only ~1e-5
        # per dimension at the conventional minimum 420.9687
        ("schwefel", np.full(2, 420.9687), 0.0, 1e-4),
        ("himmelblau", np.array([3.0, 2.0]), 0.0, 1e-14),
        ("michalewicz", np.array([2.20290552, 1.57079633]), -1.8013, 1e-4),
    ],
)
def test_known_global_minimum(name, x, fmin, tol):
    assert getattr(A, name)(x) == pytest.approx(fmin, abs=tol)
