"""Registered-solver fixtures for tests/unit/test_api_registry.py (not part of ODAT-SE)."""

import os

import numpy as np


def shifted_sphere(x, center=0.0, scale=1.0):
    """Shifted sphere sum((x - center)^2) * scale."""
    return float(scale * np.sum((np.asarray(x) - center) ** 2))


shifted_sphere.odatse_description = "sphere centred at `center`"


def needs_target(x, *, target):
    return float(np.sum((np.asarray(x) - np.asarray(target)) ** 2))


def plane2(x):
    return float(x[0] + 2 * x[1])


plane2.odatse_dimension = 2


def any_kwargs(x, **kw):
    return float(np.sum(np.asarray(x) ** 2) + len(kw))


def raises(x):
    raise ValueError("deliberate failure")


def raises_at_positive(x):
    if x[0] > 0:
        raise RuntimeError("positive x not supported")
    return float(x[0] ** 2)


def returns_nan(x):
    return float("nan")


def returns_text(x):
    return "not a number"


def returns_inf_right(x):
    return float("inf") if x[0] > 0 else float(x[0] ** 2)


def chdir_away(x):
    os.chdir("/")
    return float(np.sum(np.asarray(x) ** 2))


def mutates(x, values=(1.0,)):
    x[0] = 1e9  # must not affect the caller's vector
    if isinstance(values, list):
        values.append(1.0)  # must not leak into the next evaluation
        return float(len(values))
    return 0.0


def bad_dimension_attr(x):
    return 0.0


bad_dimension_attr.odatse_dimension = "2"


def positional_only(x, y, /):
    return 0.0


not_callable = 3


def kwargs_only(x, **kw):
    return float(np.sum(np.asarray(x) ** 2))


def vector_positional_only(x, /, **kw):
    return float(np.sum(np.asarray(x) ** 2) + kw.get("x", 0.0))


def path_named_params(x, output_dir=".", colormap=0, checkpoint_file=""):
    return float(np.sum(np.asarray(x) ** 2))


class _BadFloat:
    def __float__(self):
        raise RuntimeError("cannot convert")


def returns_bad_float(x):
    return _BadFloat()


class _Unprintable:
    def __float__(self):
        raise RuntimeError("cannot convert")

    def __repr__(self):
        raise ValueError("no repr")

    __str__ = __repr__


def returns_unprintable(x):
    return _Unprintable()


class _UnprintableError(Exception):
    def __str__(self):
        raise ValueError("no str")


def raises_unprintable(x):
    raise _UnprintableError()


def env_offset(x, var="ODATSE_TEST_OFFSET"):
    """(x1 - $var)^2; fails when the variable did not reach the run."""
    import os

    return float((x[0] - float(os.environ[var])) ** 2)


def env_has(x, var):
    """1.0 when the environment variable is set in the run, else 0.0."""
    import os

    return 1.0 if var in os.environ else 0.0
