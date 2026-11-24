import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

from odatse.util.separateT import _compute_temperature_statistics


def test_constant_samples_give_finite_zero_error():
    """Near-constant f(x) makes the one-pass variance E[x^2]-E[x]^2 come out
    slightly negative (catastrophic cancellation); the std error must be a
    finite 0.0 rather than nan."""
    x = np.array([0.1])
    # 0.1 is not exactly representable, so 0.1^2*N and (sum/N)^2 differ by a
    # tiny negative amount -> sqrt would yield nan without the clamp.
    samples = [(i, 0.1, x) for i in range(3)]

    mean, err, dlogZ, acc = _compute_temperature_statistics(
        samples, thermalization_steps=0, dbeta=0.0)

    assert mean == pytest.approx(0.1)
    assert np.isfinite(err)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_varying_samples_have_positive_error():
    samples = [(0, 1.0, np.array([0.0])),
               (1, 2.0, np.array([1.0])),
               (2, 3.0, np.array([2.0]))]
    mean, err, dlogZ, acc = _compute_temperature_statistics(
        samples, thermalization_steps=0, dbeta=0.0)
    assert mean == pytest.approx(2.0)
    assert np.isfinite(err) and err > 0.0
