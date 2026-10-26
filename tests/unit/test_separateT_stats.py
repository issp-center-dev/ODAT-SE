import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

from odatse.util.separateT import (
    _compute_temperature_statistics,
    calculate_statistics_from_separated_files,
)


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


def _write_result_T(dirpath, index, fxs):
    """Write ``result_T<index>.txt`` with one walker whose f(x) is *fxs*."""
    with open(dirpath / f"result_T{index}.txt", "w") as f:
        f.write("# T = ?\n")
        for step, fx in enumerate(fxs):
            f.write(f"{step} 0 {fx} {fx}\n")


@pytest.mark.parametrize("ascending", [True, False])
def test_fx_txt_lists_beta_and_integrates_towards_higher_beta(tmp_path, ascending):
    """``calculate_statistics_from_separated_files`` takes temperatures.  Whatever
    their order, fx.txt must list 1/T in ascending order of beta, and
    log(Z/Z0) at beta_i must come from the samples at beta_{i-1} weighted
    with exp(-(beta_i - beta_{i-1}) f).  With constant f per temperature the
    integral is exact: log(Z(1)/Z(0.5)) = -0.5 * f(T=2) = -0.5.

    Exchange used to pass betas as temperatures when the input was given as
    bmin/bmax (issue #133); the descending case is exactly what it passes
    now, since betas ascend."""
    # T = 2 (beta 0.5) has f == 1, T = 1 (beta 1) has f == 2
    if ascending:
        Ts = np.array([1.0, 2.0])
        _write_result_T(tmp_path, 0, [2.0] * 4)
        _write_result_T(tmp_path, 1, [1.0] * 4)
    else:
        Ts = np.array([2.0, 1.0])
        _write_result_T(tmp_path, 0, [1.0] * 4)
        _write_result_T(tmp_path, 1, [2.0] * 4)

    calculate_statistics_from_separated_files(
        Ts, output_dir=str(tmp_path), thermalization_steps=0, comm=None)

    rows = np.loadtxt(tmp_path / "fx.txt", ndmin=2)
    np.testing.assert_allclose(rows[:, 0], [0.5, 1.0])   # beta, ascending
    np.testing.assert_allclose(rows[:, 1], [1.0, 2.0])   # mean f at that beta
    np.testing.assert_allclose(rows[:, 4], [0.0, -0.5])  # log(Z/Z0)
