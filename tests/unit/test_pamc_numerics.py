import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

from odatse.algorithm.pamc import Algorithm


def _bare():
    return Algorithm.__new__(Algorithm)


def test_participation_ratio_normal():
    alg = _bare()
    alg.logweights = np.zeros(4)  # equal weights
    # (sum w)^2 / sum w^2 = 16 / 4 = 4
    assert alg._calc_participation_ratio() == pytest.approx(4.0)


def test_participation_ratio_degenerate_weights_not_nan():
    """All-(-inf) log-weights used to give nan (0/0); must be a finite 0.0."""
    alg = _bare()
    alg.logweights = np.full(3, -np.inf)
    pr = alg._calc_participation_ratio()
    assert np.isfinite(pr)
    assert pr == 0.0


def test_gather_information_zero_trial_acceptance_ratio_not_nan():
    """A temperature with zero trials must yield acceptance ratio 0.0, not a
    0/0 nan written to the output files."""
    alg = _bare()
    alg.nwalkers = 2
    alg.fx_from_reset = np.zeros((2, 2))
    alg.walker_ancestors = np.array([0, 1])
    # row 0: 0 accepted / 0 trials -> would be 0/0; row 1: 3/5
    alg.naccepted_from_reset = np.array([[0, 0], [3, 5]])
    alg.betas = np.array([1.0, 2.0])
    alg.Tindex = 1

    res = alg._gather_information(numT=2)

    ar = res["acceptance ratio"]
    assert np.all(np.isfinite(ar))
    assert ar[0] == 0.0
    assert ar[1] == pytest.approx(0.6)
