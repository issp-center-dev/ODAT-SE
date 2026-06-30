import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

import odatse
from odatse.algorithm.pamc import Algorithm


def _find_scheduling(info_pamc):
    # Build a bare Algorithm instance, bypassing the heavy __init__, to test
    # the scheduling helper in isolation.
    alg = Algorithm.__new__(Algorithm)
    numT = alg._find_scheduling(info_pamc)
    return numT, alg.numsteps_for_T


@pytest.mark.parametrize("numsteps,numT", [
    (10, 4),   # remainder 2
    (9, 4),    # remainder 1  (the case the off-by-one dropped entirely)
    (8, 4),    # remainder 0
    (100, 7),  # remainder 2
    (13, 5),   # remainder 3
])
def test_steps_sum_to_numsteps(numsteps, numT):
    """When numsteps and Tnum are given, the per-temperature step counts must
    sum exactly to numsteps (the remainder is spread over the first temps)."""
    nT, steps = _find_scheduling({"numsteps": numsteps, "Tnum": numT})
    assert nT == numT
    assert len(steps) == numT
    assert int(np.sum(steps)) == numsteps
    # the remainder goes to the leading temperatures, so counts are
    # non-increasing and differ by at most one
    assert steps.max() - steps.min() <= 1
    assert np.all(np.diff(steps) <= 0)


def test_remainder_one_is_not_dropped():
    """Regression: remainder of 1 used to be lost (slice [0:rem-1] == [0:0])."""
    _, steps = _find_scheduling({"numsteps": 9, "Tnum": 4})
    assert list(steps) == [3, 2, 2, 2]


def test_annealing_branch_unaffected():
    """numsteps<=0 path just replicates numsteps_annealing per temperature."""
    nT, steps = _find_scheduling({"numsteps_annealing": 5, "Tnum": 3})
    assert nT == 3
    assert list(steps) == [5, 5, 5]
