"""Resuming must not be allowed against a different point set.

``_index_start``/``_index_end`` are derived from the configured number of
points. They are either not checkpointed at all (MeshIterator) or overwritten
on restore (RandomIterator, ListIterator), so resuming with a different count
evaluates a set of points that matches neither the checkpoint nor the new
input. The generic "parameter changed" warning is printed but the run still
exits 0, so the mismatch is easy to miss.

The tests avoid MPI collectives in the test body: every rank executes the same
constructor calls, and a failing assertion on one rank must not desynchronise
the others.
"""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

from odatse.algorithm._iterator import MeshIterator, RandomIterator
from odatse.exception import InputError


def _rng():
    return np.random.RandomState(0)


def _drain(it):
    for _ in it:
        pass


def test_resume_rejects_a_larger_point_count():
    it = RandomIterator([-1.0], [1.0], 12, _rng())
    _drain(it)
    snap = it._save_state()

    bigger = RandomIterator([-1.0], [1.0], 24, _rng())
    with pytest.raises(InputError, match="number of search points changed"):
        bigger._restore_state(snap, mode="resume")


def test_resume_rejects_a_smaller_point_count():
    it = RandomIterator([-1.0], [1.0], 24, _rng())
    _drain(it)
    snap = it._save_state()

    smaller = RandomIterator([-1.0], [1.0], 12, _rng())
    with pytest.raises(InputError, match="number of search points changed"):
        smaller._restore_state(snap, mode="resume")


def test_resume_with_the_same_count_is_allowed():
    it = RandomIterator([-1.0], [1.0], 12, _rng())
    next(it)
    snap = it._save_state()

    same = RandomIterator([-1.0], [1.0], 12, _rng())
    same._restore_state(snap, mode="resume")
    assert same._total_points == 12


def test_continue_is_not_blocked_by_the_resume_guard():
    """--cont legitimately changes the count; the guard must not fire."""
    it = RandomIterator([-1.0], [1.0], 12, _rng())
    _drain(it)
    snap = it._save_state()

    bigger = RandomIterator([-1.0], [1.0], 24, _rng())
    bigger._restore_state(snap, mode="continue")  # must not raise
    bigger._extend(24)
    assert bigger._total_points == 24


def test_mesh_resume_rejects_a_changed_grid():
    """A different num_list is a different set of coordinates, not a
    continuation of the same scan."""
    it = MeshIterator([0.0], [1.0], [8])
    _drain(it)
    snap = it._save_state()

    bigger = MeshIterator([0.0], [1.0], [12])
    with pytest.raises(InputError, match="number of search points changed"):
        bigger._restore_state(snap, mode="resume")


def test_old_checkpoint_without_total_points_still_resumes():
    """Checkpoints written before _total_points was recorded carry no count to
    compare against, so they must keep resuming."""
    it = RandomIterator([-1.0], [1.0], 12, _rng())
    next(it)
    snap = it._save_state()
    del snap["_total_points"]

    same = RandomIterator([-1.0], [1.0], 12, _rng())
    same._restore_state(snap, mode="resume")
    # the position of the single consumed point was restored
    assert same.position() == 1
