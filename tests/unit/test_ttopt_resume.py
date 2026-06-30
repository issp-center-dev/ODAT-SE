import os
import sys
from pathlib import Path

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse.mpi as mpi
from odatse.algorithm.ttopt import Algorithm


def _bare():
    """A TTOpt Algorithm instance with __init__ bypassed, with just enough
    attributes set for the checkpoint helpers under test."""
    alg = Algorithm.__new__(Algorithm)
    alg.info = {}
    alg.rng = np.random.RandomState(0)
    return alg


def _snapshot(n_q_dims, rng_state):
    """A minimal checkpoint snapshot compatible with base + TTOpt _apply_state."""
    return {
        # base fields
        "algsize": mpi.algsize(),
        "algrank": mpi.algrank(),
        "rng": rng_state,
        "timer": {"init": {}, "prepare": {}, "run": {}, "post": {}},
        "info": {},
        # TTOpt _checkpoint_attrs
        "n_q_dims": n_q_dims,
        "f_eval_count": 7,
        "f_eval_count_history": [7],
        "xopt": np.array([1.0, 2.0]),
        "fopt": 0.5,
        "xopt_history": [],
        "fopt_history": [],
        "poi": [None],
        "tt_ranks": np.array([1, 1]),
        "cache": {},
        "cache_hits": 3,
    }


def test_apply_state_restores_rng_and_fields():
    alg = _bare()
    alg.n_q_dims = 1

    ref = np.random.RandomState(12345)
    data = _snapshot(n_q_dims=1, rng_state=ref.get_state())

    alg._apply_state(data, restore_rng=True)

    # RNG was restored: draws match a generator seeded the same way
    np.testing.assert_array_equal(alg.rng.rand(4), ref.rand(4))
    # checkpoint fields were applied
    assert alg.f_eval_count == 7
    assert alg.cache_hits == 3


def test_apply_state_skips_rng_when_disabled():
    alg = _bare()
    alg.n_q_dims = 1
    alg.rng = np.random.RandomState(999)
    kept = alg.rng.get_state()[1].copy()

    data = _snapshot(n_q_dims=1, rng_state=np.random.RandomState(12345).get_state())
    alg._apply_state(data, restore_rng=False)

    # RNG untouched
    assert np.array_equal(alg.rng.get_state()[1], kept)
    # but the other fields are still applied
    assert alg.f_eval_count == 7


def test_apply_state_raises_on_nqdims_mismatch():
    """The structural-consistency guard must fire when the checkpoint was made
    with a different p_points/q_points (i.e. a different n_q_dims)."""
    alg = _bare()
    alg.n_q_dims = 3  # current config

    data = _snapshot(n_q_dims=2, rng_state=np.random.RandomState(0).get_state())
    with pytest.raises(RuntimeError, match="n_q_dims mismatch"):
        alg._apply_state(data)


def test_load_state_defers_apply(tmp_path):
    """_load_state must read and stash the snapshot without applying it, so the
    later _setup_structure()/_init_counters() in _prepare() cannot clobber the
    restored state. (Single load: the snapshot is read once.)"""
    alg = _bare()
    fn = str(tmp_path / "status.pickle")
    snap = _snapshot(n_q_dims=1, rng_state=np.random.RandomState(0).get_state())
    alg._save_data(snap, filename=fn)

    alg._load_state(fn, restore_rng=True)

    # stashed, not applied
    assert alg._resume_data is not None
    assert alg._resume_restore_rng is True
    assert not hasattr(alg, "f_eval_count")
