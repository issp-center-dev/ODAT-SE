import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

import odatse
import odatse.mpi
from odatse.algorithm._algorithm import AlgorithmBase


class _Info:
    """Minimal stand-in for odatse.Info with just the algorithm section."""
    def __init__(self, algorithm):
        self.algorithm = algorithm


class _DummyAlgorithm(AlgorithmBase):
    """Concrete subclass that skips the heavy AlgorithmBase.__init__ so the
    RNG-seeding logic can be exercised in isolation."""
    def __init__(self):
        pass

    def _initialize(self):
        pass

    def _prepare(self):
        pass

    def _run(self):
        pass

    def _post(self):
        return {}


def _seed_rng(info):
    alg = _DummyAlgorithm()
    # __init_rng is name-mangled; call it directly on the instance.
    alg._AlgorithmBase__init_rng(info)
    return alg.rng


def test_seed_uses_algrank_not_global_rank(monkeypatch):
    """The per-process seed offset must be derived from the algorithm-layer
    rank (algrank), not the global MPI rank, so that the seed maps to the
    replica identity regardless of solver parallelism."""
    # Make the two ranks differ, as happens when nsolve > 1.
    monkeypatch.setattr(odatse.mpi, "algrank", lambda: 3)
    monkeypatch.setattr(odatse.mpi, "rank", lambda: 7)

    seed, seed_delta = 100, 10
    rng = _seed_rng(_Info({"seed": seed, "seed_delta": seed_delta}))

    expected = np.random.RandomState(seed + 3 * seed_delta)
    wrong = np.random.RandomState(seed + 7 * seed_delta)

    # Same first draws as the algrank-seeded generator ...
    np.testing.assert_array_equal(rng.rand(5),
                                  expected.rand(5))
    # ... and different from the (buggy) global-rank-seeded generator.
    rng2 = _seed_rng(_Info({"seed": seed, "seed_delta": seed_delta}))
    assert not np.array_equal(rng2.rand(5), wrong.rand(5))


def test_seed_none_gives_unseeded_rng(monkeypatch):
    """With no seed, a non-deterministic RandomState is created (no crash)."""
    monkeypatch.setattr(odatse.mpi, "algrank", lambda: 0)
    rng = _seed_rng(_Info({}))
    assert isinstance(rng, np.random.RandomState)
