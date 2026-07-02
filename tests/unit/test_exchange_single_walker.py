import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse.mpi as mpi
from odatse.algorithm.exchange import Algorithm


def _bare():
    return Algorithm.__new__(Algorithm)


def test_single_walker_requires_one_replica_per_rank():
    """The single-walker exchange identifies each replica with one MPI rank;
    it must assert nreplica == algsize before touching the communicator.
    The guard is the first statement, so it fires before any MPI call."""
    alg = _bare()
    alg.nwalkers = 1
    alg.nreplica = mpi.algsize() + 3  # deliberately inconsistent
    alg.Tindex = np.array([0])

    with pytest.raises(AssertionError, match="one replica per process"):
        # name-mangled private method
        alg._Algorithm__exchange_single_walker(True)


def test_single_walker_invariant_holds_serially():
    """With nreplica == algsize the guard passes and the method runs. Serially
    (algsize == 1) the exchange body is a no-op but still exercises the
    Barrier/Bcast collectives, so this needs a real communicator."""
    if not mpi.enabled():
        pytest.skip("requires an MPI communicator")
    if mpi.algsize() != 1:
        pytest.skip("serial-only no-op scenario")

    alg = _bare()
    alg.nwalkers = 1
    alg.nreplica = mpi.algsize()           # == 1, the valid single-rank case
    alg.Tindex = np.array([0])
    alg.T2rep = np.arange(alg.nreplica)
    alg.fx = np.array([1.0])
    alg.betas = np.array([1.0])
    alg.rng = np.random.RandomState(0)

    # must not raise
    alg._Algorithm__exchange_single_walker(True)
    assert alg.T2rep[0] == 0
