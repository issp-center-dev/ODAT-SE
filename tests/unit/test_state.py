import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

import odatse

def test_gather_discrete():
    from odatse.algorithm.state import DiscreteState, DiscreteStateSpace

    mpisize = odatse.mpi.size()
    mpirank = odatse.mpi.rank()
    mpicomm = odatse.mpi.comm()
    
    domain = odatse.domain.MeshGrid(param={
        "min_list": [-1.0, -1.0],
        "max_list": [1.0, 1.0],
        "num_list": [21, 21],
    })
    info = {"radius": 0.11}
    rng = np.random.RandomState(1)
    sp = DiscreteStateSpace(domain, info, rng)

    assert sp.nnodes == 441

    nwalker = mpirank + 3
    inode = [(i % sp.nnodes) for i in range(nwalker)]
    state = DiscreteState(np.array(inode), sp.node_coordinates[inode, :])
    
    nreps = mpicomm.allgather(nwalker)
    nrep_total = np.sum(nreps)

    assert nrep_total == mpisize*(mpisize-1)/2+mpisize*3

    st = sp.gather(state)

    assert nrep_total == st.inode.shape[0]

    inodes_ref = []
    for k in range(mpisize):
        inodes_ref.extend([(i % sp.nnodes) for i in range(k + 3)])

    assert np.all(st.inode == inodes_ref)

def test_gather_continuous():
    from odatse.algorithm.state import ContinuousState, ContinuousStateSpace

    mpisize = odatse.mpi.size()
    mpirank = odatse.mpi.rank()
    mpicomm = odatse.mpi.comm()
    
    domain = odatse.domain.Region(param={
        "min_list": [-1.0, -1.0],
        "max_list": [1.0, 1.0],
    })
    info = {
        "step_list": [0.1, 0.1],
        "radius": 0.11,
    }
    rng = np.random.RandomState(mpirank+1)
    lim = odatse.util.limitation.Unlimited()
    sp = ContinuousStateSpace(domain, info, rng, lim)

    nwalker = mpirank + 3
    state = sp.initialize(nwalker)

    nreps = mpicomm.allgather(nwalker)
    displ = np.cumsum(nreps) - nreps

    st = sp.gather(state)

    nrep_total = np.sum(nreps)
    assert st.x.shape[0] == nrep_total

    if mpirank == 0:
        for k in range(1, mpisize):
            v = mpicomm.recv(source=k)
            print(v.x)
            assert np.all(st.x[displ[k]:displ[k]+nreps[k], :] == v.x)
    else:
        mpicomm.send(state, dest=0)

    #assert False
