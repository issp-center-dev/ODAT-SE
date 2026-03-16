import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

import odatse
from odatse import mpi

def test_gather_discrete():
    from odatse.algorithm.state import DiscreteState, DiscreteStateSpace

    algsize = mpi.algsize()
    algrank = mpi.algrank()
    algcomm = mpi.algcomm()
    
    domain = odatse.domain.MeshGrid(param={
        "min_list": [-1.0, -1.0],
        "max_list": [1.0, 1.0],
        "num_list": [21, 21],
    })
    info = {"radius": 0.11}
    rng = np.random.RandomState(1)
    sp = DiscreteStateSpace(domain, info, rng)

    assert sp.nnodes == 441

    nwalker = algrank + 3
    inode = [(i % sp.nnodes) for i in range(nwalker)]
    state = DiscreteState(np.array(inode), sp.node_coordinates[inode, :])
    
    nreps = algcomm.allgather(nwalker)
    nrep_total = np.sum(nreps)

    assert nrep_total == algsize*(algsize-1)/2+algsize*3

    st = sp.gather(state)

    assert nrep_total == st.inode.shape[0]

    inodes_ref = []
    for k in range(algsize):
        inodes_ref.extend([(i % sp.nnodes) for i in range(k + 3)])

    assert np.all(st.inode == inodes_ref)

def test_gather_continuous():
    from odatse.algorithm.state import ContinuousState, ContinuousStateSpace

    algsize = mpi.algsize()
    algrank = mpi.algrank()
    algcomm = mpi.algcomm()
    
    domain = odatse.domain.Region(param={
        "min_list": [-1.0, -1.0],
        "max_list": [1.0, 1.0],
    })
    info = {
        "step_list": [0.1, 0.1],
        "radius": 0.11,
    }
    rng = np.random.RandomState(algrank+1)
    lim = odatse.util.limitation.Unlimited()
    sp = ContinuousStateSpace(domain, info, rng, lim)

    nwalker = algrank + 3
    state = sp.initialize(nwalker)

    nreps = algcomm.allgather(nwalker)
    displ = np.cumsum(nreps) - nreps

    st = sp.gather(state)

    nrep_total = np.sum(nreps)
    assert st.x.shape[0] == nrep_total

    if algrank == 0:
        for k in range(1, algsize):
            v = algcomm.recv(source=k)
            print(v.x)
            assert np.all(st.x[displ[k]:displ[k]+nreps[k], :] == v.x)
    else:
        algcomm.send(state, dest=0)

    #assert False
