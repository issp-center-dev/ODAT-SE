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


def test_continuous_pbc_wrap():
    """Test that PBC dimensions are wrapped into [xmin, xmax)."""
    from odatse.algorithm.state import ContinuousState, ContinuousStateSpace

    domain = odatse.domain.Region(param={
        "min_list": [0.0, -1.0],
        "max_list": [1.0, 1.0],
    })
    # First dimension PBC, second not.
    info = {
        "step_list": [0.1, 0.1],
        "pbc_list": [True, False],
    }
    rng = np.random.RandomState(42)
    lim = odatse.util.limitation.Unlimited()
    sp = ContinuousStateSpace(domain, info, rng, lim)

    # Single point: out of range on PBC dim 0
    x = np.array([[1.5, 0.0]])   # 1.5 should wrap to 0.5
    out = sp._wrap_pbc(x)
    np.testing.assert_allclose(out[0, 0], 0.5)
    assert out[0, 1] == 0.0

    # Negative on PBC dim 0
    x = np.array([[-0.3, 0.0]])  # -0.3 -> 0.7
    out = sp._wrap_pbc(x)
    np.testing.assert_allclose(out[0, 0], 0.7)
    assert out[0, 1] == 0.0

    # Non-PBC dim unchanged (no wrap)
    x = np.array([[0.5, 2.0]])
    out = sp._wrap_pbc(x)
    assert out[0, 0] == 0.5
    assert out[0, 1] == 2.0


def test_continuous_pbc_list_length_error():
    """Test that pbc_list length must match dimension."""
    from odatse.algorithm.state import ContinuousStateSpace

    domain = odatse.domain.Region(param={
        "min_list": [0.0, 0.0],
        "max_list": [1.0, 1.0],
    })
    info = {
        "step_list": [0.1, 0.1],
        "pbc_list": [True],  # length 1 != 2
    }
    rng = np.random.RandomState(42)
    lim = odatse.util.limitation.Unlimited()
    with pytest.raises(ValueError, match="pbc_list length must match"):
        ContinuousStateSpace(domain, info, rng, lim)
