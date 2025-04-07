"""
Tests odatse/algorithm/gather.py

test conditions:
a) run with a single process.
b) run with multiple processes and ODATSE_USE_MPI_BUFFERED=0
   (use allgather).
c) run with multiple processes and ODATSE_USE_MPI_BUFFERED=1
   (use Allgatherv).
"""
import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
#sys.path.append(SOURCE_PATH)
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse
from odatse.algorithm.gather import gather_replica, gather_data

def run_gather_data(shape, dtype, axis):
    mpisize = odatse.mpi.size()
    mpirank = odatse.mpi.rank()
    mpicomm = odatse.mpi.comm()

    assert axis < len(shape)

    ndata = np.prod(shape)
    data = np.arange(ndata, dtype=dtype).reshape(shape) + 100 * mpirank
    gdata = gather_data(data, axis=axis)

    assert gdata.shape[axis] == mpisize * shape[axis]

    gtmp = [(np.arange(ndata, dtype=dtype).reshape(shape) + 100 * i) for i in range(mpisize)]
    gdata_ref = np.concatenate(gtmp, axis=axis)

    assert np.all(gdata == gdata_ref)


def run_gather_replica(_shape, dtype, axis):
    mpisize = odatse.mpi.size()
    mpirank = odatse.mpi.rank()
    mpicomm = odatse.mpi.comm()

    assert axis < len(_shape)

    nreps = [i+3 for i in range(mpisize)]

    shape = list(_shape)
    shape[axis] = nreps[mpirank]
    shape = tuple(shape)

    ndata = np.prod(shape)
    data = np.arange(ndata, dtype=dtype).reshape(shape) + 100 * mpirank
    gdata = gather_replica(data, axis=axis)

    assert gdata.shape[axis] == np.sum(nreps)

    gtmp = []
    for i in range(mpisize):
        sh = list(_shape)
        sh[axis] = nreps[i]
        nd = np.prod(sh)
        gtmp.append(np.arange(nd, dtype=dtype).reshape(sh) + 100 * i)
    gdata_ref = np.concatenate(gtmp, axis=axis)

    assert np.all(gdata == gdata_ref)

def test_gather_data1():
    run_gather_data((6,), int, 0)

def test_gather_data2():
    with pytest.raises(AssertionError) as e:
        run_gather_data((6,), int, 1)

def test_gather_data3():
    run_gather_data((3,4), int, 0)

def test_gather_data4():
    run_gather_data((3,4), float, 1)

def test_gather_data5():
    run_gather_data((3,4,5), np.float64, 0)

def test_gather_data6():
    run_gather_data((3,4,5), np.float64, 1)

def test_gather_replica1():
    run_gather_replica((6,), int, 0)

def test_gather_replica2():
    run_gather_replica((3,4), int, 0)

def test_gather_replica3():
    run_gather_replica((3,4,5), float, 0)

def test_gather_replica4():
    run_gather_replica((3,4,5), float, 1)
