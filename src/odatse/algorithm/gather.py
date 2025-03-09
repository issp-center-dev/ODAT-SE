# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Dict, Union

import os
import numpy as np
#-- delay import
# from mpi4py.util.dtlib import from_numpy_dtype

import odatse.mpi

_use_buffer = int(os.environ.get("ODATSE_USE_MPI_BUFFERED", 0)) == 1

# def set_config(*, use_buffer):
#     global _use_buffer
#     _use_buffer = use_buffer

def gather_replica(data, axis=0):
    mpisize = odatse.mpi.size()
    if mpisize == 1:
        return data
    else:
        if _use_buffer:
            return _do_gather_replica_buffer(data, axis)
        else:
            return _do_gather_object(data, axis)

def gather_data(data, axis=0):
    mpisize = odatse.mpi.size()
    if mpisize == 1:
        return data
    else:
        if _use_buffer:
            return _do_gather_data_buffer(data, axis)
        else:
            return _do_gather_object(data, axis)

def _do_gather_object(data, axis):
    mpicomm = odatse.mpi.comm()
    return np.concatenate(mpicomm.allgather(data), axis=axis)

def _do_gather_replica_buffer(data, axis):
    if axis == 0:
        return _do_gather_variable_buffer(data)
    else:
        return _do_gather_variable_buffer_transpose(data, axis)

def _do_gather_data_buffer(data, axis):
    if axis == 0:
        return _do_gather_fixed_buffer(data)
    else:
        return _do_gather_fixed_buffer_transpose(data, axis)

def _do_gather_variable_buffer(data):
    from mpi4py.util.dtlib import from_numpy_dtype
    mpicomm = odatse.mpi.comm()
    mpisize = odatse.mpi.size()

    sh = data.shape
    nrep = np.array([sh[0]], dtype=np.int64)
    nreps = np.zeros(mpisize, dtype=np.int64)
    mpicomm.Allgather(nrep, nreps)

    displ = np.cumsum(nreps) - nreps
    nrep_total = np.sum(nreps)
    ndim = np.prod(sh[1:], dtype=int)

    buf = np.zeros((nrep_total, *sh[1:]), dtype=data.dtype)
    dtype = from_numpy_dtype(data.dtype)
    mpicomm.Allgatherv([data, dtype], [buf, nreps*ndim, displ*ndim, dtype])
    return buf

def _do_gather_variable_buffer_transpose(data, axis):
    axes = [i for i in range(data.ndim)]
    axes[axis], axes[0] = axes[0], axes[axis]
    axes = tuple(axes)
    data_t = np.transpose(data, axes).copy()
    data_gather = _do_gather_variable_buffer(data_t)
    return np.transpose(data_gather, axes)
        
def _do_gather_fixed_buffer(data):
    from mpi4py.util.dtlib import from_numpy_dtype
    mpicomm = odatse.mpi.comm()
    mpisize = odatse.mpi.size()

    sh = data.shape
    buf = np.zeros((mpisize*sh[0], *sh[1:]), dtype=data.dtype)
    mpicomm.Allgather(data, buf)
    return buf
    
def _do_gather_fixed_buffer_transpose(data, axis):    
    axes = [i for i in range(data.ndim)]
    axes[axis], axes[0] = axes[0], axes[axis]
    axes = tuple(axes)
    data_t = np.transpose(data, axes).copy()
    data_gather = _do_gather_fixed_buffer(data_t)
    return np.transpose(data_gather, axes)
