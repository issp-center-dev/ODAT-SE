# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing
from typing import TextIO, Union, List, Tuple
import copy
import time
from pathlib import Path

import numpy as np

import odatse
from odatse.util.neighborlist import load_neighbor_list, make_neighbor_list
import odatse.util.graph
import odatse.domain
from odatse import mpi

import abc

use_buffered = False


class ContinuousState:
    def __init__(self, x):
        self.x = copy.deepcopy(x)

class DiscreteState:
    def __init__(self, inode, x):
        self.inode = copy.deepcopy(inode)
        self.x = copy.deepcopy(x)


class StateSpace(abc.ABC):
    @abc.abstractmethod
    def propose(self, state):
        ...

    @abc.abstractmethod
    def choose(self, accept, new_state, old_state):
        ...

    @abc.abstractmethod
    def gather(self, state):
        ...

    @abc.abstractmethod
    def pick(self, state, index):
        ...

    def _gather_data(self, data):
        if use_buffered:
            return self._gather_data_buffer(data)
        else:
            return self._gather_data_object(data)

    def _gather_data_object(self, data):
        mpicomm = mpi.comm()
        return np.concatenate(mpicomm.allgather(data), axis=0)

    def _gather_data_buffer(self, data):
        from mpi4py.util.dtlib import from_numpy_dtype

        mpisize = mpi.size()
        mpirank = mpi.rank()
        mpicomm = mpi.comm()

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


class ContinuousStateSpace(StateSpace):
    def __init__(self, domain, info_param, rng, limitation):
        self.domain = domain
        self.rng = rng
        self.limitation = limitation

        self.xmin = self.domain.min_list
        self.xmax = self.domain.max_list

        if "step_list" in info_param:
            self.xstep = info_param.get("step_list")
        elif "unit_list" in info_param:
            print("WARNING: unit_list is obsolete. use step_list instead")
            self.xstep = info_param.get("unit_list")
        else:
            raise ValueError("ERROR: algorighm.param.step_list not specified")

    def initialize(self, nwalkers):
        self.domain.initialize(rng=self.rng, limitation=self.limitation, num_walkers=nwalkers)
        return ContinuousState(self.domain.initial_list)

    def propose(self, state):
        nwalkers = state.x.shape[0]
        dx = self.rng.normal(size=state.x.shape) * self.xstep
        new_state = ContinuousState(state.x + dx)
        return new_state, self._check_in_range(new_state.x), None

    def _check_in_range(self, x):
        nwalkers = x.shape[0]
        in_range = ((x >= self.xmin) & (x <= self.xmax)).all(axis=1)
        in_limit = [self.limitation.judge(x[idx,:]) for idx in range(nwalkers)]
        return in_range & in_limit

    def choose(self, accept, new_state, old_state):
        _acc = np.broadcast_to(accept.reshape(-1,1), old_state.x.shape)
        x_new = np.where(_acc, new_state.x, old_state.x)
        return ContinuousState(x_new)

    def gather(self, state):
        mpisize = mpi.size()
        if mpisize > 1:
            buf = self._gather_data(state.x)
            return ContinuousState(buf)
        else:
            return ContinuousState(state.x)

    def pick(self, state, index):
        return ContinuousState(state.x[index])

class DiscreteStateSpace(StateSpace):
    def __init__(self, domain, info_param, rng):
        self.domain = domain
        self.rng = rng

        self.node_coordinates = np.array(self.domain.grid)[:, 1:]
        self.nnodes = self.node_coordinates.shape[0]
        self._setup_neighbour(info_param)

    def initialize(self, nwalkers):
        inode = self.rng.randint(self.nnodes, size=nwalkers)
        x = self.node_coordinates[inode, :]
        return DiscreteState(inode, x)

    def propose(self, state):
        nwalkers = state.inode.shape[0]
        proposed_list = [self.rng.choice(self.neighbor_list[i]) for i in state.inode]
        proposed = np.array(proposed_list, dtype=np.int64)

        x = self.node_coordinates[proposed, :]
        new_state = DiscreteState(proposed, x)

        weight = [self.ncandidates[inode_old] / self.ncandidates[inode_new] for inode_old, inode_new in zip(state.inode, proposed_list)]

        return new_state, np.full(nwalkers, True), np.array(weight)

    def choose(self, accept, new_state, old_state):
        inode_new = np.where(accept, new_state.inode, old_state.inode)
        x_new = self.node_coordinates[inode_new, :]
        return DiscreteState(inode_new, x_new)

    def _setup_neighbour(self, info_param):
        """
        Set up the neighbor list for the discrete problem.

        Parameters
        ----------
        info_param : dict
            Dictionary containing algorithm parameters, including the path to the neighbor list file.

        Raises
        ------
        ValueError
            If the neighbor list path is not specified in the parameters.
        RuntimeError
            If the transition graph made from the neighbor list is not connected or not bidirectional.
        """
        mpirank = mpi.rank()
        mpicomm = mpi.comm()

        if "mesh_path" in info_param and "neighborlist_path" in info_param:
            nn_path = Path(info_param["neighborlist_path"]).expanduser()
            if mpirank == 0:
                nnlist = load_neighbor_list(nn_path, nnodes=self.nnodes)
            else:
                nnlist = None
            self.neighbor_list = mpicomm.bcast(nnlist, root=0)
        else:
            if "radius" not in info_param:
                raise KeyError("parameter \"algorithm.param.radius\" not specified")
            radius = info_param["radius"]
            print(f"DEBUG: create neighbor list, radius={radius}")
            self.neighbor_list = make_neighbor_list(self.node_coordinates, radius=radius, comm=mpicomm)

        # checks
        if not odatse.util.graph.is_connected(self.neighbor_list):
            raise RuntimeError(
                "ERROR: The transition graph made from neighbor list is not connected."
                "\nHINT: Increase neighborhood radius."
            )
        if not odatse.util.graph.is_bidirectional(self.neighbor_list):
            raise RuntimeError(
                "ERROR: The transition graph made from neighbor list is not bidirectional."
            )

        self.ncandidates = np.array([len(ns) - 1 for ns in self.neighbor_list], dtype=np.int64)

    def gather(self, state):
        mpisize = mpi.size()
        if mpisize > 1:
            inodes = self._gather_data(state.inode)
            return DiscreteState(inodes, self.node_coordinates[inodes, :])
        else:
            return DiscreteState(state.inode, state.x)

    def pick(self, state, index):
        return DiscreteState(state.inode[index], state.x[index])

