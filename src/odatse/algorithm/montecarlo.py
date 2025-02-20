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
import odatse.domain
from odatse import mpi
from odatse.algorithm.state import ContinuousStateSpace, DiscreteStateSpace


class AlgorithmBase(odatse.algorithm.AlgorithmBase):
    """Base of Monte Carlo

    Attributes
    ==========
    nwalkers: int
        the number of walkers (per one process)
    x: np.ndarray
        current configurations
        (NxD array, N is the number of walkers and D is the dimension)
    fx: np.ndarray
        current "Energy"s
    istep: int
        current step (or, the number of calculated energies)
    best_x: np.ndarray
        best configuration
    best_fx: float
        best "Energy"
    best_istep: int
        index of best configuration (step)
    best_iwalker: int
        index of best configuration (walker)
    comm: MPI.comm
        MPI communicator
    rank: int
        MPI rank
    Ts: np.ndarray
        List of temperatures
    Tindex: np.ndarray
        Temperature index
    """

    nwalkers: int

    iscontinuous: bool

    # # continuous problem
    # x: np.ndarray
    # xmin: np.ndarray
    # xmax: np.ndarray
    # xstep: np.ndarray

    # # discrete problem
    # inode: np.ndarray
    # nnodes: int
    # node_coordinates: np.ndarray
    # neighbor_list: List[List[int]]
    # ncandidates: np.ndarray  # len(neighbor_list[i])-1

    # state: Union[ContinuousState, DiscreteState]

    numsteps: int

    fx: np.ndarray
    istep: int
    best_x: np.ndarray
    best_fx: float
    best_istep: int
    best_iwalker: int
    betas: np.ndarray
    input_as_beta: bool
    Tindex: np.ndarray

    ntrial: int
    naccepted: int

    def __init__(self, info: odatse.Info,
             runner: odatse.Runner = None,
             domain = None,
             nwalkers: int = 1,
             run_mode: str = "initial") -> None:
        """
        Initialize the AlgorithmBase class.

        Parameters
        ----------
        info : odatse.Info
            Information object containing algorithm parameters.
        runner : odatse.Runner, optional
            Runner object for executing the algorithm (default is None).
        domain : optional
            Domain object defining the problem space (default is None).
        nwalkers : int, optional
            Number of walkers (default is 1).
        run_mode : str, optional
            Mode of the run, e.g., "initial" (default is "initial").
        """
        time_sta = time.perf_counter()
        super().__init__(info=info, runner=runner, run_mode=run_mode)
        self.nwalkers = nwalkers

        if domain:
            if isinstance(domain, odatse.domain.MeshGrid):
                self.iscontinuous = False
            elif isinstance(domain, odatse.domain.Region):
                self.iscontinuous = True
            else:
                raise ValueError("ERROR: unsupoorted domain type {}".format(type(domain)))
            self.domain = domain
        else:
            info_param = info.algorithm["param"]
            if "mesh_path" in info_param:
                self.iscontinuous = False
                self.domain = odatse.domain.MeshGrid(info)
            elif "use_grid" in info_param and info_param["use_grid"] == True:
                self.iscontinuous = False
                self.domain = odatse.domain.MeshGrid(info)
            else:
                self.iscontinuous = True
                self.domain = odatse.domain.Region(info)

        if self.iscontinuous:
            self.statespace = ContinuousStateSpace(self.domain, info_param, limitation=self.runner.limitation, rng=self.rng)
        else:
            self.statespace = DiscreteStateSpace(self.domain, info_param, rng=self.rng)

        time_end = time.perf_counter()
        self.timer["init"]["total"] = time_end - time_sta
        self.Tindex = 0
        self.input_as_beta = False

    def _initialize(self):
        """
        Initialize the algorithm state.

        This method sets up the initial state of the algorithm, including the
        positions and energies of the walkers, and resets the counters for
        accepted and trial steps.
        """
        self.state = self.statespace.initialize(self.nwalkers)
        self.fx = np.zeros(self.nwalkers)

        self.best_fx = 0.0
        self.best_istep = 0
        self.best_iwalker = 0
        self.naccepted = 0
        self.ntrial = 0

    def _evaluate(self, state, in_range: np.ndarray = None) -> np.ndarray:
        """
        Evaluate the current "Energy"s.

        This method overwrites `self.fx` with the result.

        Parameters
        ----------
        in_range : np.ndarray, optional
            Array indicating whether each walker is within the valid range (default is None).

        Returns
        -------
        np.ndarray
            Array of evaluated energies for the current configurations.
        """
        fx = np.zeros(self.nwalkers, dtype=np.float64)

        for iwalker in range(self.nwalkers):
            x = state.x[iwalker, :]
            if in_range is None or in_range[iwalker]:
                args = (self.istep, iwalker)

                time_sta = time.perf_counter()
                fx[iwalker] = self.runner.submit(x, args)
                time_end = time.perf_counter()
                self.timer["run"]["submit"] += time_end - time_sta
            else:
                fx[iwalker] = np.inf

        return fx

    def local_update(
        self,
        beta: Union[float, np.ndarray],
        file_trial: TextIO,
        file_result: TextIO,
        extra_info_to_write: Union[List, Tuple] = None,
    ):
        """
        one step of Monte Carlo

        Parameters
        ----------
        beta: np.ndarray
            inverse temperature for each walker
        file_trial: TextIO
            log file for all trial points
        file_result: TextIO
            log file for all generated samples
        extra_info_to_write: List of np.ndarray or tuple of np.ndarray
            extra information to write
        """
        # make candidate
        old_state = copy.deepcopy(self.state)
        old_fx = copy.deepcopy(self.fx)

        new_state, in_range, weight = self.statespace.propose(old_state)
        #self.state = new_state

        # evaluate "Energy"s
        new_fx = self._evaluate(new_state, in_range)
        #XXX
        self.state = new_state
        self.fx = new_fx
        self._write_result(file_trial, extra_info_to_write=extra_info_to_write)

        #print(old_fx, new_fx)
        fdiff = new_fx - old_fx

        # Ignore an overflow warning in np.exp(x) for x >~ 710
        # and an invalid operation warning in exp(nan) (nan came from 0 * inf)
        # Note: fdiff (fx) becomes inf when x is out of range
        # old_setting = np.seterr(over="ignore")
        old_setting = np.seterr(all="ignore")
        probs = np.exp(-beta * fdiff)
        #probs[np.isnan(probs)] = 0.0
        if weight is not None:
            probs *= weight
        np.seterr(**old_setting)

        tocheck = in_range & (probs < 1.0)
        num_check = np.count_nonzero(tocheck)

        accepted = in_range.copy()
        accepted[tocheck] = self.rng.rand(num_check) < probs[tocheck]

        self.naccepted += accepted.sum()
        self.ntrial += accepted.size

        # update
        self.state = self.statespace.choose(accepted, new_state, old_state)
        self.fx = np.where(accepted, new_fx, old_fx)

        minidx = np.argmin(self.fx)
        if self.fx[minidx] < self.best_fx:
            np.copyto(self.best_x, self.state.x[minidx, :])
            self.best_fx = self.fx[minidx]
            self.best_istep = self.istep
            self.best_iwalker = typing.cast(int, minidx)
        self._write_result(file_result, extra_info_to_write=extra_info_to_write)

    def _gather(self, data):
        ndata, *ndim = data.shape
        if self.mpisize > 1:
            buf = np.zeros((self.mpisize, ndata, *ndim), dtype=data.dtype)
            self.mpicomm.Allgather(data, buf)
            return buf.reshape(-1, *ndim)
        else:
            return np.copy(data)

    def _write_result_header(self, fp, extra_names=None) -> None:
        """
        Write the header for the result file.

        Parameters
        ----------
        fp : TextIO
            File pointer to the result file.
        extra_names : list of str, optional
            Additional column names to include in the header.
        """
        if self.input_as_beta:
            fp.write("# step walker beta fx")
        else:
            fp.write("# step walker T fx")
        for label in self.label_list:
            fp.write(f" {label}")
        if extra_names is not None:
            for label in extra_names:
                fp.write(f" {label}")
        fp.write("\n")

    def _write_result(self, fp, extra_info_to_write: Union[List, Tuple] = None) -> None:
        """
        Write the result of the current step to the file.

        Parameters
        ----------
        fp : TextIO
            File pointer to the result file.
        extra_info_to_write : Union[List, Tuple], optional
            Additional information to write for each walker (default is None).
        """
        for iwalker in range(self.nwalkers):
            if isinstance(self.Tindex, int):
                beta = self.betas[self.Tindex]
            else:
                beta = self.betas[self.Tindex[iwalker]]
            fp.write(f"{self.istep}")
            fp.write(f" {iwalker}")
            if self.input_as_beta:
                fp.write(f" {beta}")
            else:
                fp.write(f" {1.0/beta}")
            fp.write(f" {self.fx[iwalker]}")
            for x in self.state.x[iwalker, :]:
                fp.write(f" {x}")
            if extra_info_to_write is not None:
                for ex in extra_info_to_write:
                    fp.write(f" {ex[iwalker]}")
            fp.write("\n")
        fp.flush()

