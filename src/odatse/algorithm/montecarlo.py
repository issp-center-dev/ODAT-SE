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
from odatse.util.data_writer import DataWriter


class AlgorithmBase(odatse.algorithm.AlgorithmBase):
    """
    Base class for Monte Carlo algorithms implementing common functionality.
    
    This class provides the foundation for various Monte Carlo methods, handling both
    continuous and discrete parameter spaces. It supports parallel execution with
    multiple walkers and temperature-based sampling methods.

    Implementation Details
    ----------------------
    The class handles two types of parameter spaces:
      1. Continuous: Uses real-valued parameters within specified bounds
      2. Discrete: Uses node-based parameters with defined neighbor relationships

    For continuous problems:
      - Parameters are bounded by xmin and xmax
      - Steps are controlled by xstep for each dimension
    
    For discrete problems:
      - Parameters are represented as nodes in a graph
      - Transitions are only allowed between neighboring nodes
      - Neighbor relationships must form a connected, bidirectional graph

    The sampling process:
      1. Initializes walkers in valid positions
      2. Proposes moves based on the parameter space type
      3. Evaluates the objective function ("Energy")
      4. Accepts/rejects moves based on the Monte Carlo criterion
      5. Tracks the best solution found

    Key Methods
    -----------
      _initialize() :
         Sets up initial walker positions and counters
      propose() :
         Generates candidate moves for walkers
      local_update() :
         Performs one Monte Carlo step
      _evaluate() :
         Computes objective function values
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
            Runner object for executing the algorithm, by default None.
        domain : odatse.domain.Domain, optional
            Domain object defining the problem space, by default None.
        nwalkers : int, optional
            Number of walkers to use in the simulation, by default 1.
        run_mode : str, optional
            Mode of the run, e.g., "initial", by default "initial".
            
        Raises
        ------
        ValueError
            If an unsupported domain type is provided or required parameters are missing.
            
        Examples
        --------
        >>> info = odatse.Info(config_file_path)
        >>> runner = odatse.Runner()
        >>> algorithm = AlgorithmBase(info, runner, nwalkers=100)
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

        #-- writer
        self.fp_trial = None
        self.fp_result = None

    def _initialize(self):
        """
        Initialize the Monte Carlo simulation state.

        For continuous problems:
          - Uses domain.initialize to generate valid initial positions
          - Respects any additional limitations from the runner

        For discrete problems:
          - Randomly assigns walkers to valid nodes
          - Maps node indices to actual coordinate positions

        Also initializes:
          - Objective function values (fx) to zero
          - Best solution tracking variables
          - Acceptance counters for monitoring convergence
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
        Evaluate objective function for current walker positions.

        Optimization Features:
          - Skips evaluation for out-of-bounds positions
          - Tracks evaluation timing statistics
          - Supports parallel evaluation across walkers

        Parameters
        ----------
        in_range : np.ndarray, optional
            Boolean mask indicating valid positions
            True = position is valid and should be evaluated
            False = position is invalid, will be assigned inf

        Returns
        -------
        np.ndarray
            Array of objective function values
            Invalid positions are assigned inf
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
        extra_info_to_write: Union[List, Tuple] = None,
    ) -> None:
        """
        Perform one step of the Monte Carlo algorithm.

        Algorithm Flow:
          1. Generate proposed moves for all walkers
          2. Check if proposals are within valid bounds
          3. Evaluate objective function for valid proposals
          4. Apply Metropolis acceptance criterion:
             ``P(accept) = min(1, exp(-beta * (f_new - f_old)))``
          5. For discrete case, adjust acceptance probability by:
             ``P *= (n_neighbors_old / n_neighbors_new)``
          6. Update positions and energies
          7. Track best solution found
          8. Log results if writers are configured

        Parameters
        ----------
        beta : Union[float, np.ndarray]
            Inverse temperature(s) controlling acceptance probability
            Can be single value or array (one per walker)
        extra_info_to_write : Union[List, Tuple], optional
            Additional data to log with results

        Notes
        -----
          - Handles numerical overflow in exponential calculation
          - Maintains detailed acceptance statistics
          - Supports both single and multiple temperature values
          - Preserves best solution across all steps
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
        self._write_result(self.fp_trial, extras=extra_info_to_write)

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
        self._write_result(self.fp_result, extras=extra_info_to_write)

    def _set_writer(self, fp_trial, fp_result):
        self.fp_trial = fp_trial
        self.fp_result = fp_result

    def _write_result(self, writer, extras=None):
        for iwalker in range(self.nwalkers):
            if isinstance(self.Tindex, int):
                beta = self.betas[self.Tindex]
            else:
                beta = self.betas[self.Tindex[iwalker]]

            if self.input_as_beta:
                tval = beta
            else:
                tval = 1.0 / beta

            data = [self.istep, iwalker, tval, self.fx[iwalker], *self.state.x[iwalker,:]]
            if extras:
                for extra in extras:
                    data.append(extra[iwalker])

            writer.write(*data)
