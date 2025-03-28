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
from odatse.util.data_writer import DataWriter

class AlgorithmBase(odatse.algorithm.AlgorithmBase):
    """
    Base class for Monte Carlo algorithms implementing common functionality.
    
    This class provides the foundation for various Monte Carlo methods, handling both
    continuous and discrete parameter spaces. It supports parallel execution with
    multiple walkers and temperature-based sampling methods.

    Implementation Details
    --------------------
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
    ----------
    _initialize() : Sets up initial walker positions and counters
    propose() : Generates candidate moves for walkers
    local_update() : Performs one Monte Carlo step
    _evaluate() : Computes objective function values
    """

    nwalkers: int

    iscontinuous: bool

    # continuous problem
    x: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    xstep: np.ndarray

    # discrete problem
    inode: np.ndarray
    nnodes: int
    node_coordinates: np.ndarray
    neighbor_list: List[List[int]]
    ncandidates: np.ndarray  # len(neighbor_list[i])-1

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
            self.xmin = self.domain.min_list
            self.xmax = self.domain.max_list

            if "step_list" in info_param:
                self.xstep = info_param.get("step_list")

            elif "unit_list" in info_param:
                # for compatibility, unit_list can also be accepted for step size.
                if self.mpirank == 0:
                    print("WARNING: unit_list is obsolete. use step_list instead")
                self.xstep = info_param.get("unit_list")
            else:
                # neither step_list nor unit_list is specified, report error.
                # default value not assumed.
                raise ValueError("ERROR: algorithm.param.step_list not specified")
        else:
            self.node_coordinates = np.array(self.domain.grid)[:, 1:]
            self.nnodes = self.node_coordinates.shape[0]
            self._setup_neighbour(info_param)

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
        if self.iscontinuous:
            self.domain.initialize(rng=self.rng, limitation=self.runner.limitation, num_walkers=self.nwalkers)
            self.x = self.domain.initial_list
            self.inode = None
        else:
            self.inode = self.rng.randint(self.nnodes, size=self.nwalkers)
            self.x = self.node_coordinates[self.inode, :]

        self.fx = np.zeros(self.nwalkers)
        self.best_fx = 0.0
        self.best_istep = 0
        self.best_iwalker = 0
        self.naccepted = 0
        self.ntrial = 0

    def _setup_neighbour(self, info_param: dict) -> None:
        """
        Configure neighbor relationships for discrete parameter spaces.

        Validation Steps:
        1. Loads neighbor list from specified file
        2. Verifies graph connectivity
           - Ensures all nodes can be reached
           - Prevents isolated subgraphs
        3. Confirms bidirectional connections
           - If A -> B exists, B -> A must exist
           - Required for detailed balance

        Graph Properties:
        - Stored as adjacency lists
        - Each node maintains list of valid neighbors
        - Supports variable number of neighbors per node
        - Preserves transition symmetry

        Parameters
        ----------
        info_param : dict
            Configuration containing neighborlist_path

        Raises
        ------
        ValueError
            If neighbor list file is not specified
        RuntimeError
            If graph is not connected or not bidirectional
        """
        if "mesh_path" in info_param and "neighborlist_path" in info_param:
            nn_path = self.root_dir / Path(info_param["neighborlist_path"]).expanduser()
            if self.mpirank == 0:
                nnlist = load_neighbor_list(nn_path, nnodes=self.nnodes)
            else:
                nnlist = None
            self.neighbor_list = self.mpicomm.bcast(nnlist, root=0)
        else:
            if "radius" not in info_param:
                raise KeyError("parameter \"algorithm.param.radius\" not specified")
            radius = info_param["radius"]
            print(f"DEBUG: create neighbor list, radius={radius}")
            self.neighbor_list = make_neighbor_list(self.node_coordinates, radius=radius, comm=self.mpicomm)

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


    def _evaluate(self, in_range: np.ndarray = None) -> np.ndarray:
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
        # print(">>> _evaluate")
        for iwalker in range(self.nwalkers):
            x = self.x[iwalker, :]
            if in_range is None or in_range[iwalker]:
                args = (self.istep, iwalker)

                time_sta = time.perf_counter()
                self.fx[iwalker] = self.runner.submit(x, args)
                time_end = time.perf_counter()
                self.timer["run"]["submit"] += time_end - time_sta
            else:
                self.fx[iwalker] = np.inf
        return self.fx

    def propose(self, current: np.ndarray) -> np.ndarray:
        """
        Generate proposed moves for all walkers.

        For continuous problems:
        - Uses Gaussian perturbations scaled by xstep
        - Generates independent proposals for each dimension

        For discrete problems:
        - Randomly selects a neighbor from each node's neighbor list
        - Ensures moves respect the graph connectivity

        Parameters
        ----------
        current : np.ndarray
            Current positions of all walkers
            For continuous: Array of coordinates
            For discrete: Array of node indices

        Returns
        -------
        np.ndarray
            Proposed new positions for all walkers
        """
        if self.iscontinuous:
            dx = self.rng.normal(size=(self.nwalkers, self.dimension)) * self.xstep
            proposed = current + dx
        else:
            proposed_list = [self.rng.choice(self.neighbor_list[i]) for i in current]
            proposed = np.array(proposed_list, dtype=np.int64)
        return proposed

    def local_update(
        self,
        beta: Union[float, np.ndarray],
        #file_trial: TextIO,
        #file_result: TextIO,
        extra_info_to_write: Union[List, Tuple] = None,
    ) -> None:
        """
        Perform one step of the Monte Carlo algorithm.

        Algorithm Flow:
        1. Generate proposed moves for all walkers
        2. Check if proposals are within valid bounds
        3. Evaluate objective function for valid proposals
        4. Apply Metropolis acceptance criterion:
           P(accept) = min(1, exp(-beta * (f_new - f_old)))
        5. For discrete case, adjust acceptance probability by:
           P *= (n_neighbors_old / n_neighbors_new)
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

        Implementation Notes
        ------------------
        - Handles numerical overflow in exponential calculation
        - Maintains detailed acceptance statistics
        - Supports both single and multiple temperature values
        - Preserves best solution across all steps
        """
        # make candidate
        x_old = copy.copy(self.x)
        if self.iscontinuous:
            self.x = self.propose(x_old)
            #judgement of "in_range"
            in_range_xmin = self.xmin <= self.x
            in_range_xmax = self.x <= self.xmax
            in_range_limitation = np.full(self.nwalkers, False)
            for index_walker in range(self.nwalkers):
                in_range_limitation[index_walker] = self.runner.limitation.judge(
                                                        self.x[index_walker]
                                                            )

            in_range = (in_range_xmin & in_range_xmax).all(axis=1) \
                       &in_range_limitation
        else:
            i_old = copy.copy(self.inode)
            self.inode = self.propose(self.inode)
            self.x = self.node_coordinates[self.inode, :]
            in_range = np.ones(self.nwalkers, dtype=bool)

        # evaluate "Energy"s
        fx_old = self.fx.copy()
        self._evaluate(in_range)

        if self.fp_trial:
            self._write_result(self.fp_trial, extra_info_to_write)

        fdiff = self.fx - fx_old

        # Ignore an overflow warning in np.exp(x) for x >~ 710
        # and an invalid operation warning in exp(nan) (nan came from 0 * inf)
        # Note: fdiff (fx) becomes inf when x is out of range
        # old_setting = np.seterr(over="ignore")
        old_setting = np.seterr(all="ignore")
        probs = np.exp(-beta * fdiff)
        #probs[np.isnan(probs)] = 0.0
        np.seterr(**old_setting)

        if not self.iscontinuous:
            probs *= self.ncandidates[i_old] / self.ncandidates[self.inode]
        tocheck = in_range & (probs < 1.0)
        num_check = np.count_nonzero(tocheck)

        accepted = in_range.copy()
        accepted[tocheck] = self.rng.rand(num_check) < probs[tocheck]
        rejected = ~accepted
        self.naccepted += accepted.sum()
        self.ntrial += accepted.size

        # revert rejected steps
        self.x[rejected, :] = x_old[rejected, :]
        self.fx[rejected] = fx_old[rejected]
        if not self.iscontinuous:
            self.inode[rejected] = i_old[rejected]

        minidx = np.argmin(self.fx)
        if self.fx[minidx] < self.best_fx:
            np.copyto(self.best_x, self.x[minidx, :])
            self.best_fx = self.fx[minidx]
            self.best_istep = self.istep
            self.best_iwalker = typing.cast(int, minidx)

        if self.fp_result:
            self._write_result(self.fp_result, extra_info_to_write)

    def _set_writer(self, fp_trial, fp_result):
        self.fp_trial = fp_trial
        self.fp_result = fp_result

    def _write_result(self, writer, extras = None):
        for iwalker in range(self.nwalkers):
            if isinstance(self.Tindex, int):
                beta = self.betas[self.Tindex]
            else:
                beta = self.betas[self.Tindex[iwalker]]

            if self.input_as_beta:
                tval = beta
            else:
                tval = 1.0 / beta

            data = [self.istep, iwalker, tval, self.fx[iwalker], *self.x[iwalker,:]]
            if extras:
                for extra in extras:
                    data.append(extra[iwalker])

            writer.write(*data)

def read_Ts(info: dict, numT: int = None) -> Tuple[bool, np.ndarray]:
    """
    Read temperature or inverse-temperature values from the provided info dictionary.

    Parameters
    ----------
    info : dict
        Dictionary containing temperature or inverse-temperature parameters.
    numT : int, optional
        Number of temperature or inverse-temperature values to generate, by default None.

    Returns
    -------
    as_beta : bool
        True if using inverse-temperature, False if using temperature.
    betas : np.ndarray
        Sequence of inverse-temperature values.

    Raises
    ------
    ValueError
        If numT is not specified, or if both Tmin/Tmax and bmin/bmax are defined, or if neither are defined,
        or if bmin/bmax or Tmin/Tmax values are invalid.
    RuntimeError
        If the mode is unknown (neither set_T nor set_b).
        
    Examples
    --------
    >>> info = {"Tmin": 0.1, "Tmax": 10.0, "Tlogspace": True}
    >>> as_beta, betas = read_Ts(info, numT=50)
    >>> print(as_beta)
    False
    >>> print(betas[0], betas[-1])
    10.0 0.1
    """
    if numT is None:
        raise ValueError("read_Ts: numT is not specified")

    Tmin = info.get("Tmin", None)
    Tmax = info.get("Tmax", None)
    bmin = info.get("bmin", None)
    bmax = info.get("bmax", None)
    logscale = info.get("Tlogspace", True)

    if "Tinvspace" in info:
        raise ValueError("Tinvspace is deprecated. Use bmax/bmin instead.")

    set_b = (bmin is not None or bmax is not None)
    set_T = (Tmin is not None or Tmax is not None)

    if set_b and set_T:
        raise ValueError("both Tmin/Tmax and bmin/bmax are defined")
    if (not set_b) and (not set_T):
        raise ValueError("neither Tmin/Tmax nor bmin/bmax are defined")

    if set_b:
        if bmin is None or bmax is None:
            raise ValueError("bmin and bmax must be set")

        input_as_beta = True
        if not np.isreal(bmin) or bmin < 0.0:
            raise ValueError("bmin must be zero or a positive real number")
        if not np.isreal(bmax) or bmax < 0.0:
            raise ValueError("bmin must be zero or a positive real number")
        if bmin > bmax:
            raise ValueError("bmin must be smaller than or equal to bmax")

        if logscale:
            if bmin == 0.0:
                raise ValueError("bmin must be greater than 0.0 when Tlogspace is True")
            betas = np.logspace(start=np.log10(bmin), stop=np.log10(bmax), num=numT)
        else:
            betas = np.linspace(start=bmin, stop=bmax, num=numT)

    elif set_T:
        if Tmin is None or Tmax is None:
            raise ValueError("Tmin and Tmax must be set")

        input_as_beta = False
        if not np.isreal(Tmin) or Tmin <= 0.0:
            raise ValueError("Tmin must be a positive real number")
        if not np.isreal(Tmax) or Tmax <= 0.0:
            raise ValueError("Tmax must be a positive real number")
        if Tmin > Tmax:
            raise ValueError("Tmin must be smaller than or equal to Tmax")

        if logscale:
            Ts = np.logspace(start=np.log10(Tmin), stop=np.log10(Tmax), num=numT)
        else:
            Ts = np.linspace(start=Tmin, stop=Tmax, num=numT)

        betas = 1.0 / Ts
    else:
        raise RuntimeError("read_Ts: unknown mode: not set_T nor set_b")

    return input_as_beta, betas
