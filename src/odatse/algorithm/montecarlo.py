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


class AlgorithmBase(odatse.algorithm.AlgorithmBase):
    """
    Base class for Monte Carlo algorithms.
    
    This class provides the fundamental structure and methods for Monte Carlo
    simulations, including walker management, energy evaluation, and state tracking.
    
    Attributes
    ----------
    nwalkers : int
        The number of walkers (per one process).
    x : np.ndarray
        Current configurations (NxD array, N is the number of walkers and D is the dimension).
    fx : np.ndarray
        Current "Energy"s for each walker.
    istep : int
        Current step (or, the number of calculated energies).
    best_x : np.ndarray
        Best configuration found so far.
    best_fx : float
        Best "Energy" value found so far.
    best_istep : int
        Index of step when best configuration was found.
    best_iwalker : int
        Index of walker that found the best configuration.
    comm : MPI.comm
        MPI communicator for parallel processing.
    rank : int
        MPI rank of the current process.
    Ts : np.ndarray
        List of temperatures used in the simulation.
    Tindex : np.ndarray
        Temperature indices for each walker.
    iscontinuous : bool
        Whether the problem is continuous (True) or discrete (False).
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

    def _initialize(self) -> None:
        """
        Initialize the algorithm state.

        This method sets up the initial state of the algorithm, including the
        positions and energies of the walkers, and resets the counters for
        accepted and trial steps.
        
        Returns
        -------
        None
            Updates the internal state of the algorithm.
            
        Examples
        --------
        >>> algorithm._initialize()
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
        Set up the neighbor list for the discrete problem.

        Parameters
        ----------
        info_param : dict
            Dictionary containing algorithm parameters, including the path to 
            the neighbor list file or radius for neighbor generation.

        Raises
        ------
        ValueError
            If the neighbor list path or radius is not specified in the parameters.
        RuntimeError
            If the transition graph made from the neighbor list is not connected or not bidirectional.
            
        Returns
        -------
        None
            Updates the neighbor_list and ncandidates attributes.
            
        Examples
        --------
        >>> algorithm._setup_neighbour({"radius": 2.0})
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
        Evaluate the current "Energy"s for all walkers.

        This method overwrites `self.fx` with the evaluation results.

        Parameters
        ----------
        in_range : np.ndarray, optional
            Boolean array indicating whether each walker is within the valid range, by default None.
            If None, all walkers are considered in range.

        Returns
        -------
        np.ndarray
            Array of evaluated energies for the current configurations.
            
        Examples
        --------
        >>> energies = algorithm._evaluate()
        >>> in_range = np.ones(algorithm.nwalkers, dtype=bool)
        >>> energies = algorithm._evaluate(in_range)
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
        Propose the next candidate positions for the walkers.

        Parameters
        ----------
        current : np.ndarray
            Current positions of the walkers.

        Returns
        -------
        np.ndarray
            Proposed new positions for the walkers.
            
        Examples
        --------
        >>> current_positions = algorithm.x
        >>> proposed_positions = algorithm.propose(current_positions)
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
        file_trial: TextIO,
        file_result: TextIO,
        extra_info_to_write: Union[List, Tuple] = None,
    ) -> None:
        """
        Perform one step of Monte Carlo update.
        
        This method proposes new configurations for all walkers, evaluates their energies,
        and accepts or rejects the proposed moves according to the Metropolis criterion.
        
        Parameters
        ----------
        beta : Union[float, np.ndarray]
            Inverse temperature for each walker.
        file_trial : TextIO
            Log file for all trial points.
        file_result : TextIO
            Log file for all generated samples.
        extra_info_to_write : Union[List, Tuple], optional
            Extra information to write to the log files, by default None.
            
        Returns
        -------
        None
            This method updates the internal state and writes to log files.
            
        Examples
        --------
        >>> beta = 1.0
        >>> with open('trial.log', 'w') as trial_file, open('result.log', 'w') as result_file:
        >>>     algorithm.local_update(beta, trial_file, result_file)
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
        self._write_result(file_trial, extra_info_to_write=extra_info_to_write)

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
        self._write_result(file_result, extra_info_to_write=extra_info_to_write)

    def _write_result_header(self, fp: TextIO, extra_names: List[str] = None) -> None:
        """
        Write the header for the result file.

        Parameters
        ----------
        fp : TextIO
            File pointer to the result file.
        extra_names : list of str, optional
            Additional column names to include in the header, by default None.
            
        Returns
        -------
        None
            Writes the header to the provided file pointer.
            
        Examples
        --------
        >>> with open('result.log', 'w') as fp:
        >>>     algorithm._write_result_header(fp, ['acceptance_rate'])
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

    def _write_result(self, fp: TextIO, extra_info_to_write: Union[List, Tuple] = None) -> None:
        """
        Write the result of the current step to the file.

        Parameters
        ----------
        fp : TextIO
            File pointer to the result file.
        extra_info_to_write : Union[List, Tuple], optional
            Additional information to write for each walker, by default None.
            
        Returns
        -------
        None
            Writes the current state to the provided file pointer.
            
        Examples
        --------
        >>> with open('result.log', 'a') as fp:
        >>>     algorithm._write_result(fp, [acceptance_rates])
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
            for x in self.x[iwalker, :]:
                fp.write(f" {x}")
            if extra_info_to_write is not None:
                for ex in extra_info_to_write:
                    fp.write(f" {ex[iwalker]}")
            fp.write("\n")
        fp.flush()

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
