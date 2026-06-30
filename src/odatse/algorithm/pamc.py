# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Optional, TYPE_CHECKING

from io import open
import copy
import time
import sys

import numpy as np

import odatse
import odatse.exception
import odatse.algorithm.montecarlo
from odatse.algorithm.gather import gather_replica, gather_data
import odatse.util.resampling
from odatse.util.read_ts import read_Ts
from odatse.util.data_writer import DataWriter

if TYPE_CHECKING:
    from mpi4py import MPI


class Algorithm(odatse.algorithm.montecarlo.AlgorithmBase):
    """
    Population Annealing Monte Carlo (PAMC) Algorithm Implementation.
    
    PAMC is an advanced Monte Carlo method that combines features of simulated annealing
    and population-based methods. It maintains a population of walkers that evolves
    through a sequence of temperatures, using both Monte Carlo updates and resampling.

    Key Features:
      - Maintains a population of walkers that evolves through temperature steps
      - Uses importance sampling with weight-based resampling
      - Supports both fixed and variable population sizes
      - Calculates free energy differences between temperature steps
      - Provides error estimates through population statistics

    Algorithm Flow:
      1. Initialize walker population at highest temperature
      2. For each temperature step:
      
        a. Perform Monte Carlo updates at current temperature
        b. Calculate weights for next temperature
        c. Resample population based on weights
        d. Update statistical estimates
        
      3. Track best solutions and maintain system statistics

    Attributes
    ----------
    x : np.ndarray
        Current configurations for all walkers
    logweights : np.ndarray
        Log of importance weights for each walker
    fx : np.ndarray
        Current energy/objective values
    nreplicas : np.ndarray
        Number of replicas at each temperature
    betas : np.ndarray
        Inverse temperatures (β = 1/T)
    Tindex : int
        Current temperature index
    Fmeans : np.ndarray
        Mean free energy at each temperature
    Ferrs : np.ndarray
        Free energy error estimates
    walker_ancestors : np.ndarray
        Tracks genealogy of walkers for analysis
    """

    # x: np.ndarray
    # xmin: np.ndarray
    # xmax: np.ndarray
    # #xunit: np.ndarray
    # xstep: np.ndarray

    numsteps: int
    numsteps_annealing: int

    logweights: np.ndarray
    fx: np.ndarray
    istep: int
    nreplicas: np.ndarray
    betas: np.ndarray
    Tindex: int

    fx_from_reset: np.ndarray

    # 0th column: number of accepted MC trials
    # 1st column: number of total MC trials
    naccepted_from_reset: np.ndarray
    resampling_interval: int

    nset_bootstrap: int

    walker_ancestors: np.ndarray
    populations: np.ndarray
    family_lo: int
    family_hi: int

    Fmeans: np.ndarray
    Ferrs: np.ndarray

    # PAMC-specific fields appended to the MC base checkpoint.
    # Fields restored with slice assignment or special logic are still listed
    # here so that ``__getstate__`` saves them; ``_apply_state`` handles them
    # explicitly rather than via the generic setattr loop.
    _checkpoint_attrs: list[str] = [
        "betas", "nwalkers", "input_as_beta", "numsteps_for_T",
        "Tindex", "index_from_reset",
        "logZ", "logZs", "logweights",
        "Fmeans", "Ferrs", "nreplicas", "populations",
        "family_lo", "family_hi", "walker_ancestors", "fx_from_reset",
        "naccepted_from_reset", "acceptance_ratio", "pr_list",
    ]

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        run_mode: str = "initial",
    ) -> None:
        """
        Initialize the Algorithm class.

        Parameters
        ----------
        info : odatse.Info
            Information object containing algorithm parameters.
        runner : odatse.Runner, optional
            Runner object for executing the algorithm, by default None.
        run_mode : str, optional
            Mode in which to run the algorithm, by default "initial".
        """
        time_sta = time.perf_counter()

        info_pamc = info.algorithm["pamc"]
        nwalkers = info_pamc.get("nreplica_per_proc", 1)

        super().__init__(info=info, runner=runner, nwalkers=nwalkers, run_mode=run_mode)

        self.verbose = True and odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0

        numT = self._find_scheduling(info_pamc)

        self.input_as_beta, self.betas = read_Ts(info_pamc, numT=numT)
        self.betas.sort()

        self.fix_nwalkers = info_pamc.get("fix_num_replicas", True)
        self.resampling_interval = info_pamc.get("resampling_interval", 1)
        if self.resampling_interval < 1:
            self.resampling_interval = numT + 1

        self.export_combined_files = info_pamc.get("export_combined_files", False)
        self.separate_T = info_pamc.get("separate_T", True)
        self.anneal_from_beta0 = self.betas[0] > 0.0 and info_pamc.get(
            "anneal_from_beta0", False
        )

        time_end = time.perf_counter()
        self.timer["init"]["total"] = time_end - time_sta

    def _initialize(self) -> None:
        super()._initialize()

        # PAMC-specific counters for a fresh run
        self.Tindex = 0
        self.index_from_reset = 0
        self.istep = 0

        numT = len(self.betas)

        self.logZ = 0.0
        self.logZs = np.zeros(numT)
        self.logweights = np.zeros(self.nwalkers)

        self.Fmeans = np.zeros(numT)
        self.Ferrs = np.zeros(numT)
        nreplicas = odatse.mpi.algsize() * self.nwalkers
        self.nreplicas = np.full(numT, nreplicas)

        self.populations = np.zeros((numT, self.nwalkers), dtype=int)
        self.family_lo = self.nwalkers * odatse.mpi.algrank()
        self.family_hi = self.nwalkers * (odatse.mpi.algrank() + 1)
        self.walker_ancestors = np.arange(self.family_lo, self.family_hi)
        self.fx_from_reset = np.zeros((self.resampling_interval, self.nwalkers))
        self.naccepted_from_reset = np.zeros((self.resampling_interval, 2), dtype=int)
        self.acceptance_ratio = np.zeros(numT)
        self.pr_list = np.zeros(numT)

        self._show_parameters()

    def _find_scheduling(self, info_pamc) -> int:
        """
        Determine the temperature schedule and number of steps.

        The schedule can be specified in three ways:
          1. Total steps and steps per temperature
          2. Total steps and number of temperatures
          3. Steps per temperature and number of temperatures

        The method ensures even distribution of computational effort across
        temperature steps while respecting the specified constraints.

        Parameters
        ----------
        info_pamc : dict
            Configuration dictionary containing:
              - numsteps: Total number of Monte Carlo steps
              - numsteps_annealing: Steps per temperature
              - Tnum: Number of temperature points

        Returns
        -------
        int
            Number of temperature steps in the schedule

        Raises
        ------
        odatse.exception.InputError
            If the scheduling parameters are inconsistent
        """
        numsteps = info_pamc.get("numsteps", 0)
        numsteps_annealing = info_pamc.get("numsteps_annealing", 0)
        numT = info_pamc.get("Tnum", 0)

        oks = np.array([numsteps, numsteps_annealing, numT]) > 0
        if np.count_nonzero(oks) != 2:
            msg = "ERROR: Two of 'numsteps', 'numsteps_annealing', "
            msg += "and 'Tnum' should be positive in the input file\n"
            msg += f"  numsteps = {numsteps}\n"
            msg += f"  numsteps_annealing = {numsteps_annealing}\n"
            msg += f"  Tnum = {numT}\n"
            raise odatse.exception.InputError(msg)

        if numsteps <= 0:
            self.numsteps_for_T = np.full(numT, numsteps_annealing)
        elif numsteps_annealing <= 0:
            nr = numsteps // numT
            self.numsteps_for_T = np.full(numT, nr)
            rem = numsteps - nr * numT
            if rem > 0:
                # Distribute the remainder over the first ``rem`` temperatures
                # so that the per-temperature step counts sum to ``numsteps``.
                self.numsteps_for_T[0:rem] += 1
        else:
            ss: list[int] = []
            while numsteps > 0:
                if numsteps > numsteps_annealing:
                    ss.append(numsteps_annealing)
                    numsteps -= numsteps_annealing
                else:
                    ss.append(numsteps)
                    numsteps = 0
            self.numsteps_for_T = np.array(ss)
            numT = len(ss)

        return numT

    def _run(self) -> None:

        # dispatch は prepare() が処理済み

        writer = self._setup_writer()

        if self.mode.startswith("init"):
            beta = self.betas[self.Tindex]
            self.fx = self._evaluate(self.state)
            if self.anneal_from_beta0:
                # Anneal from beta=0 and resample
                # In _resample, self.logZ is updated
                self.logweights = -self.betas[0] * self.fx
                self._resample(at_init=True)

            self._write_result(writer["trial"], [np.exp(self.logweights), self.walker_ancestors])
            self._write_result(writer["result"], [np.exp(self.logweights), self.walker_ancestors])

            self.istep += 1

            minidx = np.argmin(self.fx)
            self.best_x = copy.copy(self.state.x[minidx, :])
            self.best_fx = np.min(self.fx[minidx])
            self.best_istep = 0
            self.best_iwalker = 0

        next_checkpoint_step = self.istep + self.checkpoint_steps
        next_checkpoint_time = time.time() + self.checkpoint_interval

        if self.verbose:
            print("\u03b2 mean[f] Err[f] nreplica log(Z/Z0) acceptance_ratio")

        numT = len(self.betas)
        while self.Tindex < numT:
            # print(">>> Tindex = {}".format(self.Tindex))
            Tindex = self.Tindex
            beta = self.betas[self.Tindex]

            if self.nwalkers != 0:
                for _ in range(self.numsteps_for_T[Tindex]):
                    self.local_update(
                        beta,
                        extra_info_to_write=[
                            np.exp(self.logweights),
                            self.walker_ancestors,
                        ],
                    )
                    self.istep += 1
            else : #self.nwalkers == 0
                pass

            # print(">>> istep={}".format(self.istep))

            self.fx_from_reset[self.index_from_reset, :] = self.fx[:]
            self.naccepted_from_reset[self.index_from_reset, 0] = self.naccepted
            self.naccepted_from_reset[self.index_from_reset, 1] = self.ntrial
            self.naccepted = 0
            self.ntrial = 0
            self.index_from_reset += 1

            # write weights
            if writer["weight"]:
                for iwalker in range(self.nwalkers):
                    writer["weight"].write(Tindex,
                                           beta,
                                           iwalker,
                                           0,
                                           self.fx[iwalker],
                                           self.logweights[iwalker],
                                           *self.state.x[iwalker,:])

            # calculate participation ratio
            pr = self._calc_participation_ratio()
            self.pr_list[Tindex] = pr

            if Tindex == numT - 1:
                break

            dbeta = self.betas[Tindex + 1] - self.betas[Tindex]
            self.logweights += -dbeta * self.fx
            if self.index_from_reset == self.resampling_interval:
                time_sta = time.perf_counter()
                self._resample()
                time_end = time.perf_counter()
                self.timer["run"]["resampling"] += time_end - time_sta
                self.index_from_reset = 0

            self.Tindex += 1

            if self.checkpoint:
                time_now = time.time()
                if self.istep >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    print(">>> checkpoint")
                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = self.istep + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        # store final state for continuation
        if self.checkpoint:
            print(">>> store final state")
            self._save_state(self.checkpoint_file)

        if self.index_from_reset > 0:
            res = self._gather_information(self.index_from_reset)
            self._save_stats(res)

        # must close explicitly
        self._close_writer(writer)

        if self.separate_T and not self.export_combined_files:
            self._split_result_file("trial")
            self._split_result_file("result")

        if odatse.mpi.algsize() > 1:
            odatse.mpi.algcomm().barrier()

        print("complete main process : rank {:08d}/{:08d}".format(odatse.mpi.algrank(), odatse.mpi.algsize()))

    def _setup_writer(self):
        write_mode = "w" if self.mode.startswith("init") else "a"

        item_list = [
            "step",
            "walker",
            ("beta" if self.input_as_beta else "T"),
            "fx",
            *self.label_list,
            "weight",
            "ancestor",
        ]

        file_trial = DataWriter("trial.txt", mode=write_mode, item_list=item_list, combined=self.export_combined_files)
        file_result = DataWriter("result.txt", mode=write_mode, item_list=item_list, combined=self.export_combined_files)
        self._set_writer(file_trial, file_result)

        item_list_weight = [
            "Tindex",
            "beta",
            "walker",
            "idnum",
            "fx",
            "log_weight",
            *self.label_list,
        ]
        file_weight = DataWriter("weight.txt", mode=write_mode, item_list=item_list_weight, combined=self.export_combined_files)

        return { "trial": file_trial, "result": file_result, "weight": file_weight }

    def _close_writer(self, writers):
        for k, v in writers.items():
            if v:
                v.close()

    def _gather_information(self, numT: int = None) -> dict[str, np.ndarray]:
        """
        Collect and organize statistical information across all processes.

        Gathers data needed for:
          1. Free energy calculations
          2. Error estimation
          3. Population statistics
          4. Acceptance rate monitoring

        Parameters
        ----------
        numT : int, optional
            Number of temperature steps to gather data for

        Returns
        -------
        dict[str, np.ndarray]
            Contains:
              - fxs: Energy values for all walkers
              - logweights: Log of importance weights
              - ns: Number of walkers per process
              - ancestors: Genealogical tracking data
              - acceptance ratio: MC acceptance rates
        """

        if numT is None:
            numT = self.resampling_interval

        res = {}

        res["ns"] = gather_data(np.array([self.nwalkers]))
        res["fxs"] = gather_replica(self.fx_from_reset[0:numT,:], axis=1)
        res["ancestors"] = gather_replica(self.walker_ancestors)
        nacc = gather_data([self.naccepted_from_reset[0:numT,:]])
        nacc = np.sum(nacc, axis=0)
        # A temperature with no trials (e.g. numsteps_for_T == 0) would give a
        # 0/0 acceptance ratio; report 0.0 there instead of nan/inf.
        ntrials = nacc[:, 1]
        res["acceptance ratio"] = np.divide(
            nacc[:, 0], ntrials,
            out=np.zeros(nacc.shape[0], dtype=np.float64),
            where=(ntrials != 0),
        )

        fxs = res["fxs"]
        nreplicas = np.sum(res["ns"])

        betas = self.betas[self.Tindex-numT+1 : self.Tindex+1]
        lw = np.zeros((numT, nreplicas), dtype=np.float64)
        for i in range(1, numT):
            dbeta = betas[i] - betas[i-1]
            lw[i, :] = lw[i-1, :] - dbeta * fxs[i-1, :]
        res["logweights"] = lw

        return res

    def _save_stats(self, info: dict[str, np.ndarray]) -> None:
        """
        Calculate and save statistical measures from the simulation.

        Performs:
          1. Free energy calculations using weighted averages
          2. Error estimation using jackknife resampling
          3. Population size tracking
          4. Acceptance rate monitoring
          5. Partition function estimation

        Uses bias-corrected jackknife for reliable error estimates
        of weighted averages in the presence of correlations.

        Parameters
        ----------
        info : dict[str, np.ndarray]
            Dictionary containing the following keys:
              - fxs: Objective function of each walker over all processes.
              - logweights: Logarithm of weights.
              - ns: Number of walkers in each process.
              - ancestors: Ancestor (origin) of each walker.
              - acceptance ratio: Acceptance ratio for each temperature.
        """
        fxs = info["fxs"]
        numT, nreplicas = fxs.shape
        endTindex = self.Tindex + 1
        startTindex = endTindex - numT

        logweights = info["logweights"]
        logweights_max = logweights.max(axis=1).reshape(-1, 1)
        weights = np.exp(logweights - logweights_max)  # to avoid overflow

        # Bias-corrected jackknife resampling method
        fs = np.zeros((numT, nreplicas))
        fw_sum = (fxs * weights).sum(axis=1)
        w_sum = weights.sum(axis=1)
        for i in range(nreplicas):
            F = fw_sum - fxs[:, i] * weights[:, i]
            W = w_sum - weights[:, i]
            fs[:, i] = F / W
        N = fs.shape[1]
        fm = N * (fw_sum / w_sum) - (N - 1) * fs.mean(axis=1)
        ferr = np.sqrt((N - 1) * fs.var(axis=1))

        self.Fmeans[startTindex:endTindex] = fm
        self.Ferrs[startTindex:endTindex] = ferr
        self.nreplicas[startTindex:endTindex] = nreplicas
        self.acceptance_ratio[startTindex:endTindex] = info["acceptance ratio"][0:numT]

        logz = np.log(np.mean(weights, axis=1))
        logz += logweights_max.flatten()
        self.logZs[startTindex:endTindex] = self.logZ + logz
        if endTindex < len(self.betas):
            bdiff = self.betas[endTindex] - self.betas[endTindex - 1]
            w = np.exp(logweights[-1, :] - bdiff * fxs[-1, :])
            self.logZ = self.logZs[startTindex] + np.log(w.mean())

        if self.verbose:
            for iT in range(startTindex, endTindex):
                print(" ".join(map(str, [
                    self.betas[iT],
                    self.Fmeans[iT],
                    self.Ferrs[iT],
                    self.nreplicas[iT],
                    self.logZs[iT],
                    self.acceptance_ratio[iT],
                ])))

    def _resample(self, at_init: bool = False) -> None:
        """
        Perform population resampling between temperature steps.

        This is a key component of PAMC that:
          1. Gathers current population statistics (unless at_init)
          2. Calculates importance weights for the temperature change
          3. Resamples walkers based on their weights
          4. Updates population statistics and free energy estimates (unless at_init)

        Parameters
        ----------
        at_init : bool, optional
            If True, use current logweights only and skip gather/save_stats
            (used when resampling at init with anneal_from_beta0).
            self.logZ is updated in this case.

        The resampling can be done in two modes:
          - Fixed: Maintains constant population size
          - Varied: Allows population size to fluctuate based on weights

        Implementation Details:
          - Uses log-weights to prevent numerical overflow
          - Maintains walker genealogy for analysis
          - Updates free energy estimates using resampling data
          - Handles MPI communication for parallel execution
        """
        if at_init:
            logweights = gather_replica(self.logweights)
            lw_max = logweights.max()
            weights = np.exp(logweights - lw_max)
            self.logZ = np.log(weights.mean()) + lw_max
            ns = gather_data(np.array([self.nwalkers]))
        else:
            res = self._gather_information()
            self._save_stats(res)
            dbeta = self.betas[self.Tindex + 1] - self.betas[self.Tindex]
            logweights = res["logweights"][-1, :] - dbeta * res["fxs"][-1, :]
            weights = np.exp(logweights - logweights.max())
            ns = res["ns"]

        if self.fix_nwalkers:
            self._resample_fixed(weights)
            self.logweights[:] = 0.0
        else:
            offsets = np.cumsum(ns) - ns
            self._resample_varied(weights, offsets[odatse.mpi.algrank()])
            self.fx_from_reset = np.zeros((self.resampling_interval, self.nwalkers))
            self.logweights = np.zeros(self.nwalkers)

    def _resample_fixed(self, weights: np.ndarray) -> None:
        """
        Perform resampling with fixed weights.

        This method resamples the walkers based on the provided weights and updates
        the state of the algorithm accordingly.

        Parameters
        ----------
        weights : np.ndarray
            Array of weights for resampling.
        """
        resampler = odatse.util.resampling.WalkerTable(weights)
        new_index = resampler.sample(self.rng, self.nwalkers)

        states = self.statespace.gather(self.state)
        ancestors = gather_replica(self.walker_ancestors)

        self.state = self.statespace.pick(states, new_index)
        self.walker_ancestors = ancestors[new_index]

        fxs = gather_replica(self.fx)
        self.fx = fxs[new_index]

    def _resample_varied(self, weights: np.ndarray, offset: int) -> None:
        """
        Resample population allowing size variation.

        Uses Poisson resampling:
          1. Calculates expected number of copies from weights
          2. Samples actual copies using Poisson distribution
          3. Creates new population with variable size
          4. Updates all walker properties

        Parameters
        ----------
        weights : np.ndarray
            Importance weights for resampling
        offset : int
            Process-specific offset in global population
        """
        weights_sum = np.sum(weights)
        expected_numbers = (self.nreplicas[0] / weights_sum) * weights[
            offset : offset + self.nwalkers
        ]
        next_numbers = self.rng.poisson(expected_numbers)

        new_index = []
        for iwalker, num in enumerate(next_numbers):
            new_index.extend([iwalker] * num)

        new_state = self.statespace.pick(self.state, new_index)
        new_fx = self.fx[new_index]

        self.state = new_state
        self.fx = new_fx
        self.walker_ancestors = np.array(new_index)

        # keep nwalkers a plain int (np.sum returns np.int64), since downstream
        # code does isinstance(..., int) checks and uses it as an array size
        self.nwalkers = int(np.sum(next_numbers))

    def _calc_participation_ratio(self) -> float:
        """
        Calculate the participation ratio of the current walker population.
        
        The participation ratio is a measure of the effective sample size and
        indicates how evenly distributed the weights are among walkers. It is
        calculated as (sum(w))²/sum(w²), where w are the normalized weights.
        
        A value close to the total number of walkers indicates well-distributed weights,
        while a small value indicates that only a few walkers dominate the population.
        
        To avoid numerical issues with large log-weights, we normalize by subtracting
        the maximum log-weight before exponentiation.
        
        Parameters
        ----------
        None
            Uses the current state of self.logweights
            
        Returns
        -------
        float
            The participation ratio, a value between 1 and the total number of walkers
            
        Notes
        -----
        In parallel execution, this method aggregates weights across all processes
        to calculate the global participation ratio.
        """
        log_weights = gather_replica(self.logweights)
        max_log_weight = np.max(log_weights)

        # Degenerate weights (e.g. all -inf) make (log_weights - max) contain
        # nan; the guard below turns that into a finite 0.0, so suppress the
        # transient invalid-value warning here.
        with np.errstate(invalid="ignore"):
            sum_weight = np.sum(np.exp(log_weights - max_log_weight))
            sum_weight_sq = np.sum(np.exp(log_weights - max_log_weight)**2)

        # sum_weight_sq is normally >= 1 (the max element contributes exp(0)=1),
        # but degenerate weights make it 0 or nan; guard so the participation
        # ratio is a finite number rather than nan.
        pr = sum_weight ** 2 / sum_weight_sq if sum_weight_sq > 0 else 0.0

        return pr

    def _prepare(self) -> None:
        """
        Prepare the algorithm for execution.

        Initialises the run-phase timers, then for ``continue`` mode applies
        the logweight update and advances ``Tindex`` to the next temperature.
        This must run after ``_apply_state()`` has extended the beta schedule
        (done inside ``_prepare()`` dispatch) and after the timers are ready.
        """
        self.timer["run"]["submit"] = 0.0
        self.timer["run"]["resampling"] = 0.0

        if self.mode.startswith("continue"):
            Tindex = self.Tindex
            dbeta = self.betas[Tindex + 1] - self.betas[Tindex]
            self.logweights += -dbeta * self.fx
            if self.index_from_reset == self.resampling_interval:
                time_sta = time.perf_counter()
                self._resample()
                time_end = time.perf_counter()
                self.timer["run"]["resampling"] += time_end - time_sta
                self.index_from_reset = 0
            self.Tindex += 1

    def _split_result_file(self, tag):
        current_beta = -1
        header_str = ""
        idx = 0
        fwrite = None

        with open(tag + ".txt", "r") as fread:
            for line in fread:
                if line.startswith("#"):
                    header_str += line
                else:
                    data = line.split()
                    beta = data[2]
                    if beta != current_beta:
                        if fwrite:
                            fwrite.close()
                            idx += 1
                        fwrite = open(tag + f"_T{idx}.txt", "w")
                        fwrite.write(header_str)
                        current_beta = beta
                    fwrite.write(line)
            if fwrite:
                fwrite.close()

    def _post(self) -> dict:
        """
        Post-processing after the algorithm execution.

        This method consolidates the results from different temperature steps
        into single files for 'result' and 'trial'. It also gathers the best
        results from all processes and writes them to 'best_result.txt'.
        """
        best_fx = gather_data(np.array([self.best_fx]))
        best_x = gather_data(np.array([self.best_x]))
        best_istep = gather_data(np.array([self.best_istep]))
        best_iwalker = gather_data(np.array([self.best_iwalker]))

        best_rank = np.argmin(best_fx)
        if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
            with open("best_result.txt", "w") as f:
                f.write(f"nprocs = {odatse.mpi.algsize()}\n")
                f.write(f"rank = {best_rank}\n")
                f.write(f"step = {best_istep[best_rank]}\n")
                f.write(f"walker = {best_iwalker[best_rank]}\n")
                f.write(f"fx = {best_fx[best_rank]}\n")
                for label, x in zip(self.label_list, best_x[best_rank]):
                    f.write(f"{label} = {x}\n")
            print("Best Result:")
            print(f"  rank = {best_rank}")
            print(f"  step = {best_istep[best_rank]}")
            print(f"  walker = {best_iwalker[best_rank]}")
            print(f"  fx = {best_fx[best_rank]}")
            for label, x in zip(self.label_list, best_x[best_rank]):
                print(f"  {label} = {x}")

            with open("fx.txt", "w") as f:
                f.write("# $1: 1/T\n")
                f.write("# $2: mean of f(x)\n")
                f.write("# $3: standard error of f(x)\n")
                f.write("# $4: number of replicas\n")
                f.write("# $5: log(Z/Z0)\n")
                f.write("# $6: acceptance ratio\n")
                for i in range(len(self.betas)):
                    f.write(f"{self.betas[i]}")
                    f.write(f" {self.Fmeans[i]} {self.Ferrs[i]}")
                    f.write(f" {self.nreplicas[i]}")
                    f.write(f" {self.logZs[i]}")
                    f.write(f" {self.acceptance_ratio[i]}")
                    f.write("\n")

            with open("pr.txt", "w") as f:
                f.write("# $1: Tindex\n")
                f.write("# $2: 1/T\n")
                f.write("# $3: participation ratio\n")
                for i in range(len(self.betas)):
                    f.write(f"{i}")
                    f.write(f" {self.betas[i]}")
                    f.write(f" {self.pr_list[i]}")
                    f.write("\n")

        return {
            "x": best_x[best_rank],
            "fx": best_fx[best_rank],
            "nprocs": odatse.mpi.algsize(),
            "rank": best_rank,
            "step": best_istep[best_rank],
            "walker": best_iwalker[best_rank],
        }

    def _apply_state(self, data: dict, mode: str = "resume", restore_rng: bool = True) -> None:
        """Restore algorithm state from a checkpoint snapshot.

        Delegates MPI validation, RNG restore, and MC-layer fields to the
        base class, then handles PAMC-specific fields explicitly.  Simple
        scalar fields are set directly; array fields that may be shorter than
        the current schedule (``logZs``, ``Fmeans``, etc.) are written with
        slice assignment after re-initialisation.

        The logweight update, optional resampling, and ``Tindex`` increment
        for ``continue`` mode are performed in ``prepare()`` after this method
        returns, so that the run-phase timers are already initialised.

        Parameters
        ----------
        data : dict
            Snapshot previously produced by ``__getstate__``.
        mode : str
            ``"resume"`` — validate that the temperature schedule is identical
            to the one stored in *data*.
            ``"continue"`` — concatenate the stored schedule with the new one
            (the stored last beta must equal the new first beta).
        restore_rng : bool
            When *True* (default) the RNG state is restored from *data*;
            when *False* a fresh RNG state is kept (``--reset_rand`` mode).
        """
        super()._apply_state(data, mode=mode, restore_rng=restore_rng)

        # -- simple scalar fields
        self.Tindex = data["Tindex"]
        self.index_from_reset = data["index_from_reset"]
        self.nwalkers = data["nwalkers"]

        # -- temperature schedule (mode-dependent)
        if mode == "resume":
            assert np.all(data["betas"] == self.betas)
            assert data["input_as_beta"] == self.input_as_beta
            assert np.all(data["numsteps_for_T"] == self.numsteps_for_T)
            assert self.Tindex < len(self.betas)
        elif mode == "continue":
            assert data["input_as_beta"] == self.input_as_beta
            if not data["betas"][-1] == self.betas[0]:
                print("ERROR: temperature is not continuous")
                sys.exit(1)
            self.betas = np.concatenate([data["betas"], self.betas[1:]])
            self.numsteps_for_T = np.concatenate([data["numsteps_for_T"], self.numsteps_for_T[1:]])

        # -- re-initialise length-numT arrays, then overwrite with saved data
        numT = len(self.betas)
        nreplicas = odatse.mpi.algsize() * self.nwalkers

        self.logZs = np.zeros(numT)
        self.Fmeans = np.zeros(numT)
        self.Ferrs = np.zeros(numT)
        self.nreplicas = np.full(numT, nreplicas)
        self.acceptance_ratio = np.zeros(numT)
        self.pr_list = np.zeros(numT)

        self.logZ = data["logZ"]
        self.logZs[:len(data["logZs"])] = data["logZs"]
        self.logweights = data["logweights"]
        self.Fmeans[:len(data["Fmeans"])] = data["Fmeans"]
        self.Ferrs[:len(data["Ferrs"])] = data["Ferrs"]
        self.nreplicas[:len(data["nreplicas"])] = data["nreplicas"]
        self.populations = data["populations"].copy()
        self.family_lo = data["family_lo"]
        self.family_hi = data["family_hi"]
        self.fx_from_reset = data["fx_from_reset"]
        self.walker_ancestors = data["walker_ancestors"]
        self.naccepted_from_reset = data["naccepted_from_reset"]
        self.acceptance_ratio[:len(data["acceptance_ratio"])] = data["acceptance_ratio"]
        self.pr_list[:len(data["pr_list"])] = data["pr_list"]

        self.statespace.rng = self.rng
