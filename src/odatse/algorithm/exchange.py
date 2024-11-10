# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from io import open
import copy
import time
import itertools
import sys

import numpy as np

import odatse
import odatse.algorithm.montecarlo
from odatse.algorithm.montecarlo import read_Ts
from odatse.util.separateT import separateT

class Algorithm(odatse.algorithm.montecarlo.AlgorithmBase):
    """Replica Exchange Monte Carlo

    Attributes
    ==========
    x: np.ndarray
        current configuration
    inode: np.ndarray
        current configuration index (discrete parameter space)
    fx: np.ndarray
        current "Energy"
    Tindex: np.ndarray
        current "Temperature" index
    istep: int
        current step (or, the number of calculated energies)
    best_x: np.ndarray
        best configuration
    best_fx: float
        best "Energy"
    best_istep: int
        index of best configuration
    nreplica: int
        The number of replicas (= the number of procs)
    T2rep: np.ndarray
        Mapping from temperature index to replica index
    rep2T: np.ndarray
        Reverse mapping from replica index to temperature index
    exchange_direction: bool
        Parity of exchange direction
    """

    x: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    #xunit: np.ndarray
    xstep: np.ndarray

    numsteps: int
    numsteps_exchange: int

    fx: np.ndarray
    istep: int
    nreplica: int
    Tindex: np.ndarray
    rep2T: np.ndarray
    T2rep: np.ndarray

    exchange_direction: bool

    def __init__(self,
                 info: odatse.Info,
                 runner: odatse.Runner = None,
                 run_mode: str = "initial"
    ) -> None:
        """
        Initialize the Algorithm class.

        Parameters
        ----------
        info : odatse.Info
            Information object containing algorithm parameters.
        runner : odatse.Runner, optional
            Runner object for executing the algorithm.
        run_mode : str, optional
            Mode to run the algorithm in, by default "initial".
        """
        time_sta = time.perf_counter()

        info_exchange = info.algorithm["exchange"]
        nwalkers = info_exchange.get("nreplica_per_proc", 1)

        super().__init__(info=info, runner=runner, nwalkers=nwalkers, run_mode=run_mode)

        self.nreplica = self.mpisize * self.nwalkers
        self.input_as_beta, self.betas = read_Ts(info_exchange, numT=self.nreplica)

        self.numsteps = info_exchange["numsteps"]
        self.numsteps_exchange = info_exchange["numsteps_exchange"]
        time_end = time.perf_counter()
        self.timer["init"]["total"] = time_end - time_sta

    def _print_info(self) -> None:
        """
        Print information about the algorithm.
        """
        if self.mpirank == 0:
            pass
        if self.mpisize > 1:
            self.mpicomm.barrier()

    def _initialize(self) -> None:
        """
        Initialize the algorithm parameters and state.
        """
        super()._initialize()

        self.Tindex = np.arange(
            self.mpirank * self.nwalkers, (self.mpirank + 1) * self.nwalkers
        )
        self.rep2T = np.arange(self.nreplica)
        self.T2rep = np.arange(self.nreplica)

        self.exchange_direction = True
        self.istep = 0

        self._show_parameters()

    def _run(self) -> None:
        """
        Run the algorithm.
        """
        # print(">>> _run")

        if self.mode is None:
            raise RuntimeError("mode unset")

        restore_rng = not self.mode.endswith("-resetrand")

        if self.mode.startswith("init"):
            self._initialize()
        elif self.mode.startswith("resume"):
            self._load_state(self.checkpoint_file, mode="resume", restore_rng=restore_rng)
        elif self.mode.startswith("continue"):
            self._load_state(self.checkpoint_file, mode="continue", restore_rng=restore_rng)
        else:
            raise RuntimeError("unknown mode {}".format(self.mode))

        beta = self.betas[self.Tindex]

        if self.mode.startswith("init"):
            # first step
            self._evaluate()

            file_trial = open("trial.txt", "w")
            self._write_result_header(file_trial)
            self._write_result(file_trial)

            file_result = open("result.txt", "w")
            self._write_result_header(file_result)
            self._write_result(file_result)

            self.istep += 1

            minidx = np.argmin(self.fx)
            self.best_x = copy.copy(self.x[minidx, :])
            self.best_fx = np.min(self.fx[minidx])
            self.best_istep = 0
            self.best_iwalker = 0
        else:
            file_trial = open("trial.txt", "a")
            file_result = open("result.txt", "a")

        next_checkpoint_step = self.istep + self.checkpoint_steps
        next_checkpoint_time = time.time() + self.checkpoint_interval

        while self.istep < self.numsteps:
            # print(">>> istep={}".format(self.istep))

            # Exchange
            if self.istep % self.numsteps_exchange == 0:
                # print(">>> do exchange")
                time_sta = time.perf_counter()
                if self.nreplica > 1:
                    self._exchange(self.exchange_direction)
                if self.nreplica > 2:
                    self.exchange_direction = not self.exchange_direction
                time_end = time.perf_counter()
                self.timer["run"]["exchange"] += time_end - time_sta
                beta = self.betas[self.Tindex]

            # print(">>> call local_update")
            self.local_update(beta, file_trial, file_result)
            self.istep += 1

            if self.checkpoint:
                time_now = time.time()
                if self.istep >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = self.istep + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        file_result.close()
        file_trial.close()
        print("complete main process : rank {:08d}/{:08d}".format(self.mpirank, self.mpisize))

        # store final state for continuation
        if self.checkpoint:
            # print(">>> store final state")
            self._save_state(self.checkpoint_file)

    def _exchange(self, direction: bool) -> None:
        """
        Try to exchange temperatures.

        Parameters
        ----------
        direction : bool
            Direction of the exchange.
        """
        if self.nwalkers == 1:
            self.__exchange_single_walker(direction)
        else:
            self.__exchange_multi_walker(direction)

    def __exchange_single_walker(self, direction: bool) -> None:
        """
        Exchange temperatures for a single walker.

        Parameters
        ----------
        direction : bool
            Direction of the exchange.
        """
        if self.mpisize > 1:
            self.mpicomm.Barrier()
        if direction:
            if self.Tindex[0] % 2 == 0:
                other_index = self.Tindex[0] + 1
                is_main = True
            else:
                other_index = self.Tindex[0] - 1
                is_main = False
        else:
            if self.Tindex[0] % 2 == 0:
                other_index = self.Tindex[0] - 1
                is_main = False
            else:
                other_index = self.Tindex[0] + 1
                is_main = True

        ibuf = np.zeros(1, dtype=np.int64)
        fbuf = np.zeros(1, dtype=np.float64)

        if 0 <= other_index < self.nreplica:
            other_rank = self.T2rep[other_index]
            if is_main:
                self.mpicomm.Recv(fbuf, source=other_rank, tag=1)
                other_fx = fbuf[0]
                beta = self.betas[self.Tindex[0]]
                other_beta = self.betas[self.Tindex[0] + 1]
                logp = (other_beta - beta) * (other_fx - self.fx[0])
                if logp >= 0.0 or self.rng.rand() < np.exp(logp):
                    ibuf[0] = self.Tindex
                    self.mpicomm.Send(ibuf, dest=other_rank, tag=2)
                    self.Tindex[0] += 1
                else:
                    ibuf[0] = self.Tindex + 1
                    self.mpicomm.Send(ibuf, dest=other_rank, tag=2)
            else:
                fbuf[0] = self.fx[0]
                self.mpicomm.Send(fbuf, dest=other_rank, tag=1)
                self.mpicomm.Recv(ibuf, source=other_rank, tag=2)
                self.Tindex[0] = ibuf[0]

        self.mpicomm.Barrier()
        if self.mpirank == 0:
            self.T2rep[self.Tindex[0]] = self.mpirank
            for other_rank in range(1, self.nreplica):
                self.mpicomm.Recv(ibuf, source=other_rank, tag=0)
                self.T2rep[ibuf[0]] = other_rank
        else:
            ibuf[0] = self.Tindex
            self.mpicomm.Send(ibuf, dest=0, tag=0)
        self.mpicomm.Bcast(self.T2rep, root=0)

    def __exchange_multi_walker(self, direction: bool) -> None:
        """
        Exchange temperatures for multiple walkers.

        Parameters
        ----------
        direction : bool
            Direction of the exchange.
        """
        comm = self.mpicomm
        if self.mpisize > 1:
            fx_all = comm.allgather(self.fx)
            fx_all = np.array(fx_all).flatten()
        else:
            fx_all = self.fx

        rep2T_diff = []
        T2rep_diff = []

        for irep in range(
            self.mpirank * self.nwalkers, (self.mpirank + 1) * self.nwalkers
        ):
            iT = self.rep2T[irep]
            if iT % 2 != 0:
                continue
            jT = iT + 1 if direction else iT - 1
            if jT < 0 or jT == self.nreplica:
                continue
            jrep = self.T2rep[jT]
            fdiff = fx_all[jrep] - fx_all[irep]
            bdiff = self.betas[jT] - self.betas[iT]
            logp = fdiff * bdiff
            if logp >= 0.0 or self.rng.rand() < np.exp(logp):
                rep2T_diff.append((irep, jT))  # this means self.rep2T[irep] = jT
                rep2T_diff.append((jrep, iT))
                T2rep_diff.append((iT, jrep))
                T2rep_diff.append((jT, irep))

        if self.mpisize > 1:
            rep2T_diff = comm.allgather(rep2T_diff)
            rep2T_diff = list(itertools.chain.from_iterable(rep2T_diff))  # flatten
            T2rep_diff = comm.allgather(T2rep_diff)
            T2rep_diff = list(itertools.chain.from_iterable(T2rep_diff))  # flatten

        for diff in rep2T_diff:
            self.rep2T[diff[0]] = diff[1]
        for diff in T2rep_diff:
            self.T2rep[diff[0]] = diff[1]
        self.Tindex = self.rep2T[
            self.mpirank * self.nwalkers : (self.mpirank + 1) * self.nwalkers
        ]

    def _prepare(self) -> None:
        """
        Prepare the algorithm for execution.
        """
        self.timer["run"]["submit"] = 0.0
        self.timer["run"]["exchange"] = 0.0

    def _post(self) -> None:
        """
        Post-process the results of the algorithm.
        """
        Ts = self.betas if self.input_as_beta else 1.0 / self.betas
        if self.mpirank == 0:
            print(f"start separateT {self.mpirank}")
            sys.stdout.flush()
        separateT(
            Ts=Ts,
            nwalkers=self.nwalkers,
            output_dir=self.output_dir,
            comm=self.mpicomm,
            use_beta=self.input_as_beta,
            buffer_size=10000,
        )
        if self.mpisize > 1:
            # NOTE:
            # ``gather`` seems not to work with many processes (say, 32) in some MPI implementation.
            # ``Gather`` and ``allgather`` seem to work fine.
            # Since the performance is not so important here, we use ``allgather`` for simplicity.
            best_fx = self.mpicomm.allgather(self.best_fx)
            best_x = self.mpicomm.allgather(self.best_x)
            best_istep = self.mpicomm.allgather(self.best_istep)
            best_iwalker = self.mpicomm.allgather(self.best_iwalker)
        else:
            best_fx = [self.best_fx]
            best_x = [self.best_x]
            best_istep = [self.best_istep]
            best_iwalker = [self.best_iwalker]

        best_rank = np.argmin(best_fx)

        if self.mpirank == 0:
            with open("best_result.txt", "w") as f:
                f.write(f"nprocs = {self.nreplica}\n")
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

        return {
            "x": best_x[best_rank],
            "fx": best_fx[best_rank],
            "nprocs": self.nreplica,
            "rank": best_rank,
            "step": best_istep[best_rank],
            "walker": best_iwalker[best_rank],
        }

    def _save_state(self, filename) -> None:
        """
        Save the current state of the algorithm.

        Parameters
        ----------
        filename : str
            The name of the file to save the state to.
        """
        data = {
            #-- _algorithm
            "mpisize": self.mpisize,
            "mpirank": self.mpirank,
            "rng": self.rng.get_state(),
            "timer": self.timer,
            "info": self.info,
            #-- montecarlo
            "x": self.x,
            "fx": self.fx,
            "inode": self.inode,
            "istep": self.istep,
            "best_x": self.best_x,
            "best_fx": self.best_fx,
            "best_istep": self.best_istep,
            "best_iwalker": self.best_iwalker,
            "naccepted": self.naccepted,
            "ntrial": self.ntrial,
            #-- exchange
            "nreplica": self.nreplica,
            "Tindex": self.Tindex,
            "rep2T": self.rep2T,
            "T2rep": self.T2rep,
            "exchange_direction": self.exchange_direction,
        }
        self._save_data(data, filename)

    def _load_state(self, filename, mode="resume", restore_rng=True):
        """
        Load the state of the algorithm from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the state from.
        mode : str, optional
            The mode to load the state in, by default "resume".
        restore_rng : bool, optional
            Whether to restore the random number generator state, by default True.
        """
        data = self._load_data(filename)
        if not data:
            print("ERROR: Load status file failed")
            sys.exit(1)

        #-- _algorithm
        assert self.mpisize == data["mpisize"]
        assert self.mpirank == data["mpirank"]

        if restore_rng:
            self.rng = np.random.RandomState()
            self.rng.set_state(data["rng"])
        self.timer = data["timer"]

        info = data["info"]
        self._check_parameters(info)

        #-- montecarlo
        self.x = data["x"]
        self.fx = data["fx"]
        self.inode = data["inode"]

        self.istep = data["istep"]

        self.best_x = data["best_x"]
        self.best_fx = data["best_fx"]
        self.best_istep = data["best_istep"]
        self.best_iwalker = data["best_iwalker"]

        self.naccepted = data["naccepted"]
        self.ntrial = data["ntrial"]

        #-- exchange
        assert self.nreplica == data["nreplica"]

        self.Tindex = data["Tindex"]
        self.rep2T = data["rep2T"]
        self.T2rep = data["T2rep"]
        self.exchange_direction = data["exchange_direction"]
