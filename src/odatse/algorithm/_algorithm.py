# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


from abc import ABCMeta, abstractmethod
from enum import IntEnum
import time
import os
import sys
import pathlib
import pickle
import shutil
import copy

import numpy as np

import odatse
import odatse.util.limitation
from odatse import exception, mpi

# for type hints
from pathlib import Path
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from mpi4py import MPI

class AlgorithmStatus(IntEnum):
    """Enumeration for the status of the algorithm."""
    INIT = 1
    PREPARE = 2
    RUN = 3

class AlgorithmBase(metaclass=ABCMeta):
    """Base class for algorithms, providing common functionality and structure."""

    rng: np.random.RandomState
    dimension: int
    label_list: list[str]
    runner: Optional[odatse.Runner]

    root_dir: Path
    output_dir: Path
    proc_dir: Path

    timer: dict[str, dict]

    status: AlgorithmStatus = AlgorithmStatus.INIT
    mode: Optional[str] = None

    @abstractmethod
    def __init__(
            self,
            info: odatse.Info,
            runner: Optional[odatse.Runner] = None,
            run_mode: str = "initial",
    ) -> None:
        """
        Initialize the algorithm with the given information and runner.

        Parameters
        ----------
        info : Info
            Information object containing algorithm and base parameters.
        runner : Runner (optional)
            Optional runner object to execute the algorithm.
        run_mode : str
            Mode in which the algorithm should run.
        """
        self.timer = {"init": {}, "prepare": {}, "run": {}, "post": {}}
        self.timer["init"]["total"] = 0.0
        self.status = AlgorithmStatus.INIT
        self.mode = run_mode.lower()

        # keep copy of parameters
        self.info = copy.deepcopy(info.algorithm)

        self.dimension = info.algorithm.get("dimension") or info.base.get("dimension")
        if not self.dimension:
            raise ValueError("ERROR: dimension is not defined")

        if "label_list" in info.algorithm:
            label = info.algorithm["label_list"]
            if len(label) != self.dimension:
                raise ValueError(f"ERROR: length of label_list and dimension do not match ({len(label)} != {self.dimension})")
            self.label_list = label
        else:
            self.label_list = [f"x{d+1}" for d in range(self.dimension)]

        # initialize random number generator
        self.__init_rng(info)

        # directories
        self.root_dir = info.base["root_dir"]
        self.output_dir = info.base["output_dir"]
        self.proc_dir = self.output_dir / str(odatse.mpi.rank())
        # create directory for each rank in case every rank has some output
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        # Some cache of the filesystem may delay making a dictionary
        # especially when mkdir just after removing the old one
        while not self.proc_dir.is_dir():
            time.sleep(0.1)
        if odatse.mpi.algcomm() is not None and odatse.mpi.algsize() > 1:
            odatse.mpi.algcomm().Barrier()

        # checkpointing
        self.checkpoint = info.algorithm.get("checkpoint", False)
        self.checkpoint_file = info.algorithm.get("checkpoint_file", "status.pickle")
        self.checkpoint_steps = info.algorithm.get("checkpoint_steps", 65536*256)  # large number
        self.checkpoint_interval = info.algorithm.get("checkpoint_interval", 86400*360)  # longer enough

        # runner
        if runner is not None:
            self.set_runner(runner)

    def __init_rng(self, info: odatse.Info) -> None:
        """
        Initialize the random number generator.

        Parameters
        ----------
        info : Info
            Information object containing algorithm parameters.
        """
        seed = info.algorithm.get("seed", None)
        seed_delta = info.algorithm.get("seed_delta", 314159)

        if seed is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = np.random.RandomState(seed + odatse.mpi.rank() * seed_delta)

    def set_runner(self, runner: odatse.Runner) -> None:
        """
        Set the runner for the algorithm.

        Parameters
        ----------
        runner : Runner
            Runner object to execute the algorithm.
        """
        self.runner = runner

    def prepare(self) -> None:
        """
        Prepare the algorithm for execution.
        """
        if self.runner is None:
            msg = "Runner is not assigned"
            raise RuntimeError(msg)
        self._prepare()
        self.status = AlgorithmStatus.PREPARE

    @abstractmethod
    def _prepare(self) -> None:
        """Abstract method to be implemented by subclasses for preparation steps."""
        pass

    def run(self) -> None:
        """
        Run the algorithm.
        """
        if self.status < AlgorithmStatus.PREPARE:
            msg = "algorithm has not prepared yet"
            raise RuntimeError(msg)
        original_dir = os.getcwd()
        os.chdir(self.proc_dir)
        self.runner.prepare(self.proc_dir)
        self._run()
        self.runner.post()
        os.chdir(original_dir)
        self.status = AlgorithmStatus.RUN

    @abstractmethod
    def _run(self) -> None:
        """Abstract method to be implemented by subclasses for running steps."""
        pass

    def post(self) -> dict:
        """
        Perform post-processing after the algorithm has run.

        Returns
        -------
        dict
            Dictionary containing post-processing results.
        """
        if self.status < AlgorithmStatus.RUN:
            msg = "algorithm has not run yet"
            raise RuntimeError(msg)
        original_dir = os.getcwd()
        os.chdir(self.output_dir)
        result = self._post()
        os.chdir(original_dir)
        return result

    @abstractmethod
    def _post(self) -> dict:
        """Abstract method to be implemented by subclasses for post-processing steps."""
        pass

    def main(self):
        """
        Main method to execute the algorithm.
        """
        if odatse.mpi.algrank() is not None: # master branch, run solver
            time_sta = time.perf_counter()
            self.prepare()
            time_end = time.perf_counter()
            self.timer["prepare"]["total"] = time_end - time_sta
            if odatse.mpi.algsize() is not None and odatse.mpi.algsize() > 1:
                odatse.mpi.algcomm().Barrier()

            time_sta = time.perf_counter()
            self.run()
            time_end = time.perf_counter()
            self.timer["run"]["total"] = time_end - time_sta
            print("end of run")
            if odatse.mpi.algsize() is not None and odatse.mpi.algsize() > 1:
                odatse.mpi.algcomm().Barrier()

            if odatse.mpi.solsize() > 1: # signal workers to finish running
                odatse.mpi.solcomm().bcast(None, root=0)

            time_sta = time.perf_counter()
            result = self.post()
            time_end = time.perf_counter()
            self.timer["post"]["total"] = time_end - time_sta

            if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
                self.write_timer(self.proc_dir / "time.log")
            return result
        else: # worker branch, enter waiting state
            if odatse.mpi.solsize() > 1:
                self.runner.solver.worker_loop()
            return None

    def write_timer(self, filename: Path):
        """
        Write the timing information to a file.

        Parameters
        ----------
        filename : Path
            Path to the file where timing information will be written.
        """
        with open(filename, "w") as fw:
            fw.write("#in units of seconds\n")

            def output_file(type):
                d = self.timer[type]
                fw.write("#{}\n total = {}\n".format(type, d["total"]))
                for key, t in d.items():
                    if key == "total":
                        continue
                    fw.write(" - {} = {}\n".format(key, t))

            output_file("init")
            output_file("prepare")
            output_file("run")
            output_file("post")

    def _save_data(self, data, filename="state.pickle", ngen=3) -> None:
        """
        Save data to a file with versioning.

        Parameters
        ----------
        data
            Data to be saved.
        filename
            Name of the file to save the data.
        ngen : int, default: 3
            Number of generations for versioning.
        """
        try:
            fn = Path(filename + ".tmp")
            with open(fn, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print("ERROR: {}".format(e))
            sys.exit(1)

        for idx in range(ngen-1, 0, -1):
            fn_from = Path(filename + "." + str(idx))
            fn_to = Path(filename + "." + str(idx+1))
            if fn_from.exists():
                shutil.move(fn_from, fn_to)
        if ngen > 0:
            if Path(filename).exists():
                fn_to = Path(filename + "." + str(1))
                shutil.move(Path(filename), fn_to)
        shutil.move(Path(filename + ".tmp"), Path(filename))
        print("save_state: write to {}".format(filename))

    def _load_data(self, filename="state.pickle") -> dict:
        """
        Load data from a file.

        Parameters
        ----------
        filename
            Name of the file to load the data from.

        Returns
        -------
        dict
            Dictionary containing the loaded data.
        """
        if Path(filename).exists():
            try:
                fn = Path(filename)
                with open(fn, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print("ERROR: {}".format(e))
                sys.exit(1)
            print("load_state: load from {}".format(filename))
        else:
            print("ERROR: file {} not exist.".format(filename))
            data = {}
        return data

    def _show_parameters(self):
        """
        Show the parameters of the algorithm.
        """
        if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
            info = flatten_dict(self.info)
            for k, v in info.items():
                print("{:16s}: {}".format(k, v))

    def _check_parameters(self, param=None):
        """
        Check the parameters of the algorithm against previous parameters.

        Parameters
        ----------
        param (optional)
            Previous parameters to check against.
        """
        info = flatten_dict(self.info)
        info_prev = flatten_dict(param)

        for k,v in info.items():
            w = info_prev.get(k, None)
            if v != w:
                if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
                    print("WARNING: parameter {} changed from {} to {}".format(k, w, v))
            if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
                print("{:16s}: {}".format(k, v))

# utility
def flatten_dict(d, parent_key="", separator="."):
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d
        Dictionary to flatten.
    parent_key : str, default : ""
        Key for the parent dictionary.
    separator : str, default : "."
        Separator to use between keys.

    Returns
    -------
    dict
        Flattened dictionary.
    """
    items = []
    if d:
        for key_, val in d.items():
            key = parent_key + separator + key_ if parent_key else key_
            if isinstance(val, dict):
                items.extend(flatten_dict(val, key, separator=separator).items())
            else:
                items.append((key, val))
    return dict(items)
