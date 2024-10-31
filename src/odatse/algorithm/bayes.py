# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List
import time
import shutil
import copy
from pathlib import Path

import physbo
import numpy as np

import odatse
import odatse.domain

class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    A class to represent the Bayesian optimization algorithm.

    Attributes
    ----------
    mesh_list : np.ndarray
        The mesh grid list.
    label_list : List[str]
        The list of labels.
    random_max_num_probes : int
        The maximum number of random probes.
    bayes_max_num_probes : int
        The maximum number of Bayesian probes.
    score : str
        The scoring method.
    interval : int
        The interval for Bayesian optimization.
    num_rand_basis : int
        The number of random basis.
    xopt : np.ndarray
        The optimal solution.
    best_fx : List[float]
        The list of best function values.
    best_action : List[int]
        The list of best actions.
    fx_list : List[float]
        The list of function values.
    param_list : List[np.ndarray]
        The list of parameters.
    """

    def __init__(self, info: odatse.Info, runner: odatse.Runner = None, domain = None, run_mode: str = "initial") -> None:
        """
        Constructs all the necessary attributes for the Algorithm object.

        Parameters
        ----------
        info : odatse.Info
            The information object.
        runner : odatse.Runner, optional
            The runner object (default is None).
        domain : optional
            The domain object (default is None).
        run_mode : str, optional
            The run mode (default is "initial").
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        info_param = info.algorithm.get("param", {})
        info_bayes = info.algorithm.get("bayes", {})

        for key in ("random_max_num_probes", "bayes_max_num_probes", "score", "interval", "num_rand_basis"):
            if key in info_param and key not in info_bayes:
                print(f"WARNING: algorithm.param.{key} is deprecated. Use algorithm.bayes.{key} .")
                info_bayes[key] = info_param[key]

        self.random_max_num_probes = info_bayes.get("random_max_num_probes", 20)
        self.bayes_max_num_probes = info_bayes.get("bayes_max_num_probes", 40)
        self.score = info_bayes.get("score", "TS")
        self.interval = info_bayes.get("interval", 5)
        self.num_rand_basis = info_bayes.get("num_rand_basis", 5000)

        if self.mpirank == 0:
            print("# parameter")
            print(f"random_max_num_probes = {self.random_max_num_probes}")
            print(f"bayes_max_num_probes = {self.bayes_max_num_probes}")
            print(f"score = {self.score}")
            print(f"interval = {self.interval}")
            print(f"num_rand_basis = {self.num_rand_basis}")

        if domain and isinstance(domain, odatse.domain.MeshGrid):
            self.domain = domain
        else:
            self.domain = odatse.domain.MeshGrid(info)
        self.mesh_list = np.array(self.domain.grid)

        X_normalized = physbo.misc.centering(self.mesh_list[:, 1:])
        comm = self.mpicomm if self.mpisize > 1 else None
        self.policy = physbo.search.discrete.policy(test_X=X_normalized, comm=comm)

        if "seed" in info.algorithm:
            seed = info.algorithm["seed"]
            self.policy.set_seed(seed)

        self.file_history = "history.npz"
        self.file_training = "training.npz"
        self.file_predictor = "predictor.dump"

    def _initialize(self):
        """
        Initializes the algorithm parameters and timers.
        """
        self.istep = 0
        self.param_list = []
        self.fx_list = []
        self.timer["run"]["random_search"] = 0.0
        self.timer["run"]["bayes_search"] = 0.0

        self._show_parameters()

    def _run(self) -> None:
        """
        Runs the Bayesian optimization process.
        """
        runner = self.runner
        mesh_list = self.mesh_list

        class simulator:
            def __call__(self, action: np.ndarray) -> float:
                """
                Simulates the function evaluation for a given action.

                Parameters
                ----------
                action : np.ndarray
                    The action to be evaluated.

                Returns
                -------
                float
                    The negative function value.
                """
                a = int(action[0])
                args = (a, 0)
                x = mesh_list[a, 1:]
                fx = runner.submit(x, args)
                fx_list.append(fx)
                param_list.append(mesh_list[a])
                return -fx

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

        fx_list = self.fx_list
        param_list = self.param_list

        if self.mode.startswith("init"):
            time_sta = time.perf_counter()
            res = self.policy.random_search(
                max_num_probes=self.random_max_num_probes, simulator=simulator()
            )
            time_end = time.perf_counter()
            self.timer["run"]["random_search"] = time_end - time_sta

            if self.checkpoint:
                self._save_state(self.checkpoint_file)
        else:
            if self.istep >= self.bayes_max_num_probes:
                res = copy.deepcopy(self.policy.history)

        next_checkpoint_step = self.istep + self.checkpoint_steps
        next_checkpoint_time = time.time() + self.checkpoint_interval

        while self.istep < self.bayes_max_num_probes:
            intv = 0 if self.istep % self.interval == 0 else -1

            time_sta = time.perf_counter()
            res = self.policy.bayes_search(
                max_num_probes=1,
                simulator=simulator(),
                score=self.score,
                interval=intv,
                num_rand_basis=self.num_rand_basis,
            )
            time_end = time.perf_counter()
            self.timer["run"]["bayes_search"] += time_end - time_sta

            self.istep += 1

            if self.checkpoint:
                time_now = time.time()
                if self.istep >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    self.fx_list = fx_list
                    self.param_list = param_list

                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = self.istep + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        self.best_fx, self.best_action = res.export_all_sequence_best_fx()
        self.xopt = mesh_list[int(self.best_action[-1]), 1:]
        self.fx_list = fx_list
        self.param_list = param_list

        if self.checkpoint:
            self._save_state(self.checkpoint_file)

    def _prepare(self) -> None:
        """
        Prepares the algorithm for execution.
        """
        pass

    def _post(self) -> None:
        """
        Finalizes the algorithm execution and writes the results to a file.
        """
        label_list = self.label_list
        if self.mpirank == 0:
            with open("BayesData.txt", "w") as file_BD:
                file_BD.write("#step")
                for label in label_list:
                    file_BD.write(f" {label}")
                file_BD.write(" fx")
                for label in label_list:
                    file_BD.write(f" {label}_action")
                file_BD.write(" fx_action\n")

                for step, fx in enumerate(self.fx_list):
                    file_BD.write(str(step))
                    best_idx = int(self.best_action[step])
                    for v in self.mesh_list[best_idx][1:]:
                        file_BD.write(f" {v}")
                    file_BD.write(f" {-self.best_fx[step]}")

                    for v in self.param_list[step][1:]:
                        file_BD.write(f" {v}")
                    file_BD.write(f" {fx}\n")
            print("Best Solution:")
            for x, y in zip(label_list, self.xopt):
                print(x, "=", y)
        return {"x": self.xopt, "fx": self.best_fx}

    def _save_state(self, filename):
        """
        Saves the current state of the algorithm to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the state.
        """
        data = {
            "mpisize": self.mpisize,
            "mpirank": self.mpirank,
            "rng": self.rng.get_state(),
            "timer": self.timer,
            "info": self.info,
            "istep": self.istep,
            "param_list": self.param_list,
            "fx_list": self.fx_list,
            "file_history": self.file_history,
            "file_training": self.file_training,
            "file_predictor": self.file_predictor,
            "random_number": np.random.get_state(),
        }
        self._save_data(data, filename)

        self.policy.save(file_history=Path(self.output_dir, self.file_history),
                         file_training=Path(self.output_dir, self.file_training),
                         file_predictor=Path(self.output_dir, self.file_predictor))

    def _load_state(self, filename, mode="resume", restore_rng=True):
        """
        Loads the state of the algorithm from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the state from.
        mode : str, optional
            The mode to load the state (default is "resume").
        restore_rng : bool, optional
            Whether to restore the random number generator state (default is True).
        """
        data = self._load_data(filename)
        if not data:
            print("ERROR: Load status file failed")
            sys.exit(1)

        assert self.mpisize == data["mpisize"]
        assert self.mpirank == data["mpirank"]

        if restore_rng:
            self.rng = np.random.RandomState()
            self.rng.set_state(data["rng"])
            np.random.set_state(data["random_number"])
        self.timer = data["timer"]

        info = data["info"]
        self._check_parameters(info)

        self.istep = data["istep"]
        self.param_list = data["param_list"]
        self.fx_list = data["fx_list"]

        self.policy.load(file_history=Path(self.output_dir, self.file_history),
                         file_training=Path(self.output_dir, self.file_training),
                         file_predictor=Path(self.output_dir, self.file_predictor))