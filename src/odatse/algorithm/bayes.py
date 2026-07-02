# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional, TYPE_CHECKING
import time
import copy
from pathlib import Path
import sys

import physbo
import numpy as np

import odatse
import odatse.domain

if TYPE_CHECKING:
    from mpi4py import MPI

class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    A class to represent the Bayesian optimization algorithm.

    Attributes
    ----------
    mesh_list : np.ndarray
        The mesh grid list.
    label_list : list[str]
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
    best_fx : list[float]
        The list of best function values.
    best_action : list[int]
        The list of best actions.
    fx_list : list[float]
        The list of function values.
    param_list : list[np.ndarray]
        The list of parameters.
    """

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        domain=None,
        run_mode: str = "initial",
    ) -> None:
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

        if not odatse.mpi.run_on_algorithm():
            return

        info_bayes = info.algorithm.get("bayes", {})

        # CHECK: deprecated parameters
        info_param = info.algorithm.get("param", {})
        for key in ("random_max_num_probes", "bayes_max_num_probes", "score", "interval", "num_rand_basis"):
            if key in info_param and key not in info_bayes:
                print(f"WARNING: algorithm.param.{key} is deprecated. Use algorithm.bayes.{key} .")
                info_bayes[key] = info_param[key]

        self.random_max_num_probes = info_bayes.get("random_max_num_probes", 20)
        self.bayes_max_num_probes = info_bayes.get("bayes_max_num_probes", 40)
        self.score = info_bayes.get("score", "TS")
        self.interval = info_bayes.get("interval", 5)
        self.num_rand_basis = info_bayes.get("num_rand_basis", 5000)

        if odatse.mpi.algrank() == 0:
            print("# parameter")
            print(f"random_max_num_probes = {self.random_max_num_probes}")
            print(f"bayes_max_num_probes = {self.bayes_max_num_probes}")
            print(f"score = {self.score}")
            print(f"interval = {self.interval}")
            print(f"num_rand_basis = {self.num_rand_basis}")

        if domain and isinstance(domain, odatse.domain.MeshGrid):
            self.domain = domain
        else:
            self.domain = odatse.domain.MeshGrid(info, rng=self.rng)
        self.mesh_list = np.array(self.domain.grid)

        X_normalized = physbo.misc.centering(self.mesh_list[:, 1:])
        comm = odatse.mpi.algcomm()

        if physbo.__version__ < "3":
            self.policy = physbo.search.discrete.policy(test_X=X_normalized, comm=comm)
        else:
            self.policy = physbo.search.discrete.Policy(test_X=X_normalized, comm=comm)

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

    def _prepare(self) -> None:
        pass

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

        # dispatch は _prepare() が処理済み
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

    def _post(self) -> dict:
        """
        Finalizes the algorithm execution and writes the results to a file.
        """
        label_list = self.label_list
        if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
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

    # Bayes-specific fields (simple getattr/setattr).
    _checkpoint_attrs: list[str] = ["istep", "param_list", "fx_list"]

    def __getstate__(self) -> dict:
        """Return a checkpoint snapshot, extending the base with the global RNG."""
        state = super().__getstate__()
        state["random_number"] = np.random.get_state()
        return state

    def _save_state(self, filename):
        """Save the current state of the algorithm to a file.

        Extends the base implementation with physbo policy files.
        The pickle snapshot (including the global numpy RNG captured in
        ``__getstate__``) is written by ``super()._save_state()``.
        """
        super()._save_state(filename)
        self.policy.save(file_history=Path(self.output_dir, self.file_history),
                         file_training=Path(self.output_dir, self.file_training),
                         file_predictor=Path(self.output_dir, self.file_predictor))

    def _apply_state(self, data: dict, mode: str = "resume", restore_rng: bool = True) -> None:
        """Restore algorithm state from a checkpoint snapshot.

        Delegates MPI validation, timer restore, parameter check, and the
        instance RNG restore to the base class; additionally restores the
        global numpy RNG used by physbo, applies the Bayes-specific fields,
        then loads the physbo policy from the saved files.

        ``_load_state()`` is not overridden; the base class implementation
        calls ``self._apply_state()`` (this method), so policy files are
        loaded automatically when resuming from a checkpoint.

        Parameters
        ----------
        data : dict
            Snapshot previously produced by ``__getstate__``.
        mode : str
            ``"resume"`` or ``"continue"``; forwarded to the base class.
            Bayes optimisation does not distinguish between the two modes.
        restore_rng : bool
            When *True* (default) both RNG states are restored from *data*.
        """
        super()._apply_state(data, mode=mode, restore_rng=restore_rng)
        if restore_rng:
            np.random.set_state(data["random_number"])
        for attr in Algorithm._checkpoint_attrs:
            setattr(self, attr, data[attr])
        self.policy.load(file_history=Path(self.output_dir, self.file_history),
                         file_training=Path(self.output_dir, self.file_training),
                         file_predictor=Path(self.output_dir, self.file_predictor))
