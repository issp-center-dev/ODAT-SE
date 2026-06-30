# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Optional

from pathlib import Path
from io import open
import numpy as np
import os
import sys
import time

import odatse
import odatse.domain
from ._algorithm import AlgorithmBase



class Algorithm(AlgorithmBase):
    """
    Algorithm class for the data analysis framework.
    Inherits from odatse.algorithm.AlgorithmBase.
    """
    #mesh_list: List[Union[int, float]]

    def __init__(self,
                 info: odatse.Info,
                 runner: Optional[odatse.Runner] = None,
                 run_mode: str = "initial",
                 iterator = None,
    ) -> None:
        """
        Initialize the Algorithm instance.

        Parameters
        ----------
        info : Info
            Information object containing algorithm parameters.
        runner : Runner
            Optional runner object for submitting tasks.
        run_mode : str
            Mode to run the algorithm, defaults to "initial".
        iterator : Iterator
            Iterator object.
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        self._iter = iterator

        self.colormap_file = info.algorithm.get("colormap", "ColorMap.txt")
        self.local_colormap_file = Path(self.colormap_file).name + ".tmp"

    def _initialize(self) -> None:
        """
        Initialize the algorithm parameters and timer.
        """
        #self.fx_list = []
        self.results = []

        self.opt_fx = np.inf
        self.opt_mesh = None

        self.timer["run"]["submit"] = 0.0
        self._show_parameters()

    def _prepare(self) -> None:
        pass

    def _run(self) -> None:
        """
        Execute the main algorithm process.
        """
        # dispatch は prepare() が処理済み

        # local colormap file
        fp = open(self.local_colormap_file, "a")
        if self.mode.startswith("init"):
            fp.write("#" + " ".join(self.label_list) + " fval\n")

        #iterations = len(self.mesh_list)
        #istart = len(self.fx_list)
        istart = 0

        next_checkpoint_step = istart + self.checkpoint_steps
        next_checkpoint_time = time.time() + self.checkpoint_interval

        for icount, (idx, coord) in enumerate(self._iter):

            print("Iteration : {}/{}".format(icount+1, self._iter.size()))
            args = (idx, 0)
            x = np.array(coord)

            time_sta = time.perf_counter()
            fx = self.runner.submit(x, args)
            if isinstance(fx, np.ndarray) and fx.size == 1:
                fx = fx[0]
            time_end = time.perf_counter()
            self.timer["run"]["submit"] += time_end - time_sta

            #self.fx_list.append([mesh[0], fx])
            #self.fx_list.append([idx, fx])
            self.results.append([idx, coord, fx])

            # write to local colormap file
            fp.write(" ".join(
                map(lambda v: "{:8f}".format(v), (*x, fx))
            ) + "\n")

            # update optimal value
            if fx < self.opt_fx:
                self.opt_fx = fx
                self.opt_mesh = (idx, coord)

            # checkpointing
            if self.checkpoint:
                time_now = time.time()
                if icount+1 >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = icount + 1 + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        # close local colormap file
        fp.close()

        if not np.isinf(self.opt_fx):
            print(f"[{odatse.mpi.algrank()}] minimum_value: {self.opt_fx:12.8e} at {self.opt_mesh[0]} (mesh {self.opt_mesh[1]})")

        # self._output_results()

        # if Path(self.local_colormap_file).exists():
        #     os.remove(Path(self.local_colormap_file))

        print("complete main process : rank {:08d}/{:08d}".format(odatse.mpi.algrank(), odatse.mpi.algsize()))

    def _output_results(self, results, opt_fx, opt_mesh):
        """
        Output the results to the colormap file.
        """

        print("Make ColorMap")
        time_sta = time.perf_counter()

        with open(self.colormap_file, "w") as fp:
            fp.write("#" + " ".join(self.label_list) + " fval\n")
            for idx, coord, fx in results:
                fp.write(" ".join(
                    map(lambda v: "{:8f}".format(v), (*coord, fx))
                ) + "\n")

            if not np.isinf(opt_fx):
                fp.write("#Index of the minimum point : {:d}\n".format(opt_mesh[0]))
                fp.write("#Coordinates of the minimum point : " + " ".join(
                    map(lambda v: "{:8f}".format(v), opt_mesh[1])
                ) + "\n")
                fp.write("#f(x) at the minimum point : {:8f}\n".format(opt_fx))
            else:
                fp.write("# No mesh point\n")

        time_end = time.perf_counter()
        self.timer["run"]["file_CM"] = time_end - time_sta

    def _post(self) -> dict:
        """
        Post-process the results and gather data from all MPI ranks.

        Returns
        -------
        dict
            Dictionary of results.
        """
        if odatse.mpi.algsize() > 1:
            # gather results
            results_lists = odatse.mpi.algcomm().allgather(self.results)
            results = [v for vs in results_lists for v in vs]

            # gather local optimal values and find minimum among them
            opt_fx_all = odatse.mpi.algcomm().allgather(self.opt_fx)
            opt_mesh_all = odatse.mpi.algcomm().allgather(self.opt_mesh)

            idx = np.argmin(np.array(opt_fx_all))
            opt_fx = opt_fx_all[idx]
            opt_mesh = opt_mesh_all[idx]
        else:
            results = self.results
            opt_fx = self.opt_fx
            opt_mesh = self.opt_mesh

        if odatse.mpi.algrank() == 0:
            self._output_results(results, opt_fx, opt_mesh)

        return {}

    # Mapper-specific fields (simple getattr/setattr).
    _checkpoint_attrs: list[str] = ["results", "opt_fx", "opt_mesh"]

    def __getstate__(self) -> dict:
        """Return a checkpoint snapshot including iterator state.

        Extends the base ``__getstate__()`` with the iterator's own state
        so that a single pickle file captures everything needed to resume.
        """
        state = super().__getstate__()
        state.update(self._iter._save_state())
        return state

    def _apply_state(self, data: dict, mode: str = "resume", restore_rng: bool = True) -> None:
        """Restore algorithm state from a checkpoint snapshot.

        Delegates MPI validation, timer restore, and parameter check to the
        base class, applies the mapper-specific fields, then restores the
        iterator position.

        Parameters
        ----------
        data : dict
            Snapshot previously produced by ``__getstate__``.
        mode : str
            ``"resume"`` is the only supported mode; ``"continue"`` raises
            ``RuntimeError`` because mapper has no concept of extending a run.
        restore_rng : bool
            Forwarded to the base class and to the iterator's state restore
            (e.g. RandomIterator restores its RNG state when this is True).
        """
        if mode == "continue":
            raise RuntimeError("continue mode is not supported for mapper")
        super()._apply_state(data, mode=mode, restore_rng=restore_rng)
        for attr in Algorithm._checkpoint_attrs:
            setattr(self, attr, data[attr])
        self._iter._restore_state(data)
