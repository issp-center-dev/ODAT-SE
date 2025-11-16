# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Union, Dict, Optional

from pathlib import Path
from io import open
import numpy as np
import os
import time

import odatse
import odatse.domain
from ._algorithm import AlgorithmBase



class Algorithm(AlgorithmBase):
    """
    Algorithm class for data analysis of quantum beam diffraction experiments.
    Inherits from odatse.algorithm.AlgorithmBase.
    """
    #mesh_list: List[Union[int, float]]

    def __init__(self,
                 info: odatse.Info,
                 runner: Optional[odatse.Runner] = None,
                 run_mode: str = "initial",
                 mpicomm: Optional["MPI.Comm"] = None,
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
        domain :
            Optional domain object, defaults to MeshGrid.
        run_mode : str
            Mode to run the algorithm, defaults to "initial".
        meshgrid : bool
            Whether to use mesh grid or points.
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode, mpicomm=mpicomm)

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

    def _run(self) -> None:
        """
        Execute the main algorithm process.
        """
        # Make ColorMap

        if self.mode is None:
            raise RuntimeError("mode unset")

        if self.mode.startswith("init"):
            self._initialize()
        elif self.mode.startswith("resume"):
            self._load_state(self.checkpoint_file)
        else:
            raise RuntimeError("unknown mode {}".format(self.mode))

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

        if self.opt_fx is not None:
            print(f"[{self.mpirank}] minimum_value: {self.opt_fx:12.8e} at {self.opt_mesh[0]} (mesh {self.opt_mesh[1]})")

        self._output_results()

        # if Path(self.local_colormap_file).exists():
        #     os.remove(Path(self.local_colormap_file))

        print("complete main process : rank {:08d}/{:08d}".format(self.mpirank, self.mpisize))

    def _output_results(self):
        """
        Output the results to the colormap file.
        """
        print("Make ColorMap")
        time_sta = time.perf_counter()

        with open(self.colormap_file, "w") as fp:
            fp.write("#" + " ".join(self.label_list) + " fval\n")
            for idx, coord, fx in self.results:
                fp.write(" ".join(
                    map(lambda v: "{:8f}".format(v), (*coord, fx))
                ) + "\n")

            if self.opt_fx is not None:
                fp.write("#Minimum point : " + " ".join(
                    map(lambda v: "{:8f}".format(v), self.opt_mesh[1])
                ) + "\n")
                fp.write("#R-factor : {:8f}\n".format(self.opt_fx))
                fp.write("#see Log{:d}\n".format(round(self.opt_mesh[0])))
            else:
                fp.write("# No mesh point\n")

        time_end = time.perf_counter()
        self.timer["run"]["file_CM"] = time_end - time_sta

    def _prepare(self) -> None:
        """
        Prepare the algorithm (no operation).
        """
        pass

    def _post(self) -> Dict:
        """
        Post-process the results and gather data from all MPI ranks.

        Returns
        -------
        Dict
            Dictionary of results.
        """
        if self.mpisize > 1:
            _data = self.mpicomm.allgather(self.results)
            results = [v for vs in _data for v in vs]
        else:
            results = self.results

        if self.mpirank == 0:
            with open(self.colormap_file, "w") as fp:
                for idx, coord, fx in results:
                    fp.write(" ".join(
                        map(lambda v: "{:8f}".format(v), (*coord, fx))
                    ) + "\n")

        return {}

    def _save_state(self, filename) -> None:
        """
        Save the current state of the algorithm to a file.

        Parameters
        ----------
        filename
            The name of the file to save the state to.
        """
        data = {
            "mpisize": self.mpisize,
            "mpirank": self.mpirank,
            "timer": self.timer,
            "info": self.info,
            #"fx_list": self.fx_list,
            #"mesh_size": len(self.mesh_list),
        }
        self._save_data(data, filename)

    def _load_state(self, filename, restore_rng=True):
        """
        Load the state of the algorithm from a file.

        Parameters
        ----------
        filename
            The name of the file to load the state from.
        restore_rng : bool
            Whether to restore the random number generator state.
        """
        data = self._load_data(filename)
        if not data:
            print("ERROR: Load status file failed")
            sys.exit(1)

        assert self.mpisize == data["mpisize"]
        assert self.mpirank == data["mpirank"]

        self.timer = data["timer"]

        info = data["info"]
        self._check_parameters(info)

        #self.fx_list = data["fx_list"]

        #assert len(self.mesh_list) == data["mesh_size"]
