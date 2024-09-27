# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import List, Union, Dict

from pathlib import Path
from io import open
import numpy as np
import os
import time

import odatse
import odatse.domain

class Algorithm(odatse.algorithm.AlgorithmBase):
    mesh_list: List[Union[int, float]]

    def __init__(self, info: odatse.Info,
                 runner: odatse.Runner = None,
                 domain = None,
                 run_mode: str = "initial"
    ) -> None:
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        if domain and isinstance(domain, odatse.domain.MeshGrid):
            self.domain = domain
        else:
            self.domain = odatse.domain.MeshGrid(info)

        self.domain.do_split()
        self.mesh_list = self.domain.grid_local

        self.colormap_file = info.algorithm.get("colormap", "ColorMap.txt")
        self.local_colormap_file = Path(self.colormap_file).name + ".tmp"

    def _initialize(self) -> None:
        self.fx_list = []
        self.timer["run"]["submit"] = 0.0
        self._show_parameters()

    def _run(self) -> None:
        # Make ColorMap

        if self.mode is None:
            raise RuntimeError("mode unset")

        if self.mode.startswith("init"):
            # print(">>> initialize")
            self._initialize()
        elif self.mode.startswith("resume"):
            # print(">>> resume")
            self._load_state(self.checkpoint_file)
        else:
            raise RuntimeError("unknown mode {}".format(self.mode))

        # local colormap file
        fp = open(self.local_colormap_file, "a")
        if self.mode.startswith("init"):
            fp.write("#" + " ".join(self.label_list) + " fval\n")

        iterations = len(self.mesh_list)
        istart = len(self.fx_list)

        # print(">>> iterations={}, istart={}".format(iterations, istart))

        next_checkpoint_step = istart + self.checkpoint_steps
        next_checkpoint_time = time.time() + self.checkpoint_interval

        for icount in range(istart, iterations):
            print("Iteration : {}/{}".format(icount+1, iterations))
            mesh = self.mesh_list[icount]

            # update information
            args = (int(mesh[0]), 0)
            x = np.array(mesh[1:])

            time_sta = time.perf_counter()
            fx = self.runner.submit(x, args)
            time_end = time.perf_counter()
            self.timer["run"]["submit"] += time_end - time_sta

            self.fx_list.append([mesh[0], fx])

            # write to local colormap file
            fp.write(" ".join(
                map(lambda v: "{:8f}".format(v), (*x, fx))
            ) + "\n")

            if self.checkpoint:
                time_now = time.time()
                if icount+1 >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = icount + 1 + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        if iterations > 0:
            opt_index = np.argsort(self.fx_list, axis=0)[0][1]
            opt_id, opt_fx = self.fx_list[opt_index]
            opt_mesh = self.mesh_list[opt_index]

            # assert opt_id == opt_mesh[0]

            self.opt_fx = opt_fx
            self.opt_mesh = opt_mesh

            print(f"[{self.mpirank}] minimum_value: {opt_fx:12.8e} at {opt_mesh[1:]} (mesh {opt_mesh[0]})")

        self._output_results()

        if Path(self.local_colormap_file).exists():
            # print(">>> remove local colormap file {}".format(self.local_colormap_file))
            os.remove(Path(self.local_colormap_file))

        print("complete main process : rank {:08d}/{:08d}".format(self.mpirank, self.mpisize))

    def _output_results(self):
        print("Make ColorMap")
        time_sta = time.perf_counter()

        with open(self.colormap_file, "w") as fp:
            fp.write("#" + " ".join(self.label_list) + " fval\n")

            for x, (idx, fx) in zip(self.mesh_list, self.fx_list):
                fp.write(" ".join(
                    map(lambda v: "{:8f}".format(v), (*x[1:], fx))
                    ) + "\n")

            if len(self.mesh_list) > 0:
                fp.write("#Minimum point : " + " ".join(
                    map(lambda v: "{:8f}".format(v), self.opt_mesh[1:])
                ) + "\n")
                fp.write("#R-factor : {:8f}\n".format(self.opt_fx))
                fp.write("#see Log{:d}\n".format(round(self.opt_mesh[0])))
            else:
                fp.write("# No mesh point\n")

        time_end = time.perf_counter()
        self.timer["run"]["file_CM"] = time_end - time_sta

    def _prepare(self) -> None:
        # do nothing
        pass

    def _post(self) -> Dict:
        if self.mpisize > 1:
            fx_lists = self.mpicomm.allgather(self.fx_list)
            results = [v for vs in fx_lists for v in vs]
        else:
            results = self.fx_list

        if self.mpirank == 0:
            with open(self.colormap_file, "w") as fp:
                # fp.write("#" + " ".join(self.label_list) + " fval\n")
                for x, (idx, fx) in zip(self.domain.grid, results):
                    assert x[0] == idx
                    fp.write(" ".join(
                        map(lambda v: "{:8f}".format(v), (*x[1:], fx))
                    ) + "\n")

        return {}

    def _save_state(self, filename) -> None:
        data = {
            #-- _algorithm
            "mpisize": self.mpisize,
            "mpirank": self.mpirank,
            #"rng": self.rng.get_state(),
            "timer": self.timer,
            "info": self.info,
            #-- mapper
            "fx_list": self.fx_list,
            "mesh_size": len(self.mesh_list),
        }
        self._save_data(data, filename)

    def _load_state(self, filename, restore_rng=True):
        data = self._load_data(filename)
        if not data:
            print("ERROR: Load status file failed")
            sys.exit(1)

        #-- _algorithm
        assert self.mpisize == data["mpisize"]
        assert self.mpirank == data["mpirank"]

        # if restore_rng:
        #     self.rng = np.random.RandomState()
        #     self.rng.set_state(data["rng"])
        self.timer = data["timer"]

        info = data["info"]
        self._check_parameters(info)

        #-- mapper
        self.fx_list = data["fx_list"]

        assert len(self.mesh_list) == data["mesh_size"]
