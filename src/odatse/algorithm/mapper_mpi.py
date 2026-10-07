# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path
import numpy as np

import odatse
from .mapper_mpi_base import Algorithm as MapperMPIAlgorithm
from ._iterator import MeshIterator, ListIterator
from odatse.domain.meshgrid import load_mesh_file


class Algorithm(MapperMPIAlgorithm):
    """
    Algorithm class for mapping the objective function over a set of points.
    Inherits from odatse.algorithm.mapper_mpi_base.Algorithm.
    """

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        run_mode: str = "initial",
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
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        if odatse.mpi.run_on_algorithm():
            info_param = info.algorithm.get("param", {})
            if "mesh_path" in info_param:
                self._iter = self._read_mesh_file(info_param)
            else:
                self._iter = self._find_mesh_info(info_param)
        else:
            self._iter = None

    def _read_mesh_file(self, info_param):
        """
        Setup the grid from a file.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        # The shared reader validates the file on every algorithm rank;
        # ListIterator then scatters the rows from algorithm rank 0.
        # mesh data format: index x1 x2 ...
        _data = load_mesh_file(self.root_dir, info_param, root_only=True)
        data = [[int(idx), *v] for idx, *v in _data] if _data is not None else None
        return ListIterator(data)

    def _find_mesh_info(self, info_param):
        """
        Setup the grid based on min, max, and num lists.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = info_param["min_list"]

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = info_param["max_list"]

        if "num_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_list is not defined in the input")
        num_list = info_param["num_list"]

        if len(min_list) != len(max_list) or len(min_list) != len(num_list):
            raise ValueError("ERROR: lengths of min_list, max_list, num_list do not match")

        return MeshIterator(min_list, max_list, num_list)
