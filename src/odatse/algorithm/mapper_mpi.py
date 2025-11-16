# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Union, Dict, Optional, TYPE_CHECKING

from pathlib import Path
from io import open
import numpy as np
import os
import time

import odatse
import odatse.domain
from .mapper_mpi_base import Algorithm as MapperMPIAlgorithm
from ._iterator import MeshIterator, ListIterator, DistributedListIterator


class Algorithm(MapperMPIAlgorithm):
    """
    Algorithm class for data analysis of quantum beam diffraction experiments.
    Inherits from odatse.algorithm.AlgorithmBase.
    """
    mesh_list: List[Union[int, float]]

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        domain=None,
        run_mode: str = "initial",
        mpicomm: Optional["MPI.Comm"] = None,
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
        mpicomm : MPI.Comm
            MPI communicator to use for parallelization.
            If not provided, the default MPI communicator (MPI.COMM_WORLD) will be used if mpi4py is installed.
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode, mpicomm=mpicomm)

        if domain:
            iter = DistributedListIterator(domain.grid_local, mpicomm)
        else:
            info_param = info.algorithm.get("param", {})
            if "mesh_path" in info_param:
                iter = self._read_mesh_file(info_param, mpicomm)
            else:
                iter = self._find_mesh_info(info_param, mpicomm)

        # delayed setup
        self._iter = iter

    def _read_mesh_file(self, info_param, mpicomm=None):
        """
        Setup the grid from a file.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "mesh_path" not in info_param:
            raise ValueError("ERROR: mesh_path not defined")
        mesh_path = self.root_dir / Path(info_param["mesh_path"]).expanduser()

        if not mesh_path.exists():
            raise FileNotFoundError("mesh_path not found: {}".format(mesh_path))

        comments = info_param.get("comments", "#")
        delimiter = info_param.get("delimiter", None)
        skiprows = info_param.get("skiprows", 0)

        if self.mpirank == 0:
            # mesh data format: index x1 x2 ...
            _data = np.loadtxt(mesh_path, comments=comments, delimiter=delimiter, skiprows=skiprows)
            if _data.ndim == 1:
                _data = _data.reshape(1, -1)
            data = [[int(v[0]), *v[1:]] for v in _data]
        else:
            data = None

        return ListIterator(data, mpicomm)

    def _find_mesh_info(self, info_param, mpicomm=None):
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

        return MeshIterator(min_list, max_list, num_list, mpicomm)
