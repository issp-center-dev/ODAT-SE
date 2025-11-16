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

from .mapper_mpi_base import Algorithm as MapperMPIAlgorithm
from ._iterator import RandomIterator, ListIterator


class Algorithm(MapperMPIAlgorithm):
    """
    Algorithm class for data analysis of quantum beam diffraction experiments.
    Inherits from odatse.algorithm.mapper_mpi.Algorithm.
    """
    mesh_list: List[Union[int, float]]

    def __init__(self,
                 info: odatse.Info,
                 runner: Optional[odatse.Runner] = None,
                 domain = None,
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
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode, mpicomm=mpicomm)

        info_mode = info.algorithm.get("mode", None)
        if info_mode is None:
            mode = "random"
        else:
            mode = info_mode.get("mode", "random")

        info_param = info.algorithm.get("param", {})

        if mode == "random":
            iter = self._random_iterator(info_param, self.rng, mpicomm)
        elif mode == "quasi-random":
            seq = info_mode.get("sequence", "sobol")
            iter = self._quasi_random_iterator(info_param, seq, mpicomm)
        else:
            raise ValueError("ERROR: algorithm.mode.mode = {} is not supported".format(mode))

        # delayed setup
        self._iter = iter

    def _random_iterator(self, info_param, rng, mpicomm=None):
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

        if "num_points" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_points is not defined in the input")
        num_points = info_param["num_points"]

        if len(min_list) != len(max_list):
            raise ValueError("ERROR: lengths of min_list and max_list do not match")
        if num_points <= 0:
            raise ValueError("ERROR: num_points must be positive")

        return RandomIterator(min_list, max_list, num_points, rng, mpicomm)

    def _quasi_random_iterator(self, info_param, seq, mpicomm=None):
        from scipy.stats import qmc

        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = info_param["min_list"]

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = info_param["max_list"]

        if "num_points" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_points is not defined in the input")
        num_points = info_param["num_points"]

        if len(min_list) != len(max_list):
            raise ValueError("ERROR: lengths of min_list and max_list do not match")
        if num_points <= 0:
            raise ValueError("ERROR: num_points must be positive")

        if self.mpirank == 0:
            d = len(min_list)

            if seq == "sobol":
                sampler = qmc.Sobol(d, scramble=True, optimization=None)
            elif seq == "halton":
                sampler = qmc.Halton(d, scramble=True, optimization=None)
            elif seq == "latin":
                sampler = qmc.LatinHypercube(d, scramble=True, strength=1, optimization=None)
            else:
                ValueError("unknown sequence type {}".format(seq))

            # generate samples on rank 0 all at once
            idx = np.arange(num_points)
            sample = sampler.random(n=num_points)
            print("discrepancy=", qmc.discrepancy(sample))
            sample = qmc.scale(sample, min_list, max_list)

            data = [[i, *x] for i, x in zip(idx, sample)]
        else:
            data = None

        return ListIterator(data, mpicomm)
