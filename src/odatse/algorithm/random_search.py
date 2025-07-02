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

class Algorithm(MapperMPIAlgorithm):
    """
    Algorithm class for data analysis of quantum beam diffraction experiments.
    Inherits from odatse.algorithm.mapper_mpi.Algorithm.
    """
    mesh_list: List[Union[int, float]]

    def __init__(self, info: odatse.Info,
                 runner: Optional[odatse.Runner] = None,
                 domain = None,
                 run_mode: str = "initial"
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

        super().__init__(info=info, runner=runner, domain=domain, run_mode=run_mode, meshgrid=False)
