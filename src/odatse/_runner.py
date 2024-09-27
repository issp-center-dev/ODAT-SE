# -*- coding: utf-8 -*-

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

from abc import ABCMeta, abstractmethod

import numpy as np

import odatse
import odatse.util.read_matrix
import odatse.util.mapping
import odatse.util.limitation
from odatse.util.logger import Logger
from odatse.exception import InputError

# type hints
from pathlib import Path
from typing import List, Optional
from . import mpi


class Run(metaclass=ABCMeta):
    def __init__(self, nprocs=None, nthreads=None, comm=None):
        """
        Parameters
        ----------
        nprocs : int
            Number of process which one solver uses
        nthreads : int
            Number of threads which one solver process uses
        comm : MPI.Comm
            MPI Communicator
        """
        self.nprocs = nprocs
        self.nthreads = nthreads
        self.comm = comm

    @abstractmethod
    def submit(self, solver):
        pass


class Runner(object):
    #solver: "odatse.solver.SolverBase"
    logger: Logger

    def __init__(self,
                 solver,
                 info: Optional[odatse.Info] = None,
                 mapping = None,
                 limitation = None) -> None:
        """

        Parameters
        ----------
        Solver: odatse.solver.SolverBase object
        """
        self.solver = solver
        self.solver_name = solver.name
        self.logger = Logger(info)

        if mapping is not None:
            self.mapping = mapping
        elif "mapping" in info.runner:
            info_mapping = info.runner["mapping"]
            # N.B.: only Affine mapping is supported at present
            self.mapping = odatse.util.mapping.Affine.from_dict(info_mapping)
        else:
            # trivial mapping
            self.mapping = odatse.util.mapping.TrivialMapping()
        
        if limitation is not None:
            self.limitation = limitation
        elif "limitation" in info.runner:
            info_limitation = info.runner["limitation"]
            self.limitation = odatse.util.limitation.Inequality.from_dict(info_limitation)
        else:
            self.limitation = odatse.util.limitation.Unlimited()

    def prepare(self, proc_dir: Path):
        self.logger.prepare(proc_dir)

    def submit(
            self, x: np.ndarray, args = (), nprocs: int = 1, nthreads: int = 1
    ) -> float:
        if self.limitation.judge(x):
            xp = self.mapping(x)
            result = self.solver.evaluate(xp, args)
        else:
            result = np.inf
        self.logger.count(x, args, result)
        return result

    def post(self) -> None:
        self.logger.write()
