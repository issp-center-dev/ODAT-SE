# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
        Initialize the Run class.

        Parameters
        ----------
        nprocs : int
            Number of processes which one solver uses.
        nthreads : int
            Number of threads which one solver process uses.
        comm : MPI.Comm
            MPI Communicator.
        """
        self.nprocs = nprocs
        self.nthreads = nthreads
        self.comm = comm

    @abstractmethod
    def submit(self, solver):
        """
        Abstract method to submit a solver.

        Parameters
        ----------
        solver : object
            Solver object to be submitted.
        """
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
        Initialize the Runner class.

        Parameters
        ----------
        solver : odatse.solver.SolverBase
            Solver object.
        info : Optional[odatse.Info]
            Information object.
        mapping : object, optional
            Mapping object.
        limitation : object, optional
            Limitation object.
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
        """
        Prepare the logger with the given process directory.

        Parameters
        ----------
        proc_dir : Path
            Path to the process directory.
        """
        self.logger.prepare(proc_dir)

    def submit(
            self, x: np.ndarray, args = (), nprocs: int = 1, nthreads: int = 1
    ) -> float:
        """
        Submit the solver with the given parameters.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        args : tuple, optional
            Additional arguments.
        nprocs : int, optional
            Number of processes.
        nthreads : int, optional
            Number of threads.

        Returns
        -------
        float
            Result of the solver evaluation.
        """
        if self.limitation.judge(x):
            xp = self.mapping(x)
            result = self.solver.evaluate(xp, args)
        else:
            result = np.inf
        self.logger.count(x, args, result)
        return result

    def post(self) -> None:
        """
        Write the logger data.
        """
        self.logger.write()
