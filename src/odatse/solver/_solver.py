# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import abstractmethod

import numpy as np

import odatse
import odatse.mpi

# type hints
from pathlib import Path


class SolverBase(object, metaclass=odatse.mpi.FailTogetherMeta):
    """
    Abstract base class for solvers in ODAT-SE.

    Construction takes part in the agreement of ``odatse.mpi.fail_together()``
    (``FailTogetherMeta``): a constructor that fails on some processes only
    makes every process leave ``Solver(info)``, the failing ones with their
    own exception and the others with ``OtherAlgorithmProcessError``, instead
    of leaving them waiting in a later collective (issue #112). Every process
    of the job must therefore construct the solver, and collectives inside a
    subclass constructor must stay balanced across the processes.
    """

    root_dir: Path
    output_dir: Path
    proc_dir: Path
    work_dir: Path
    _name: str
    dimension: int
    timer: dict[str, dict]

    @abstractmethod
    def __init__(self, info: odatse.Info) -> None:
        """
        Initialize the solver with the given information.

        Parameters
        ----------
        info : Info
            Information object containing configuration details.
        """
        self.root_dir = info.base["root_dir"]
        self.output_dir = info.base["output_dir"]
        self.proc_dir = self.output_dir / str(odatse.mpi.algrank())
        self.work_dir = self.proc_dir
        self._name = ""
        self.timer = {"prepare": {}, "run": {}, "post": {}}
        if "dimension" in info.solver:
            self.dimension = info.solver["dimension"]
        else:
            self.dimension = info.base["dimension"]

    @property
    def name(self) -> str:
        """
        Get the name of the solver.

        Returns
        -------
        str
            The name of the solver.
        """
        return self._name

    @abstractmethod
    def evaluate(self, x: np.ndarray, args: tuple = ()) -> float:
        """
        Evaluate the solver with the given parameters.

        Parameters
        ----------
        x : np.ndarray
            Input data array.
        args : tuple, optional
            Additional arguments for evaluation. Defaults to ().

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()
