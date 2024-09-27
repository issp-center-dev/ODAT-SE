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

import subprocess
import numpy as np

import odatse
import odatse.mpi

# type hints
from pathlib import Path
from typing import Dict, List, Tuple


class SolverBase(object, metaclass=ABCMeta):
    root_dir: Path
    output_dir: Path
    proc_dir: Path
    work_dir: Path
    _name: str
    dimension: int
    timer: Dict[str, Dict]

    @abstractmethod
    def __init__(self, info: odatse.Info) -> None:
        self.root_dir = info.base["root_dir"]
        self.output_dir = info.base["output_dir"]
        self.proc_dir = self.output_dir / str(odatse.mpi.rank())
        self.work_dir = self.proc_dir
        self._name = ""
        self.timer = {"prepare": {}, "run": {}, "post": {}}
        if "dimension" in info.solver:
            self.dimension = info.solver["dimension"]
        else:
            self.dimension = info.base["dimension"]

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def evaluate(self, x: np.ndarray, arg: Tuple = (), nprocs: int = 1, nthreads: int = 1) -> None:
        raise NotImplementedError()
