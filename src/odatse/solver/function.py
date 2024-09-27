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

import os
import numpy as np
import odatse
import time

# type hints
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple


class Solver(odatse.solver.SolverBase):
    x: np.ndarray
    fx: float
    _func: Optional[Callable[[np.ndarray], float]]

    def __init__(self, info: odatse.Info) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
        """
        super().__init__(info)
        self._name = "function"
        self._func = None

        # for debug purpose
        self.delay = info.solver.get("delay", 0.0)

    def evaluate(self, x: np.ndarray, args: Tuple = (), nprocs: int = 1, nthreads: int = 1) -> float:
        self.prepare(x, args)
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.run(nprocs, nthreads)
        os.chdir(cwd)
        result = self.get_results()
        return result

    def prepare(self, x: np.ndarray, args = ()) -> None:
        self.x = x

    def run(self, nprocs: int = 1, nthreads: int = 1) -> None:
        if self._func is None:
            raise RuntimeError(
                "ERROR: function is not set. Make sure that `set_function` is called."
            )
        self.fx = self._func(self.x)
        # for debug purpose
        if self.delay > 0.0:
            time.sleep(self.delay)

    def get_results(self) -> float:
        return self.fx

    def set_function(self, f: Callable[[np.ndarray], float]) -> None:
        self._func = f
