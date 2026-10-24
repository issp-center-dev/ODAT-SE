# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import numpy as np
import odatse
import time

# type hints
from pathlib import Path
from typing import Callable, Optional


class Solver(odatse.solver.SolverBase):
    """
    Solver class for evaluating functions with given parameters.
    """
    x: np.ndarray
    fx: float
    _func: Optional[Callable[[np.ndarray], float]]
    _grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    _supports_gradient: Optional[bool] = None

    def __init__(self, info: odatse.Info) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
            Information object containing solver configuration.
        """
        super().__init__(info)
        self._name = "function"
        self._func = None
        self._grad_func = None

        # for debug purpose
        self.delay = info.solver.get("delay", 0.0)

    def evaluate(self, x: np.ndarray, args: tuple = ()) -> float:
        """
        Evaluate the function with given parameters.

        Parameters
        ----------
        x : np.ndarray
            Input array for the function.
        args : tuple, optional
            Additional arguments for the function.

        Returns
        -------
        float
            Result of the function evaluation.
        """
        self.prepare(x, args)
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        self.run()
        result = self.get_results()
        os.chdir(cwd)
        return result

    def prepare(self, x: np.ndarray, args = ()) -> None:
        """
        Prepare the solver with the given parameters.

        Parameters
        ----------
        x : np.ndarray
            Input array for the function.
        args : tuple, optional
            Additional arguments for the function.
        """
        self.x = x

    def run(self) -> None:
        """
        Run the function evaluation.

        Raises
        ------
        RuntimeError
            If the function is not set.
        """
        if self._func is None:
            raise RuntimeError(
                "ERROR: function is not set. Make sure that `set_function` is called."
            )
        self.fx = self._func(self.x)
        # for debug purpose
        if self.delay > 0.0:
            time.sleep(self.delay)

    def get_results(self) -> float:
        """
        Get the results of the function evaluation.

        Returns
        -------
        float
            Result of the function evaluation.
        """
        return self.fx

    def set_function(self, f: Callable[[np.ndarray], float]) -> None:
        """
        Set the function to be evaluated.

        Parameters
        ----------
        f : Callable[[np.ndarray], float]
            Function to be evaluated.
        """
        self._func = f
        # Keeping the previous gradient could silently differentiate another function.
        self._grad_func = None

    def set_gradient(self, g: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Set the gradient of the function to be evaluated.

        Parameters
        ----------
        g : Callable[[np.ndarray], np.ndarray]
            Function returning the gradient for the input array.
        """
        self._grad_func = g

    def gradient(self, x: np.ndarray, args: tuple = ()) -> np.ndarray:
        """
        Evaluate the gradient in the caller's current directory without delay.

        Parameters
        ----------
        x : np.ndarray
            Input array for the gradient function.
        args : tuple, optional
            Additional arguments, not passed to the gradient function.

        Returns
        -------
        np.ndarray
            Gradient of the function.

        Raises
        ------
        NotImplementedError
            If the gradient is not set.
        """
        if self._grad_func is None:
            raise NotImplementedError(
                "gradient is not set. Make sure that `set_gradient` is called."
            )
        return self._grad_func(x)

    # Not a plain attribute assigned by set_function()/set_gradient(): that
    # assignment would fail for subclasses declaring a read-only property.
    # The setter keeps `self.supports_gradient = ...` in subclasses working.
    @property
    def supports_gradient(self) -> bool:
        """Whether a gradient is provided, unless explicitly overridden."""
        if self._supports_gradient is not None:
            return self._supports_gradient
        return self._grad_func is not None

    @supports_gradient.setter
    def supports_gradient(self, value: bool) -> None:
        self._supports_gradient = value
