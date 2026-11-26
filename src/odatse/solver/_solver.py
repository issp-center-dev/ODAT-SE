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
    # Subclasses may declare gradient support as a class attribute, an
    # instance attribute, or a property; a read-only base property would
    # prevent assignment in subclass constructors.
    supports_gradient: bool = False

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

    def gradient(self, x: np.ndarray, args: tuple = ()) -> np.ndarray:
        """
        Return the gradient of the objective w.r.t. x at the given point.

        Optional. Only called when supports_gradient is True. Same shape as x.
        x is in the same space as passed to evaluate (e.g. after Runner's mapping).
        Subclasses must override this method when supports_gradient is True
        (function.Solver subclasses may call set_gradient() instead).
        The Runner uses non-finite return values as returned. A RuntimeError
        (including NotImplementedError) raised by gradient() is treated as an
        all-NaN gradient when ignore_error is True; otherwise it propagates.

        Parameters
        ----------
        x : np.ndarray
            Input data array (same as for evaluate).
        args : tuple, optional
            Additional arguments. Defaults to ().

        Returns
        -------
        np.ndarray
            Gradient vector, same shape as x.

        Raises
        ------
        NotImplementedError
            This solver does not support gradient.
        """
        raise NotImplementedError("This solver does not support gradient.")

    def check_gradient(
        self, x, args: tuple = (0, 0), step=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient and central differences without judging agreement.

        Parameters
        ----------
        x : array_like
            Non-empty, one-dimensional sequence of finite real coordinates.
        args : tuple, optional
            Passed unchanged to every solver call. Defaults to (0, 0).
        step : float or array_like, optional
            Positive finite absolute steps, scalar or shaped like x. Defaults
            to eps**(1/3) * maximum(1, abs(x)) in double precision.

        Returns
        -------
        g : np.ndarray
            Independent float copy of gradient(), retaining its original shape.
        d : np.ndarray
            Central differences, shaped like x. Coincident points give NaN.
            Non-finite differences are returned without warnings.

        Raises
        ------
        ValueError
            Invalid x, step, or non-finite difference coordinates, detected
            before any solver call.

        Exceptions raised by evaluate() and gradient() propagate unchanged.
        A return value of gradient() that cannot be converted to a float
        array fails in that conversion, after the two leading calls.

        Notes
        -----
        Calls evaluate(x), then gradient(x), then evaluate at the positive and
        negative displacement for each component in order. Each call receives
        a fresh float array. There are 2*n + 1 evaluations and one gradient
        call, minus two evaluations per component with coincident points.
        No error measure or pass/fail decision is computed or printed.
        Solver state and working files are not restored. The caller prepares
        any required working directory and chooses suitable points and steps.
        No MPI communication or exception aggregation across ranks is added;
        parallel solvers require all participating ranks to call consistently.
        """
        try:
            with np.errstate(all="ignore"):
                if np.iscomplexobj(x) or np.iscomplexobj(step):
                    raise ValueError("x and step must be real")
                x = np.array(x, dtype=float, copy=True)
                if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
                    raise ValueError("x must be a non-empty finite 1-D array")
                if step is None:
                    scale = np.maximum(1.0, np.abs(x))
                    h = np.finfo(float).eps ** (1.0 / 3.0) * scale
                else:
                    h = np.asarray(step, dtype=float)
                if h.ndim != 0 and h.shape != x.shape:
                    raise ValueError("step must be scalar or have the shape of x")
                if not np.all(np.isfinite(h) & (h > 0)):
                    raise ValueError("step must be positive and finite")
                plus, minus = x + h, x - h
                if not np.all(np.isfinite(plus) & np.isfinite(minus)):
                    raise ValueError("difference coordinates must be finite")
                # Nominal 2*h would ignore rounding in the actual coordinates.
                denominator = plus - minus
        except (TypeError, OverflowError) as exc:
            raise ValueError("x and step must contain finite real numbers") from exc

        self.evaluate(x.copy(), args)
        # Later evaluations may overwrite the solver's gradient buffer.
        g = np.array(self.gradient(x.copy(), args), dtype=float, copy=True)
        d = np.full(x.shape, np.nan)
        for i in range(x.size):
            if plus[i] == minus[i]:
                continue
            xp, xm = x.copy(), x.copy()
            xp[i], xm[i] = plus[i], minus[i]
            fp = self.evaluate(xp, args)
            fm = self.evaluate(xm, args)
            with np.errstate(all="ignore"):
                d[i] = (np.float64(fp) - np.float64(fm)) / denominator[i]
        return g, d
