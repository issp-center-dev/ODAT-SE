# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np

import odatse
import odatse.solver.function

# type hints
from typing import Callable, Optional

def quadratics(xs: np.ndarray) -> float:
    """
    Quadratic (sphere) function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the quadratic function.

    Notes
    -----
    It has one global minimum f(xs)=0 at xs = [0,0,...,0].
    """
    return np.sum(xs * xs)

def quadratics_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of quadratics (sphere): 2*x."""
    return 2.0 * xs.copy()

def quartics(xs: np.ndarray) -> float:
    """
    Quartic function with two global minima.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the quartic function.

    Notes
    -----
    It has two global minima f(xs)=0 at xs = [1,1,...,1] and [-1,-1,...,-1].
    It has one saddle point f(0,0,...,0) = 1.0.
    """
    return np.mean((xs - 1.0) ** 2) * np.mean((xs + 1.0) ** 2)

def quartics_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of quartics."""
    n = xs.size
    m1 = np.mean((xs - 1.0) ** 2)
    m2 = np.mean((xs + 1.0) ** 2)
    return (2.0 / n) * (m2 * (xs - 1.0) + m1 * (xs + 1.0))

def ackley(xs: np.ndarray) -> float:
    """
    Ackley's function in arbitrary dimension

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of Ackley's function.

    Notes
    -----
    It has one global minimum f(xs)=0 at xs=[0,0,...,0].
    It has many local minima.
    """
    a = np.mean(xs ** 2, axis = 0)
    a = 20 * np.exp(-0.2 * np.sqrt(a))
    b = np.cos(2.0 * np.pi * xs)
    b = np.exp(np.mean(b, axis = 0))
    return 20.0 + np.exp(1.0) - a - b

def ackley_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of Ackley's function."""
    n = xs.size
    sqrt_mean = np.sqrt(np.mean(xs ** 2))
    a = 20 * np.exp(-0.2 * sqrt_mean)
    b = np.exp(np.mean(np.cos(2.0 * np.pi * xs)))
    if sqrt_mean == 0:
        da_dx = np.zeros_like(xs, dtype=float)
    else:
        da_dx = a * (-0.2 / sqrt_mean) * (xs / n)
    db_dx = b * (-2.0 * np.pi / n) * np.sin(2.0 * np.pi * xs)
    return -da_dx - db_dx

def alpine(xs: np.ndarray) -> float:
    """
    Alpine function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Alpine function.

    Notes
    -----
    It has a global minimum f(x)=0 at x=[0,0,...,0].
    """
    return np.sum(np.abs((xs*np.sin(xs))+(0.1*xs)),axis=0)

def exponential(xs: np.ndarray) -> float:
    """
    Exponential function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Exponential function.

    Notes
    -----
    It has a global minimum f(x)=-1 at x=[0,0,...,0].
    """
    return -np.exp(-0.5*np.sum(xs**2,axis=0))

def griewank(xs: np.ndarray) -> float:
    """
    Griewank function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Griewank function.

    Notes
    -----
    It has a global minimum f(x)=0 at x=[0,0,...,0].
    """
    return 1+(np.sum(xs**2,axis=0)/4000)-np.prod(np.cos(xs/np.sqrt((np.arange(xs.shape[0])+1))),axis=0)

def himmelblau(xs: np.ndarray) -> float:
    """
    Himmelblau's function.

    Parameters
    ----------
    xs : np.ndarray
        Input array of shape (2,).

    Returns
    -------
    float
        The calculated value of Himmelblau's function.

    Notes
    -----
    It has four global minima f(xs) = 0 at
    xs=[3,2], [-2.805118..., 3.131312...], [-3.779310..., -3.2831860], and [3.584428..., -1.848126...].
    """
    if xs.shape[0] != 2:
        raise RuntimeError(
            f"ERROR: himmelblau expects d=2 input, but receives d={xs.shape[0]} one"
        )
    return (xs[0] ** 2 + xs[1] - 11.0) ** 2 + (xs[0] + xs[1] ** 2 - 7.0) ** 2

def himmelblau_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of Himmelblau's function (d=2)."""
    a = xs[0] ** 2 + xs[1] - 11.0
    b = xs[0] + xs[1] ** 2 - 7.0
    return np.array([
        2.0 * a * 2.0 * xs[0] + 2.0 * b * 1.0,
        2.0 * a * 1.0 + 2.0 * b * 2.0 * xs[1],
    ])

def michalewicz(xs: np.ndarray) -> float:
    """
    Michalewicz function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Michalewicz function.

    Notes
    -----
    The global minimum value and location depend on the dimension. There are d! local minima.
    For d=2, it has a global minimum f(x)=-1.8013 at x=[2.2051, 1.5698].
    For d=5, it has a global minimum f(x)=-4.6876.
    For d=10, it has a global minimum f(x)=-9.6602.
    """
    return -np.sum(np.sin(xs)*(np.sin(((np.arange(xs.shape[0])+1))*(xs**2)/np.pi)**(2*10)),axis=0)

def qing(xs: np.ndarray) -> float:
    """
    Qing function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Qing function.

    Notes
    -----
    It has a global minimum f(x)=0 at x=[+/-sqrt(n), ...], where n runs from 1 to d.
    """
    return np.sum(((xs**2)-((np.arange(xs.shape[0])+1)))**2,axis=0)

def rastrigin(xs: np.ndarray) -> float:
    """
    Rastrigin function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Rastrigin function.

    Notes
    -----
    It has a global minimum f(x)=0 at x=[0,0,...,0].
    """
    return (xs.shape[0]*10)+np.sum(xs**2-(10*np.cos(2*np.pi*xs)),axis=0)

def rosenbrock(xs: np.ndarray) -> float:
    """
    Rosenbrock's function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of Rosenbrock's function.

    Notes
    -----
    It has one global minimum f(xs) = 0 at xs=[1,1,...,1].
    """
    return np.sum(100.0 * (xs[1:] - xs[:-1] ** 2) ** 2 + (1.0 - xs[:-1]) ** 2, axis = 0)

def rosenbrock_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock's function.

    For dimension 1, return a zero vector because the function is identically zero.
    """
    g = np.zeros_like(xs, dtype=float)
    if xs.size == 1:
        return g
    g[0] = (
        -400.0 * xs[0] * (xs[1] - xs[0] ** 2)
        - 2.0 * (1.0 - xs[0])
    )
    g[-1] = 200.0 * (xs[-1] - xs[-2] ** 2)
    g[1:-1] = (
        200.0 * (xs[1:-1] - xs[:-2] ** 2)
        - 400.0 * xs[1:-1] * (xs[2:] - xs[1:-1] ** 2)
        - 2.0 * (1.0 - xs[1:-1])
    )
    return g

def schaffer(xs: np.ndarray) -> float:
    """
    Schaffer function (generalized).

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Schaffer function.
    """
    a=(xs[:-1]**2)+(xs[1:]**2)
    return np.sum(0.5+(((np.sin(a)**2)-0.5)/((1+(0.001*a))**2)),axis=0)

def schwefel(xs: np.ndarray) -> float:
    """
    Schwefel function.

    Parameters
    ----------
    xs : np.ndarray
        Input array.

    Returns
    -------
    float
        The calculated value of the Schwefel function.

    Notes
    -----
    It has a global minimum f(x)=0 at x=[420.9687..., ...].
    """
    return (xs.shape[0]*418.9829)-np.sum(xs*np.sin(np.sqrt(np.abs(xs))),axis=0)

def linear_regression_test(xs: np.ndarray) -> float:
    """
    Negative log likelihood of linear regression with Gaussian noise N(0,sigma)

    y = ax + b

    trained by xdata = [1, 2, 3, 4, 5, 6] and ydata = [1, 3, 2, 4, 3, 5].

    Model parameters (a, b, sigma) are corresponding to xs as the following,
    a = xs[0], b = xs[1], log(sigma**2) = xs[2]

    It has a global minimum f(xs) = 1.005071.. at
    xs = [0.628571..., 0.8, -0.664976...].

    Parameters
    ----------
    xs : np.ndarray
        Input array of model parameters.

    Returns
    -------
    float
        The negative log likelihood of the linear regression model.
    """
    if xs.shape[0] != 3:
        raise RuntimeError(
            f"ERROR: regression expects d=3 input, but receives d={xs.shape[0]} one"
        )

    xdata = np.array([1, 2, 3, 4, 5, 6])
    ydata = np.array([1, 3, 2, 4, 3, 5])
    n = len(ydata)

    return 0.5 * (
        n * xs[2] + np.sum((xs[0] * xdata + xs[1] - ydata) ** 2) / np.exp(xs[2])
    )

def linear_regression_test_gradient(xs: np.ndarray) -> np.ndarray:
    """Gradient of linear_regression_test (negative log likelihood, d=3)."""
    xdata = np.array([1, 2, 3, 4, 5, 6])
    ydata = np.array([1, 3, 2, 4, 3, 5])
    n = len(ydata)
    resid = xs[0] * xdata + xs[1] - ydata
    s = np.sum(resid ** 2)
    exp_s2 = np.exp(xs[2])
    return np.array([
        np.sum(resid * xdata) / exp_s2,
        np.sum(resid) / exp_s2,
        0.5 * (n - s / exp_s2),
    ])

class Solver(odatse.solver.function.Solver):
    """Function Solver with pre-defined benchmark functions and gradients."""

    x: np.ndarray
    fx: float
    # None for the functions whose gradient is not implemented
    _grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def set_function(self, f: Callable[[np.ndarray], float]) -> None:
        # the gradient of the previous function would silently be wrong for f
        super().set_function(f)
        self._grad_func = None
        self.supports_gradient = False

    def gradient(self, x: np.ndarray, args: tuple = ()) -> np.ndarray:
        if self._grad_func is None:
            raise NotImplementedError(
                f"gradient of {self._function_name} is not implemented"
            )
        return self._grad_func(x)

    def __init__(self, info: odatse.Info) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
            Information object containing solver configuration.
        """
        super().__init__(info)
        self._name = "analytical"
        function_name = info.solver.get("function_name", "quadratics")
        self._function_name = function_name

        if function_name == "quadratics":
            self.set_function(quadratics)
            self._grad_func = quadratics_gradient
        elif function_name == "quartics":
            self.set_function(quartics)
            self._grad_func = quartics_gradient
        elif function_name == "ackley":
            self.set_function(ackley)
            self._grad_func = ackley_gradient
        elif function_name == "alpine":
            self.set_function(alpine)
        elif function_name == "exponential":
            self.set_function(exponential)
        elif function_name == "griewank":
            self.set_function(griewank)
        elif function_name == "himmelblau":
            dimension = self.dimension
            if int(dimension) != 2:
                raise RuntimeError(
                    f"ERROR: himmelblau works only with dimension=2 but input is dimension={dimension}"
                )
            self.set_function(himmelblau)
            self._grad_func = himmelblau_gradient
        elif function_name == "michalewicz":
            self.set_function(michalewicz)
        elif function_name == "qing":
            self.set_function(qing)
        elif function_name == "rastrigin":
            self.set_function(rastrigin)
        elif function_name == "rosenbrock":
            self.set_function(rosenbrock)
            self._grad_func = rosenbrock_gradient
        elif function_name == "schaffer":
            self.set_function(schaffer)
        elif function_name == "schwefel":
            self.set_function(schwefel)
        elif function_name == "linear_regression_test":
            dimension = self.dimension
            if int(dimension) != 3:
                raise RuntimeError(
                    f"ERROR: regression works only with dimension=3 but input is dimension={dimension}"
                )
            self.set_function(linear_regression_test)
            self._grad_func = linear_regression_test_gradient
        else:
            raise RuntimeError(f"ERROR: Unknown function, {function_name}")
        # a plain attribute, not a property, so that subclasses can assign it
        self.supports_gradient = self._grad_func is not None
