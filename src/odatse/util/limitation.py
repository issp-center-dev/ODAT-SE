# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABCMeta, abstractmethod

import numpy as np

from .read_matrix import read_matrix, read_vector


class LimitationBase(metaclass=ABCMeta):
    """
    Abstract base class for limitations.
    """

    @abstractmethod
    def __init__(self, is_limitary: bool):
        """
        Initialize the limitation.

        Parameters
        ----------
        is_limitary : bool
            Boolean indicating if the limitation is active.
        """
        self.is_limitary = is_limitary

    @abstractmethod
    def judge(self, x: np.ndarray) -> bool:
        """
        Abstract method to judge if the limitation is satisfied.

        Parameters
        ----------
        x : np.ndarray
            Input array to be judged.

        Returns
        -------
        bool
            Boolean indicating if the limitation is satisfied.
        """
        raise NotImplementedError

class Unlimited(LimitationBase):
    """
    Class representing an unlimited (no limitation) condition.
    """

    def __init__(self):
        """
        Initialize the unlimited condition.
        """
        super().__init__(False)

    def judge(self, x: np.ndarray) -> bool:
        """
        Always returns True as there is no limitation.

        Parameters
        ----------
        x : np.ndarray
            Input array to be judged.

        Returns
        -------
        bool
            Always True.
        """
        return True

class Inequality(LimitationBase):
    """
    Class representing an inequality limitation.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, is_limitary: bool):
        """
        Initialize the inequality limitation.

        Parameters
        ----------
        a : np.ndarray
            Coefficient matrix.
        b : np.ndarray
            Constant vector.
        is_limitary : bool
            Boolean indicating if the limitation is active.
        """
        super().__init__(is_limitary)
        if self.is_limitary:
            self.a = np.array(a)
            self.b = np.array(b)
            self.minusb = -np.array(b)
            self.n_formula = a.shape[0]
            self.ndim = a.shape[1]

    def judge(self, x: np.ndarray) -> bool:
        """
        Judge if the inequality limitation is satisfied.

        Parameters
        ----------
        x : np.ndarray
            Input array to be judged.

        Returns
        -------
        bool
            Boolean indicating if the limitation is satisfied.
        """
        if self.is_limitary:
            Ax_b = np.dot(self.a, x) + self.b
            judge_result = np.all(Ax_b > 0)
        else:
            judge_result = True
        return judge_result

    @classmethod
    def from_dict(cls, d):
        """
        Create an Inequality instance from a dictionary.

        Parameters
        ----------
        d
            Dictionary containing 'co_a' and 'co_b' keys.

        Returns
        -------
        Inequality
            an Inequality instance.
        """
        co_a: np.ndarray = read_matrix(d.get("co_a", []))
        co_b: np.ndarray = read_matrix(d.get("co_b", []))

        if co_a.size == 0:
            is_set_co_a = False
        else:
            if co_a.ndim == 2:
                is_set_co_a = True
            else:
                raise ValueError("co_a should be a matrix of size equal to number of constraints times dimension")

        if co_b.size == 0:
            is_set_co_b = False
        else:
            if co_b.ndim == 2 and co_b.shape == (co_a.shape[0], 1):
                is_set_co_b = True
            else:
                raise ValueError("co_b should be a column vector of size equal to number of constraints")

        if is_set_co_a and is_set_co_b:
            is_limitary = True
        elif (not is_set_co_a) and (not is_set_co_b):
            is_limitary = False
        else:
            msg = "ERROR: Both co_a and co_b must be defined."
            raise ValueError(msg)

        return cls(co_a, co_b.reshape(-1), is_limitary)
