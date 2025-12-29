# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Custom exceptions for the ODAT-SE KKR module.
"""

from typing import List, Optional


class ConvergenceError(Exception):
    """
    Exception raised when KKR calculation fails to converge.

    This exception is raised when the AkaiKKR solver fails to achieve
    convergence after all retry attempts have been exhausted.

    Parameters
    ----------
    message : str
        Error message describing the convergence failure.
    ewidth_tried : Optional[List[float]]
        List of ewidth values that were tried before giving up.
    attempts : int
        Total number of attempts made.

    Attributes
    ----------
    message : str
        Error message describing the convergence failure.
    ewidth_tried : Optional[List[float]]
        List of ewidth values that were tried.
    attempts : int
        Total number of attempts made.

    Examples
    --------
    >>> raise ConvergenceError(
    ...     "Calculation did not converge",
    ...     ewidth_tried=[2.0, 2.5, 3.0],
    ...     attempts=3,
    ... )
    """

    def __init__(
        self,
        message: str,
        ewidth_tried: Optional[List[float]] = None,
        attempts: int = 0,
    ) -> None:
        """
        Initialize the ConvergenceError.

        Parameters
        ----------
        message : str
            Error message describing the convergence failure.
        ewidth_tried : Optional[List[float]]
            List of ewidth values that were tried before giving up.
        attempts : int
            Total number of attempts made.
        """
        super().__init__(message)
        self.message = message
        self.ewidth_tried = ewidth_tried or []
        self.attempts = attempts

    def __str__(self) -> str:
        """
        Return string representation of the error.

        Returns
        -------
        str
            Formatted error message including ewidth values tried.
        """
        base_msg = self.message
        if self.ewidth_tried:
            base_msg += f" (tried ewidth values: {self.ewidth_tried})"
        if self.attempts > 0:
            base_msg += f" after {self.attempts} attempt(s)"
        return base_msg

