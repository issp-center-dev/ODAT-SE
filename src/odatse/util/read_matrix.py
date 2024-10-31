# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, List

import numpy as np


def read_vector(inp: Union[str, List[float]]) -> np.ndarray:
    """
    Converts an input string or list of floats into a numpy array vector.

    Parameters
    ----------
    inp : Union[str, List[float]]
        Input data, either as a space-separated string of numbers or a list of floats.

    Returns
    -------
    np.ndarray
        A numpy array representing the vector.

    Raises
    ------
    RuntimeError
        If the input is not a vector.
    """
    if isinstance(inp, str):
        vlist = [float(w) for w in inp.split()]
    else:
        vlist = inp
    v = np.array(vlist)
    if v.ndim > 1:
        msg = f"input is not vector ({inp})"
        raise RuntimeError(msg)
    return v

def read_matrix(inp: Union[str, List[List[float]]]) -> np.ndarray:
    """
    Converts an input string or list of lists of floats into a numpy array matrix.

    Parameters
    ----------
    inp : Union[str, List[List[float]]]
        Input data, either as a string with rows of space-separated numbers or a list of lists of floats.

    Returns
    -------
    np.ndarray
        A numpy array representing the matrix.

    Raises
    ------
    RuntimeError
        If the input is not a matrix.
    """
    if isinstance(inp, str):
        Alist: List[List[float]] = []
        for line in inp.split("\n"):
            if not line.strip():  # empty
                continue
            Alist.append([float(w) for w in line.strip().split()])
    else:
        Alist = inp
    A = np.array(Alist)
    if A.size == 0 or A.ndim == 2:
        return A
    msg = f"input is not matrix ({inp})"
    raise RuntimeError(msg)
