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

from typing import Union, List

import numpy as np


def read_vector(inp: Union[str, List[float]]) -> np.ndarray:
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
