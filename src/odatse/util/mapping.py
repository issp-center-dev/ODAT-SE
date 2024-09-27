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

import copy
import numpy as np

from .read_matrix import read_matrix, read_vector

# type hints
from typing import Optional

class MappingBase:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented


class TrivialMapping(MappingBase):
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


class Affine(MappingBase):
    A: Optional[np.ndarray]
    b: Optional[np.ndarray]

    def __init__(self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None):
        # copy arguments
        self.A = np.array(A) if A is not None else None
        self.b = np.array(b) if b is not None else None

        # check
        if self.A is not None:
            if not self.A.ndim == 2:
                raise ValueError("A is not a matrix")
        if self.b is not None:
            if not self.b.ndim == 1:
                raise ValueError("b is not a vector")
        if self.A is not None and self.b is not None:
            if not self.A.shape[0] == self.b.shape[0]:
                raise ValueError("shape of A and b mismatch")


    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.A is None:
            ret = copy.copy(x)
        else:
            ret = np.dot(self.A, x)
        if self.b is None:
            return ret
        else:
            return ret + self.b

    @classmethod
    def from_dict(cls, d):
        A: Optional[np.ndarray] = read_matrix(d.get("A", []))
        b: Optional[np.ndarray] = read_matrix(d.get("b", []))

        if A is None:
            pass
        elif A.size == 0:
            A = None
        else:
            if not A.ndim == 2:
                raise ValueError("A should be a matrix")

        if b is None:
            pass
        elif b.size == 0:
            b = None
        else:
            if not (b.ndim == 2 and b.shape[1] == 1):
                raise ValueError("b should be a column vector")
            if not (A is not None and b.shape[0] == A.shape[0]):
                raise ValueError("shape of A and b does not match")
            b = b.reshape(-1)

        return cls(A, b)
