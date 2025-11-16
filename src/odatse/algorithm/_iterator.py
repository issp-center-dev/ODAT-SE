# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Union, Dict, Optional, TYPE_CHECKING

from pathlib import Path
from io import open
import os

import numpy as np
from odatse import mpi


class IteratorBase(object):
    def __init__(self, mpicomm):
        if mpicomm is None:
            self.mpicomm = mpi.comm()
            self.mpisize = mpi.size()
            self.mpirank = mpi.rank()
        else:
            self.mpicomm = mpicomm
            self.mpisize = mpicomm.Get_size()
            self.mpirank = mpicomm.Get_rank()

        self._index_start = 0
        self._index_end = 0

    def __iter__(self):
        return self

    def _set_index_range(self, count):
        if self.mpisize > 1:
            v, r = divmod(count, self.mpisize)
            ns = [v + 1 if i < r else v for i in range(self.mpisize)]
            self._index_start = sum(ns[0:self.mpirank])
            self._index_end = self._index_start + ns[self.mpirank]
        else:
            self._index_start = 0
            self._index_end = count

    def size(self):
        return self._index_end - self._index_start


class MeshIterator(IteratorBase):
    def __init__(self, xmin, xmax, xnum, mpicomm=None):
        super().__init__(mpicomm)
        
        self._xlist = [np.linspace(l, h, n) for l, h, n in zip(xmin, xmax, xnum)]
        self._num = np.array(xnum)
        #self._stride = np.cumprod([1]+xnum[::-1])[::-1][1:]  # row major
        self._stride = np.cumprod([1]+xnum)[:-1]  # column major

        self._set_index_range(np.prod(xnum))
        self._i = self._index_start

    def __next__(self):
        if self._i == self._index_end:
            raise StopIteration()
        idx = self._i // self._stride % self._num
        coord = [x[i] for x, i in zip(self._xlist, idx)]
        tag = self._i
        self._i += 1
        return tag, coord


class ListIterator(IteratorBase):
    def __init__(self, data, mpicomm=None):
        # input: rank 0 has all data
        # split data and distrubute to other ranks
        super().__init__(mpicomm)

        self._data = self._setup(data)

        self._index_start = 0
        self._index_end = len(self._data)
        self._i = self._index_start

    def __next__(self):
        if self._i == self._index_end:
            raise StopIteration()
        data = self._data[self._i]
        self._i += 1
        return data[0], data[1:]

    def _setup(self, data):
        if self.mpisize > 1:
            if self.mpirank == 0:
                n, r = divmod(len(data), self.mpisize)
                ns = [n + 1 if p < r else n for p in range(self.mpisize)]
                _start = [sum(ns[:p]) for p in range(self.mpisize)]
                _end = [sum(ns[:p+1]) for p in range(self.mpisize)]
                data_block = [data[_start[p]:_end[p]] for p in range(self.mpisize)]
            else:
                data_block = None
            data = self.mpicomm.scatter(data_block, root=0)
        return data

class RandomIterator(IteratorBase):
    def __init__(self, xmin, xmax, count, rng, mpicomm=None):
        super().__init__(mpicomm)

        self._rng = rng
        self._xmin = np.array(xmin)
        self._xmax = np.array(xmax)
        self._count = count

        self._set_index_range(self._count)
        self._i = self._index_start

    def __next__(self):
        if self._i == self._index_end:
            raise StopIteration()
        coord = self._rng.uniform(self._xmin, self._xmax)
        tag = self._i
        self._i += 1
        return tag, coord
