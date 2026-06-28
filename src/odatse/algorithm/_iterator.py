# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from odatse import mpi


class IteratorBase(object):
    # Fields saved/restored at checkpoint; subclasses declare their own list.
    _checkpoint_attrs: list[str] = []

    def __init__(self):
        # aliases
        self.mpicomm = mpi.algcomm()
        self.mpisize = mpi.algsize()
        self.mpirank = mpi.algrank()

        self._index_start = 0
        self._index_end = 0

    def _save_state(self) -> dict:
        """Return a snapshot of the iterator position as a plain dict."""
        return {attr: getattr(self, attr) for attr in type(self)._checkpoint_attrs}

    def _restore_state(self, d: dict) -> None:
        """Restore the iterator position from a snapshot dict."""
        for attr in type(self)._checkpoint_attrs:
            setattr(self, attr, d[attr])

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
    _checkpoint_attrs: list[str] = ["_i"]

    def __init__(self, xmin, xmax, xnum):
        super().__init__()

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
    _checkpoint_attrs: list[str] = ["_i", "_data"]

    def __init__(self, data):
        # input: rank 0 has all data
        # split data and distrubute to other ranks
        super().__init__()

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
                data_block = np.array_split(data, self.mpisize)
            else:
                data_block = None
            data = self.mpicomm.scatter(data_block, root=0)
            data = [[int(idx), *v] for idx, *v in data]
        return data


class DistributedListIterator(IteratorBase):
    _checkpoint_attrs: list[str] = ["_i", "_data"]

    def __init__(self, data, mpicomm=None):
        # all ranks have their own chunk of data
        super().__init__()

        self._data = data

        self._index_start = 0
        self._index_end = len(self._data)
        self._i = self._index_start

    def __next__(self):
        if self._i == self._index_end:
            raise StopIteration()
        data = self._data[self._i]
        self._i += 1
        return data[0], data[1:]


class RandomIterator(IteratorBase):
    _checkpoint_attrs: list[str] = ["_i"]

    def __init__(self, xmin, xmax, count, rng):
        super().__init__()

        self._rng = rng
        self._xmin = np.array(xmin)
        self._xmax = np.array(xmax)
        self._count = count

        self._set_index_range(self._count)
        self._i = self._index_start

        # # spin
        # for _ in range(self._index_start):
        #     self._rng.uniform(self._xmin, self._xmax)

    def __next__(self):
        if self._i == self._index_end:
            raise StopIteration()
        coord = self._rng.uniform(self._xmin, self._xmax)
        tag = self._i
        self._i += 1
        return tag, coord

    def _save_state(self) -> dict:
        state = super()._save_state()
        state["rng_state"] = self._rng.get_state()
        return state

    def _restore_state(self, d: dict) -> None:
        super()._restore_state(d)
        self._rng = np.random.RandomState()
        self._rng.set_state(d["rng_state"])
