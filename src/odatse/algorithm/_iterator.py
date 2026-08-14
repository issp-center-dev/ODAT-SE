# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from odatse import mpi
from odatse import exception


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
        self._i = 0
        # size of the whole point set this iterator was built for; used to
        # detect a resume whose input asks for a different number of points
        self._total_points = 0

    def _save_state(self) -> dict:
        """Return a snapshot of the iterator position as a plain dict."""
        state = {attr: getattr(self, attr) for attr in type(self)._checkpoint_attrs}
        state["_total_points"] = self._total_points
        return state

    def _restore_state(self, d: dict, mode: str = "resume") -> None:
        """Restore the iterator position from a snapshot dict.

        Attributes missing from the snapshot are left at the values computed
        by the constructor, so that checkpoints written by older versions
        (with a shorter ``_checkpoint_attrs`` list) can still be resumed.

        In ``"resume"`` mode the point set must be the one the checkpoint was
        written for. The index range is derived from the configured number of
        points and is either not checkpointed or overwritten here, so resuming
        against a different point set would evaluate a set of points that
        matches neither input.
        """
        total_now = self._total_points
        total_saved = d.get("_total_points", total_now)
        if mode == "resume" and total_now != total_saved:
            # InputError so the CLI reports it as a single ERROR: line on
            # rank 0 rather than a traceback per rank: the input file, not
            # the calculation, is what needs fixing
            raise exception.InputError(
                "cannot resume: the number of search points changed "
                "({} -> {}); resume continues the run the checkpoint was "
                "written for. Use --cont to extend a completed run, or "
                "--init to start again.".format(total_saved, total_now))

        for attr in type(self)._checkpoint_attrs:
            if attr in d:
                setattr(self, attr, d[attr])
        self._total_points = total_saved

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

    def position(self):
        """Number of points already consumed on this rank."""
        return self._i - self._index_start


class MeshIterator(IteratorBase):
    _checkpoint_attrs: list[str] = ["_i"]

    def __init__(self, xmin, xmax, xnum):
        super().__init__()

        self._xlist = [np.linspace(l, h, n) for l, h, n in zip(xmin, xmax, xnum)]
        self._num = np.array(xnum)
        #self._stride = np.cumprod([1]+xnum[::-1])[::-1][1:]  # row major
        self._stride = np.cumprod([1]+xnum)[:-1]  # column major

        self._total_points = int(np.prod(xnum))
        self._set_index_range(self._total_points)
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
        self._total_points = self._count_all(len(self._data))

    def _count_all(self, n: int) -> int:
        """Total number of points over all ranks (only rank 0 gets the list)."""
        return int(self.mpicomm.allreduce(n)) if self.mpisize > 1 else n

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

    def _restore_state(self, d: dict, mode: str = "resume") -> None:
        super()._restore_state(d, mode=mode)
        # The constructor may have generated a point set of a different size
        # (e.g. continue mode with a larger num_points); keep the end of the
        # index range consistent with the restored data.
        self._index_end = len(self._data)

    def _extend(self, data) -> None:
        """Append additional points for continue mode.

        Rank 0 passes the additional ``[tag, *coords]`` rows; the other ranks
        pass None. The rows are scattered across the ranks in the same way as
        in the constructor and appended to this rank's share, so iteration
        continues with the new points only.
        """
        if self._i != self._index_end:
            raise RuntimeError(
                "cannot continue: the checkpoint does not correspond to a "
                "completed run; resume it to completion first (--resume)")
        ext = self._setup(data)
        self._data = list(self._data) + [[int(idx), *v] for idx, *v in ext]
        self._index_end = len(self._data)
        self._total_points = self._count_all(len(self._data))


class RandomIterator(IteratorBase):
    # _count and the index range are checkpointed so that a run started in
    # continue mode (whose range is the extension segment, not the default
    # division of [0, count)) can itself be resumed.
    _checkpoint_attrs: list[str] = ["_i", "_count", "_index_start", "_index_end"]

    def __init__(self, xmin, xmax, count, rng):
        super().__init__()

        self._rng = rng
        self._xmin = np.array(xmin)
        self._xmax = np.array(xmax)
        self._count = count

        self._total_points = int(count)
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

    def _restore_state(self, d: dict, mode: str = "resume") -> None:
        super()._restore_state(d, mode=mode)
        self._rng = np.random.RandomState()
        self._rng.set_state(d["rng_state"])

    def _extend(self, new_count) -> None:
        """Switch to this rank's share of the additional points (continue mode).

        The previously evaluated points keep their tags [0, count); the
        additional new_count - count points are divided among the ranks
        independently and tagged [count, new_count), so tags stay unique.
        Coordinates are drawn from the restored RNG state, continuing the
        random stream of the previous run.
        """
        add = new_count - self._count
        if add < 0:
            raise RuntimeError(
                "cannot continue: num_points ({}) is smaller than in the "
                "previous run ({})".format(new_count, self._count))
        if self._i != self._index_end:
            raise RuntimeError(
                "cannot continue: the checkpoint does not correspond to a "
                "completed run; resume it to completion first (--resume)")
        v, r = divmod(add, self.mpisize)
        ns = [v + 1 if i < r else v for i in range(self.mpisize)]
        self._index_start = self._count + sum(ns[0:self.mpirank])
        self._index_end = self._index_start + ns[self.mpirank]
        self._i = self._index_start
        self._count = new_count
        self._total_points = new_count
