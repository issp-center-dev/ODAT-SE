# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing
from typing import Union, List, Iterable

import abc

import collections
import itertools
import numpy as np


class Resampler(abc.ABC):
    @abc.abstractmethod
    def reset(self, weights: Iterable):
        ...

    @abc.abstractmethod
    def sample(self, rs: np.random.RandomState, size=None) -> Union[int, np.ndarray]:
        ...


class BinarySearch(Resampler):
    """
    A resampler that uses binary search to sample based on given weights.
    """
    weights_accumulated: List[float]
    wmax: float

    def __init__(self, weights: Iterable):
        """
        Initialize the BinarySearch resampler with the given weights.

        Parameters
        ----------
        weights : Iterable
            An iterable of weights.
        """
        self.reset(weights)

    def reset(self, weights: Iterable):
        """
        Reset the resampler with new weights.

        Parameters
        ----------
        weights : Iterable
            An iterable of weights.
        """
        self.weights_accumulated = list(itertools.accumulate(weights))
        self.wmax = self.weights_accumulated[-1]

    @typing.overload
    def sample(self, rs: np.random.RandomState) -> int:
        """
        Sample a single index based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.

        Returns
        -------
        int
            A single sampled index.
        """
        ...

    @typing.overload
    def sample(self, rs: np.random.RandomState, size) -> np.ndarray:
        """
        Sample multiple indices based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.
        size :
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            An array of sampled indices.
        """
        ...

    def sample(self, rs: np.random.RandomState, size=None) -> Union[int, np.ndarray]:
        """
        Sample indices based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.
        size :
            The number of samples to generate. If None, a single sample is generated.

        Returns
        -------
        int or np.ndarray
            A single sampled index or an array of sampled indices.
        """
        if size is None:
            return self._sample(self.wmax * rs.rand())
        else:
            return np.array([self._sample(r) for r in self.wmax * rs.rand(size)])

    def _sample(self, r: float) -> int:
        """
        Perform a binary search to find the index corresponding to the given random number.

        Parameters
        ----------
        r : float
            A random number scaled by the maximum weight.

        Returns
        -------
        int
            The index corresponding to the random number.
        """
        return typing.cast(int, np.searchsorted(self.weights_accumulated, r))


class WalkerTable(Resampler):
    """
    A resampler that uses Walker's alias method to sample based on given weights.
    """
    N: int
    itable: np.ndarray
    ptable: np.ndarray

    def __init__(self, weights: Iterable):
        """
        Initialize the WalkerTable resampler with the given weights.

        Parameters
        ----------
        weights : Iterable
            An iterable of weights.
        """
        self.reset(weights)

    def reset(self, weights: Iterable):
        """
        Reset the resampler with new weights.

        Parameters
        ----------
        weights : Iterable
            An iterable of weights.
        """
        self.ptable = np.array(weights).astype(np.float64).flatten()
        self.N = len(self.ptable)
        self.itable = np.full(self.N, -1)
        mean = self.ptable.mean()
        self.ptable -= mean
        shorter = collections.deque([i for i, p in enumerate(self.ptable) if p < 0.0])
        longer = collections.deque([i for i, p in enumerate(self.ptable) if p >= 0.0])

        while len(longer) > 0 and len(shorter) > 0:
            ilong = longer[0]
            ishort = shorter.popleft()
            self.itable[ishort] = ilong
            self.ptable[ilong] += self.ptable[ishort]
            if self.ptable[ilong] <= 0.0:
                longer.popleft()
                shorter.append(ilong)
        self.ptable += mean
        self.ptable *= 1.0 / mean

    @typing.overload
    def sample(self, rs: np.random.RandomState) -> int:
        """
        Sample a single index based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.

        Returns
        -------
        int
            A single sampled index.
        """
        ...

    @typing.overload
    def sample(self, rs: np.random.RandomState, size) -> np.ndarray:
        """
        Sample multiple indices based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.
        size :
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            An array of sampled indices.
        """
        ...

    def sample(self, rs: np.random.RandomState, size=None) -> Union[int, np.ndarray]:
        """
        Sample indices based on the weights.

        Parameters
        ----------
        rs : np.random.RandomState
            A random state for generating random numbers.
        size :
            The number of samples to generate. If None, a single sample is generated.

        Returns
        -------
        int or np.ndarray
            A single sampled index or an array of sampled indices.
        """
        if size is None:
            r = rs.rand() * self.N
            return self._sample(r)
        else:
            r = rs.rand(size) * self.N
            i = np.floor(r).astype(np.int64)
            p = r - i
            ret = np.where(p < self.ptable[i], i, self.itable[i])
            return ret

    def _sample(self, r: float) -> int:
        """
        Perform a sampling operation based on the given random number.

        Parameters
        ----------
        r : float
            A random number scaled by the number of weights.

        Returns
        -------
        int
            The index corresponding to the random number.
        """
        i = int(np.floor(r))
        p = r - i
        if p < self.ptable[i]:
            return i
        else:
            return self.itable[i]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="resampling test")
    parser.add_argument(
        "-s", "--seed", type=int, default=12345, help="random number seed"
    )
    parser.add_argument("-m", "--method", default="walker", help="method to resample")
    parser.add_argument("-N", type=int, default=100000, help="Number of samples")

    args = parser.parse_args()

    rs = np.random.RandomState(args.seed)

    ps = [0.0, 1.0, 2.0, 3.0]
    # ps = rs.rand(5)
    S = np.sum(ps)
    if args.method == "walker":
        resampler = WalkerTable(ps)
    else:
        resampler = BinarySearch(ps)
    samples = resampler.sample(rs, args.N)

    print("#i result exact diff")
    for i, p in enumerate(ps):
        r = np.count_nonzero(samples == i) / args.N
        print(f"{i} {r} {p/S} {r - p/S}")
