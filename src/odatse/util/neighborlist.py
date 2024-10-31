# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing
from typing import List, Set
from os import PathLike

import sys
import itertools

import numpy as np

from odatse import mpi

try:
    from tqdm import tqdm

    has_tqdm = True
except:
    has_tqdm = False


class Cells:
    """
    A class to represent a grid of cells for spatial partitioning.
    """

    cells: List[Set[int]]
    dimension: int
    mins: np.ndarray
    maxs: np.ndarray
    Ns: np.ndarray
    ncell: int
    cellsize: float

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, cellsize: float):
        """
        Initialize the Cells object.

        Parameters
        ----------
        mins : np.ndarray
            The minimum coordinates of the grid.
        maxs : np.ndarray
            The maximum coordinates of the grid.
        cellsize : float
            The size of each cell.
        """
        self.dimension = len(mins)
        self.mins = mins
        Ls = (maxs - mins) * 1.001
        self.Ns = np.ceil(Ls / cellsize).astype(np.int64)
        self.maxs = self.mins + cellsize * self.Ns
        self.cellsize = cellsize
        self.ncell = typing.cast(int, np.prod(self.Ns))
        self.cells = [set() for _ in range(self.ncell)]

    def coord2cellindex(self, x: np.ndarray) -> int:
        """
        Convert coordinates to a cell index.

        Parameters
        ----------
        x : np.ndarray
            The coordinates to convert.

        Returns
        -------
        int
            The index of the cell.
        """
        return self.cellcoord2cellindex(self.coord2cellcoord(x))

    def coord2cellcoord(self, x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates to cell coordinates.

        Parameters
        ----------
        x : np.ndarray
            The coordinates to convert.

        Returns
        -------
        np.ndarray
            The cell coordinates.
        """
        return np.floor((x - self.mins) / self.cellsize).astype(np.int64)

    def cellcoord2cellindex(self, ns: np.ndarray) -> int:
        """
        Convert cell coordinates to a cell index.

        Parameters
        ----------
        ns : np.ndarray
            The cell coordinates to convert.

        Returns
        -------
        int
            The index of the cell.
        """
        index = 0
        oldN = 1
        for n, N in zip(ns, self.Ns):
            index *= oldN
            index += n
            oldN = N
        return index

    def cellindex2cellcoord(self, index: int) -> np.ndarray:
        """
        Convert a cell index to cell coordinates.

        Parameters
        ----------
        index : int
            The index of the cell.

        Returns
        -------
        np.ndarray
            The cell coordinates.
        """
        ns = np.zeros(self.dimension, dtype=np.int64)
        for d in range(self.dimension):
            d = self.dimension - d - 1
            N = self.Ns[d]
            ns[d] = index % N
            index = index // N
        return ns

    def out_of_bound(self, ns: np.ndarray) -> bool:
        """
        Check if cell coordinates are out of bounds.

        Parameters
        ----------
        ns : np.ndarray
            The cell coordinates to check.

        Returns
        -------
        bool
            True if out of bounds, False otherwise.
        """
        if np.any(ns < 0):
            return True
        if np.any(ns >= self.Ns):
            return True
        return False

    def neighborcells(self, index: int) -> List[int]:
        """
        Get the indices of neighboring cells.

        Parameters
        ----------
        index : int
            The index of the cell.

        Returns
        -------
        List[int]
            The indices of the neighboring cells.
        """
        neighbors: List[int] = []
        center_coord = self.cellindex2cellcoord(index)
        for diff in itertools.product([-1, 0, 1], repeat=self.dimension):
            other_coord = center_coord + np.array(diff)
            if self.out_of_bound(other_coord):
                continue
            other_coord_index = self.cellcoord2cellindex(other_coord)
            neighbors.append(other_coord_index)
        return neighbors

def make_neighbor_list_cell(
    X: np.ndarray,
    radius: float,
    allow_selfloop: bool,
    show_progress: bool,
    comm: mpi.Comm = None,
) -> List[List[int]]:
    """
    Create a neighbor list using cell-based spatial partitioning.

    Parameters
    ----------
    X : np.ndarray
        The coordinates of the points.
    radius : float
        The radius within which neighbors are considered.
    allow_selfloop : bool
        Whether to allow self-loops in the neighbor list.
    show_progress : bool
        Whether to show a progress bar.
    comm : mpi.Comm, optional
        The MPI communicator.

    Returns
    -------
    List[List[int]]
        The neighbor list.
    """
    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.size
        mpirank = comm.rank

    mins = typing.cast(np.ndarray, X.min(axis=0))
    maxs = typing.cast(np.ndarray, X.max(axis=0))
    cells = Cells(mins, maxs, radius * 1.001)
    npoints = X.shape[0]
    for n in range(npoints):
        xs = X[n, :]
        index = cells.coord2cellindex(xs)
        cells.cells[index].add(n)

    points = np.array_split(range(npoints), mpisize)[mpirank]
    npoints_local = len(points)
    nnlist: List[List[int]] = [[] for _ in range(npoints_local)]
    if show_progress and mpirank == 0:
        if has_tqdm:
            desc = "rank 0" if mpisize > 1 else None
            ns = tqdm(points, desc=desc)
        else:
            print("WARNING: cannot show progress because tqdm package is not available")
            ns = points
    else:
        ns = points
    for n in ns:
        xs = X[n, :]
        cellindex = cells.coord2cellindex(xs)
        for neighborcell in cells.neighborcells(cellindex):
            for other in cells.cells[neighborcell]:
                if (not allow_selfloop) and n == other:
                    continue
                ys = X[other, :]
                r = np.linalg.norm(xs - ys)
                if r <= radius:
                    nnlist[n - points[0]].append(other)
    if mpisize > 1:
        nnlist = list(itertools.chain.from_iterable(comm.allgather(nnlist)))
    return nnlist


def make_neighbor_list_naive(
    X: np.ndarray,
    radius: float,
    allow_selfloop: bool,
    show_progress: bool,
    comm: mpi.Comm = None,
) -> List[List[int]]:
    """
    Create a neighbor list using a naive all-pairs approach.

    Parameters
    ----------
    X : np.ndarray)
        The coordinates of the points.
    radius : float
        The radius within which neighbors are considered.
    allow_selfloop : bool
        Whether to allow self-loops in the neighbor list.
    show_progress : bool
        Whether to show a progress bar.
    comm : mpi.Comm, optional
        The MPI communicator.

    Returns
    -------
    List[List[int]]
        The neighbor list.
    """
    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.size
        mpirank = comm.rank

    npoints = X.shape[0]
    points = np.array_split(range(npoints), mpisize)[mpirank]
    npoints_local = len(points)
    nnlist: List[List[int]] = [[] for _ in range(npoints_local)]
    if show_progress and mpirank == 0:
        if has_tqdm:
            desc = "rank 0" if mpisize > 1 else None
            ns = tqdm(points, desc=desc)
        else:
            print("WARNING: cannot show progress because tqdm package is not available")
            ns = points
    else:
        ns = points
    for n in ns:
        xs = X[n, :]
        for m in range(npoints):
            if (not allow_selfloop) and n == m:
                continue
            ys = X[m, :]
            r = np.linalg.norm(xs - ys)
            if r <= radius:
                nnlist[n - points[0]].append(m)
    if mpisize > 1:
        nnlist = list(itertools.chain.from_iterable(comm.allgather(nnlist)))
    return nnlist


def make_neighbor_list(
    X: np.ndarray,
    radius: float,
    allow_selfloop: bool = False,
    check_allpairs: bool = False,
    show_progress: bool = False,
    comm: mpi.Comm = None,
) -> List[List[int]]:
    """
    Create a neighbor list for given points.

    Parameters
    ----------
    X : np.ndarray
        The coordinates of the points.
    radius : float
        The radius within which neighbors are considered.
    allow_selfloop : bool
        Whether to allow self-loops in the neighbor list.
    check_allpairs : bool
        Whether to use the naive all-pairs approach.
    show_progress : bool
        Whether to show a progress bar.
    comm : mpi.Comm, optional
        The MPI communicator.

    Returns
    -------
    List[List[int]]
        The neighbor list.
    """
    if check_allpairs:
        return make_neighbor_list_naive(
            X,
            radius,
            allow_selfloop=allow_selfloop,
            show_progress=show_progress,
            comm=comm,
        )
    else:
        return make_neighbor_list_cell(
            X,
            radius,
            allow_selfloop=allow_selfloop,
            show_progress=show_progress,
            comm=comm,
        )


def load_neighbor_list(filename: PathLike, nnodes: int = None) -> List[List[int]]:
    """
    Load a neighbor list from a file.

    Parameters
    ----------
    filename : PathLike
        The path to the file containing the neighbor list.
    nnodes : int, optional
        The number of nodes. If None, it will be determined from the file.

    Returns
    -------
    List[List[int]]
        The neighbor list.
    """
    if nnodes is None:
        nnodes = 0
        with open(filename) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if len(line) == 0:
                    continue
                nnodes += 1

    neighbor_list: List[List[int]] = [[] for _ in range(nnodes)]
    with open(filename) as f:
        for line in f:
            line = line.strip().split("#")[0]
            if len(line) == 0:
                continue
            words = line.split()
            i = int(words[0])
            nn = [int(w) for w in words[1:]]
            neighbor_list[i] = nn
    return neighbor_list


def write_neighbor_list(
    filename: str,
    nnlist: List[List[int]],
    radius: float = None,
    unit: np.ndarray = None,
):
    """
    Write the neighbor list to a file.

    Parameters
    ----------
    filename : str
        The path to the output file.
    nnlist : List[List[int]]
        The neighbor list to write.
    radius : float, optional
        The neighborhood radius. Defaults to None.
    unit : np.ndarray, optional
        The unit for each coordinate. Defaults to None.
    """
    with open(filename, "w") as f:
        if radius is not None:
            f.write(f"# radius = {radius}\n")
        if unit is not None:
            f.write(f"# unit =")
            for u in unit:
                f.write(f" {u}")
            f.write("\n")
        for i, nn in enumerate(nnlist):
            f.write(str(i))
            for o in sorted(nn):
                f.write(f" {o}")
            f.write("\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Make neighbor-list file from mesh-data file",
        epilog="""
Note:
  - The first column of an input file will be ignored.
  - UNIT is used for changing aspect ratio (a kind of normalization)
    - Each coodinate will be divided by the corresponding unit
    - UNIT is given as a string separated with white space
      - For example, -u "1.0 0.5 1.0" for 3D case
  - tqdm python package is required to show progress bar
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument(
        "-o", "--output", type=str, default="neighborlist.txt", help="output file"
    )
    parser.add_argument(
        "-r", "--radius", type=float, default=1.0, help="neighborhood radius"
    )
    parser.add_argument(
        "-u",
        "--unit",
        default=None,
        help="length unit for each coordinate (see Note)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Do not show progress bar"
    )
    parser.add_argument("--allow-selfloop", action="store_true", help="allow self loop")
    parser.add_argument(
        "--check-allpairs",
        action="store_true",
        help="check all pairs (bruteforce algorithm)",
    )

    args = parser.parse_args()

    inputfile = args.input
    outputfile = args.output
    radius = args.radius

    X = np.zeros((0, 0))

    if mpi.rank() == 0:
        X = np.loadtxt(inputfile)

    if mpi.size() > 1:
        sh = mpi.comm().bcast(X.shape, root=0)
        if mpi.rank() != 0:
            X = np.zeros(sh)
        mpi.comm().Bcast(X, root=0)

    D = X.shape[1] - 1

    if args.unit is None:
        unit = np.ones(D, dtype=float)
    else:
        unit = np.array([float(w) for w in args.unit.split()])
        if len(unit) != D:
            print(f"--unit option expects {D} floats as a string but {len(unit)} given")
            sys.exit(1)
    X = X[:, 1:] / unit

    nnlist = make_neighbor_list(
        X,
        radius,
        allow_selfloop=args.allow_selfloop,
        check_allpairs=args.check_allpairs,
        show_progress=(not args.quiet),
        comm=mpi.comm(),
    )

    write_neighbor_list(outputfile, nnlist, radius=radius, unit=unit)
