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
    
    This class divides a spatial region into a grid of cells to efficiently
    find neighboring points within a specified radius. Each cell contains
    a set of point indices that fall within its spatial boundaries.
    
    Attributes
    ----------
    cells : List[Set[int]]
        List of sets, where each set contains indices of points in that cell.
    dimension : int
        Number of spatial dimensions.
    mins : np.ndarray
        Minimum coordinates of the grid in each dimension.
    maxs : np.ndarray
        Maximum coordinates of the grid in each dimension.
    Ns : np.ndarray
        Number of cells in each dimension.
    ncell : int
        Total number of cells in the grid.
    cellsize : float
        Size of each cell.
    
    Examples
    --------
    >>> mins = np.array([0.0, 0.0, 0.0])
    >>> maxs = np.array([10.0, 10.0, 10.0])
    >>> cells = Cells(mins, maxs, cellsize=2.0)
    >>> point_index = 0
    >>> point_coords = np.array([1.5, 3.2, 4.7])
    >>> cell_index = cells.coord2cellindex(point_coords)
    >>> cells.cells[cell_index].add(point_index)
    """

    cells: List[Set[int]]
    dimension: int
    mins: np.ndarray
    maxs: np.ndarray
    Ns: np.ndarray
    ncell: int
    cellsize: float

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, cellsize: float) -> None:
        """
        Initialize the Cells object.
        
        Parameters
        ----------
        mins : np.ndarray
            The minimum coordinates of the grid in each dimension.
        maxs : np.ndarray
            The maximum coordinates of the grid in each dimension.
        cellsize : float
            The size of each cell.
            
        Returns
        -------
        None
            This method initializes the object's attributes.
            
        Examples
        --------
        >>> mins = np.array([0.0, 0.0, 0.0])
        >>> maxs = np.array([10.0, 10.0, 10.0])
        >>> cells = Cells(mins, maxs, cellsize=2.0)
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
        Convert spatial coordinates to a cell index.
        
        This method transforms a point's spatial coordinates into the index
        of the cell containing it, by first converting to cell coordinates
        and then to a cell index.
        
        Parameters
        ----------
        x : np.ndarray
            The spatial coordinates of a point.
            
        Returns
        -------
        int
            The index of the cell containing the point.
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> cells.coord2cellindex(np.array([3.5, 4.2]))
        12
        """
        return self.cellcoord2cellindex(self.coord2cellcoord(x))

    def coord2cellcoord(self, x: np.ndarray) -> np.ndarray:
        """
        Convert spatial coordinates to cell coordinates.
        
        Cell coordinates are integer indices that identify the position of a cell
        within the grid along each dimension.
        
        Parameters
        ----------
        x : np.ndarray
            The spatial coordinates of a point.
            
        Returns
        -------
        np.ndarray
            The cell coordinates (integer indices for each dimension).
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> cells.coord2cellcoord(np.array([3.5, 4.2]))
        array([1, 2])
        """
        return np.floor((x - self.mins) / self.cellsize).astype(np.int64)

    def cellcoord2cellindex(self, ns: np.ndarray) -> int:
        """
        Convert cell coordinates to a cell index.
        
        This method converts multi-dimensional cell coordinates to a single
        index that uniquely identifies the cell in the flat cells list.
        
        Parameters
        ----------
        ns : np.ndarray
            The cell coordinates (integer indices for each dimension).
            
        Returns
        -------
        int
            The index of the cell in the flat cells list.
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> cells.cellcoord2cellindex(np.array([1, 2]))
        12
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
        
        This method is the inverse of cellcoord2cellindex and converts a flat cell
        index back to multi-dimensional cell coordinates.
        
        Parameters
        ----------
        index : int
            The index of the cell in the flat cells list.
            
        Returns
        -------
        np.ndarray
            The cell coordinates (integer indices for each dimension).
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> cells.cellindex2cellcoord(12)
        array([1, 2])
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
        
        This method verifies whether the given cell coordinates are within
        the valid range of the grid.
        
        Parameters
        ----------
        ns : np.ndarray
            The cell coordinates to check.
            
        Returns
        -------
        bool
            True if the coordinates are out of bounds, False otherwise.
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> cells.out_of_bound(np.array([-1, 2]))
        True
        >>> cells.out_of_bound(np.array([3, 4]))
        False
        """
        if np.any(ns < 0):
            return True
        if np.any(ns >= self.Ns):
            return True
        return False

    def neighborcells(self, index: int) -> List[int]:
        """
        Get the indices of neighboring cells, including the cell itself.
        
        This method returns the indices of all cells that are adjacent to the specified
        cell in all dimensions, along with the cell itself.
        
        Parameters
        ----------
        index : int
            The index of the cell.
            
        Returns
        -------
        List[int]
            The indices of the neighboring cells (including the cell itself).
            
        Examples
        --------
        >>> cells = Cells(np.array([0, 0]), np.array([10, 10]), 2.0)
        >>> neighbors = cells.neighborcells(12)
        >>> len(neighbors)  # 3x3 neighborhood in 2D
        9
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
    
    This function uses the Cells class to efficiently find neighboring points
    within the specified radius by only considering points in adjacent cells.
    
    Parameters
    ----------
    X : np.ndarray
        The coordinates of the points (N x D array where N is the number of points
        and D is the dimensionality).
    radius : float
        The radius within which points are considered neighbors.
    allow_selfloop : bool
        Whether to allow a point to be its own neighbor.
    show_progress : bool
        Whether to show a progress bar during computation.
    comm : mpi.Comm, optional
        The MPI communicator for parallel processing, by default None.
        
    Returns
    -------
    List[List[int]]
        A list of lists, where each inner list contains the indices of
        neighboring points for a given point.
        
    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    >>> neighbor_list = make_neighbor_list_cell(coords, radius=1.5, 
    ...                                        allow_selfloop=False, 
    ...                                        show_progress=False)
    >>> neighbor_list
    [[1, 2], [0, 2], [0, 1], []]
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

    if mpirank == 0 and show_progress and has_tqdm:
        desc = "rank 0" if mpisize > 1 else None
        ns = tqdm(points, desc=desc)
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

    nnlist = [sorted(nn) for nn in nnlist]
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
    
    This function computes the distance between all pairs of points to find
    neighbors within the specified radius. This is less efficient than the
    cell-based approach for large point sets but may be more straightforward.
    
    Parameters
    ----------
    X : np.ndarray
        The coordinates of the points (N x D array where N is the number of points
        and D is the dimensionality).
    radius : float
        The radius within which points are considered neighbors.
    allow_selfloop : bool
        Whether to allow a point to be its own neighbor.
    show_progress : bool
        Whether to show a progress bar during computation.
    comm : mpi.Comm, optional
        The MPI communicator for parallel processing, by default None.
        
    Returns
    -------
    List[List[int]]
        A list of lists, where each inner list contains the indices of
        neighboring points for a given point.
        
    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    >>> neighbor_list = make_neighbor_list_naive(coords, radius=1.5, 
    ...                                         allow_selfloop=False, 
    ...                                         show_progress=False)
    >>> neighbor_list
    [[1, 2], [0, 2], [0, 1], []]
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

    if mpirank == 0 and show_progress and has_tqdm:
        desc = "rank 0" if mpisize > 1 else None
        ns = tqdm(points, desc=desc)
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

    nnlist = [sorted(nn) for nn in nnlist]
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
    
    This function serves as a unified interface to create neighbor lists,
    choosing between cell-based or naive implementation based on the parameters.
    
    Parameters
    ----------
    X : np.ndarray
        The coordinates of the points (N x D array where N is the number of points
        and D is the dimensionality).
    radius : float
        The radius within which points are considered neighbors.
    allow_selfloop : bool, optional
        Whether to allow a point to be its own neighbor, by default False.
    check_allpairs : bool, optional
        Whether to use the naive all-pairs approach instead of the cell-based one,
        by default False.
    show_progress : bool, optional
        Whether to show a progress bar during computation, by default False.
    comm : mpi.Comm, optional
        The MPI communicator for parallel processing, by default None.
        
    Returns
    -------
    List[List[int]]
        A list of lists, where each inner list contains the indices of
        neighboring points for a given point.
        
    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    >>> neighbor_list = make_neighbor_list(coords, radius=1.5)
    >>> neighbor_list
    [[1, 2], [0, 2], [0, 1], []]
    
    >>> # Force all-pairs algorithm
    >>> neighbor_list = make_neighbor_list(coords, radius=1.5, check_allpairs=True)
    >>> neighbor_list
    [[1, 2], [0, 2], [0, 1], []]
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
    
    The file format is expected to have one line per node, with the first number
    being the node index and subsequent numbers being its neighbors.
    
    Parameters
    ----------
    filename : PathLike
        The path to the file containing the neighbor list.
    nnodes : int, optional
        The number of nodes. If None, it will be determined from the file
        by counting the number of non-empty lines.
        
    Returns
    -------
    List[List[int]]
        The neighbor list loaded from the file.
        
    Examples
    --------
    >>> # Example file content:
    >>> # 0 1 2
    >>> # 1 0 2
    >>> # 2 0 1
    >>> neighbor_list = load_neighbor_list("neighborlist.txt")
    >>> neighbor_list
    [[1, 2], [0, 2], [0, 1]]
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
) -> None:
    """
    Write the neighbor list to a file.
    
    The file format has one line per node, with the first number being the 
    node index and subsequent numbers being its neighbors. Optional metadata
    can be included as comments at the beginning of the file.
    
    Parameters
    ----------
    filename : str
        The path to the output file.
    nnlist : List[List[int]]
        The neighbor list to write.
    radius : float, optional
        The neighborhood radius, written as a comment in the file, by default None.
    unit : np.ndarray, optional
        The unit lengths for each coordinate dimension, written as a comment
        in the file, by default None.
        
    Returns
    -------
    None
        This function writes the neighbor list to a file.
        
    Examples
    --------
    >>> neighbor_list = [[1, 2], [0, 2], [0, 1]]
    >>> write_neighbor_list("neighborlist.txt", neighbor_list, radius=1.5,
    ...                    unit=np.array([1.0, 1.0, 1.0]))
    """
    with open(filename, "w") as f:
        if radius is not None:
            f.write(f"# radius = {radius}\n")
        if unit is not None:
            f.write(f"# unit =")
            for u in unit:
                f.write(f" {u}")
            f.write("\n")
        for idx, nn in enumerate(nnlist):
            f.write(" ".join(map(str, [idx, *nn])) + "\n")

def main() -> None:
    """
    Command-line utility for creating neighbor lists from mesh data files.
    
    This function parses command-line arguments and creates a neighbor list file
    from a given input mesh file.
    
    Returns
    -------
    None
        This function is the main entry point for the command-line utility.
    """
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
    parser.add_argument(
        "--progress", action="store_true", help="show progress bar"
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

    if (args.progress or not args.quiet) and not has_tqdm:
        print("WARNING: cannot show progress because tqdm package is not available")

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
        show_progress=(args.progress or not args.quiet),
        comm=mpi.comm(),
    )

    write_neighbor_list(outputfile, nnlist, radius=radius, unit=unit)
