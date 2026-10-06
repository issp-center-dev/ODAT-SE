# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Sequence, Union, Any

from pathlib import Path
import warnings
import numpy as np

import odatse
from odatse.exception import InputError
from ._domain import DomainBase

def load_mesh_file(root_dir, info_param: dict, *, root_only: bool = False):
    """
    Read a mesh file on algorithm rank 0 and validate it on every algorithm rank.

    This is the single mesh-file reader, shared by ``MeshGrid`` (used by
    ``bayes``, ``exchange`` and ``pamc``) and by ``mapper``. It must be
    called on every rank of the algorithm layer (it is collective over
    ``algcomm``), and only there.

    Parameters
    ----------
    root_dir : Path
        Directory that a relative ``mesh_path`` is resolved against.
    info_param : dict
        ``mesh_path`` (required) and the optional ``comments``, ``delimiter``
        and ``skiprows`` passed to ``numpy.loadtxt``.
    root_only : bool
        If True, the rows are returned on algorithm rank 0 only (``None``
        elsewhere); otherwise they are broadcast to every algorithm rank.

    Returns
    -------
    np.ndarray or None
        2-D array of rows ``index x1 ... xD``.

    Raises
    ------
    odatse.exception.InputError
        on every algorithm rank, when the file cannot be read, has no data
        rows, or has no coordinate columns (fewer than two columns). The
        number of coordinates is not compared with the ``dimension`` of the
        algorithm: a mesh may hold the points in the solver's coordinates
        (see ``tests/transform``), and the solver dimension is checked when a
        point is evaluated.
    """
    if "mesh_path" not in info_param:
        raise InputError("mesh_path not defined")
    mesh_path = root_dir / Path(info_param["mesh_path"]).expanduser()

    comments = info_param.get("comments", "#")
    delimiter = info_param.get("delimiter", None)
    skiprows = info_param.get("skiprows", 0)

    # Read on one rank; the outcome (shape or error) is agreed on every
    # algorithm rank before anything else happens, so that a bad file makes
    # all of them raise instead of leaving the others in a collective.
    data = None
    outcome = None
    if odatse.mpi.algrank() == 0:
        try:
            if not mesh_path.exists():
                raise FileNotFoundError(f"mesh_path not found: {mesh_path}")
            with warnings.catch_warnings():
                # an empty file is reported below, not by numpy
                warnings.simplefilter("ignore", UserWarning)
                data = np.loadtxt(mesh_path, comments=comments, delimiter=delimiter,
                                  skiprows=skiprows, ndmin=2)
            outcome = ("ok", data.shape)
        except Exception as e:
            outcome = ("error", f"{type(e).__name__}: {e}")
    if odatse.mpi.algsize() > 1:
        outcome = odatse.mpi.algcomm().bcast(outcome, root=0)

    kind, detail = outcome
    if kind == "error":
        raise InputError(f"cannot read mesh file {mesh_path}: {detail}")
    nrows, ncols = detail
    if nrows == 0:
        raise InputError(f"mesh file {mesh_path}: no data rows")
    if ncols < 2:
        raise InputError(
            f"mesh file {mesh_path}: expected at least 2 columns "
            f"(index and at least one coordinate), got {ncols}"
        )

    if not root_only and odatse.mpi.algsize() > 1:
        data = odatse.mpi.algcomm().bcast(data, root=0)
    return data


class MeshGrid(DomainBase):
    """
    MeshGrid class for handling grid data for the data analysis framework.
    """

    # whole grid and local chunk: list of vectors.
    # These are initialised per-instance in __init__; declared here only as
    # type annotations (no shared class-level mutable list).
    grid: Sequence[Sequence[Union[int, float]]]
    grid_local: Sequence[Sequence[Union[int, float]]]

    def __init__(
        self,
        info: odatse.Info = None,
        *,
        param: dict[str, Any] = None,
    ):
        """
        Initialize the MeshGrid object.

        Parameters
        ----------
        info : Info, optional
            Information object containing algorithm parameters.
        param : dict, optional
            Dictionary containing parameters for setting up the grid.
        """
        super().__init__(info)

        # per-instance defaults so distinct MeshGrid objects never share a list
        self.grid = []
        self.grid_local = []

        if info:
            if "param" in info.algorithm:
                self._setup(info.algorithm["param"])
            else:
                raise ValueError("ERROR: algorithm.param not defined")
        elif param:
            self._setup(param)
        else:
            pass

    def do_split(self):
        """
        Split the grid data among MPI processes.
        """
        if odatse.mpi.run_on_algorithm():
            if odatse.mpi.algsize() > 1:
                _data = np.array_split(self.grid, odatse.mpi.algsize())[odatse.mpi.algrank()]
                self.grid_local = [[idx, *v] for idx, *v in _data]
            else:
                self.grid_local = self.grid
        else:
            self.grid_local = []

    def _setup(self, info_param):
        """
        Setup the grid based on provided parameters.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "mesh_path" in info_param:
            self._setup_from_file(info_param)
        else:
            self._setup_grid(info_param)

    def _setup_from_file(self, info_param):
        """
        Setup the grid from a file.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        # load mesh file (validated on every algorithm rank) and distribute
        if odatse.mpi.run_on_algorithm():
            _data = load_mesh_file(self.root_dir, info_param)
        else:
            _data = []

        self.grid = [[int(idx), *v] for idx, *v in _data]
        self.do_split()

    def _setup_grid(self, info_param):
        """
        Setup the grid based on min, max, and num lists.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = np.array(info_param["min_list"], dtype=float)

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = np.array(info_param["max_list"], dtype=float)

        if "num_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_list is not defined in the input")
        num_list = np.array(info_param["num_list"], dtype=int)

        if len(min_list) != len(max_list) or len(min_list) != len(num_list):
            raise ValueError("ERROR: lengths of min_list, max_list, num_list do not match")
        xs = [
            np.linspace(mn, mx, num=nm)
            for mn, mx, nm in zip(min_list, max_list, num_list)
        ]
        self.grid = [
            [idx, *v] for idx, v in enumerate(
                np.array(
                    np.meshgrid(*xs, indexing='xy')
                ).reshape(len(xs), -1).transpose()
            )
        ]
        self.do_split()

    def store_file(self, store_path, *, header=""):
        """
        Store the grid data to a file.

        Parameters
        ----------
        store_path
            Path to the file where the grid data will be stored.
        header
            Header to be included in the file.
        """
        #if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
        if odatse.mpi.run_on_algorithm():
            if odatse.mpi.algrank() == 0:
                np.savetxt(store_path, [[*v] for idx, *v in self.grid], header=header)

    @classmethod
    def from_file(cls, mesh_path):
        """
        Create a MeshGrid object from a file.

        Parameters
        ----------
        mesh_path
            Path to the file containing the grid data.

        Returns
        -------
        MeshGrid
            a MeshGrid object.
        """
        return cls(param={"mesh_path": mesh_path})

    @classmethod
    def from_dict(cls, param):
        """
        Create a MeshGrid object from a dictionary of parameters.

        Parameters
        ----------
        param
            Dictionary containing parameters for setting up the grid.

        Returns
        -------
        MeshGrid
            a MeshGrid object.
        """
        return cls(param=param)


if __name__ == "__main__":
    ms = MeshGrid.from_dict({
        'min_list': [0,0,0],
        'max_list': [1,1,1],
        'num_list': [5,5,5],
    })
    ms.store_file("meshfile.dat", header="sample mesh data")

    ms2 = MeshGrid.from_file("meshfile.dat")
    #ms2.do_split()

    if odatse.mpi.rank() == 0:
        print(ms2.grid)
    print(odatse.mpi.rank(), ms2.grid_local)

    ms2.store_file("meshfile2.dat", header="store again")
