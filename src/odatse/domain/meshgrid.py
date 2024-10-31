# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Dict, Union, Any

from pathlib import Path
import numpy as np

import odatse
from ._domain import DomainBase

class MeshGrid(DomainBase):
    """
    MeshGrid class for handling grid data for quantum beam diffraction experiments.
    """

    grid: List[Union[int, float]] = []
    grid_local: List[Union[int, float]] = []
    candicates: int

    def __init__(self, info: odatse.Info = None, *, param: Dict[str, Any] = None):
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
        if self.mpisize > 1:
            index = [idx for idx, *v in self.grid]
            index_local = np.array_split(index, self.mpisize)[self.mpirank]
            self.grid_local = [[idx, *v] for idx, *v in self.grid if idx in index_local]
        else:
            self.grid_local = self.grid

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

        self.ncandicates = len(self.grid)

    def _setup_from_file(self, info_param):
        """
        Setup the grid from a file.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "mesh_path" not in info_param:
            raise ValueError("ERROR: mesh_path not defined")
        mesh_path = self.root_dir / Path(info_param["mesh_path"]).expanduser()

        if not mesh_path.exists():
            raise FileNotFoundError("mesh_path not found: {}".format(mesh_path))

        comments = info_param.get("comments", "#")
        delimiter = info_param.get("delimiter", None)
        skiprows = info_param.get("skiprows", 0)

        if self.mpirank == 0:
            data = np.loadtxt(mesh_path, comments=comments, delimiter=delimiter, skiprows=skiprows)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # old format: index x1 x2 ... -> omit index
            data = data[:, 1:]
        else:
            data = None

        if self.mpisize > 1:
            data = odatse.mpi.comm().bcast(data, root=0)

        self.grid = [[idx, *v] for idx, v in enumerate(data)]

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
        if self.mpirank == 0:
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
    ms2.do_split()

    if odatse.mpi.rank() == 0:
        print(ms2.grid)
    print(odatse.mpi.rank(), ms2.grid_local)

    ms2.store_file("meshfile2.dat", header="store again")
