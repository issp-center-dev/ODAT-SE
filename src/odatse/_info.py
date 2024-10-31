# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import MutableMapping, Optional
from pathlib import Path
from fnmatch import fnmatch

from .util import toml
from . import mpi
from . import exception


class Info:
    """
    A class to represent the information structure for the data-analysis software.
    """

    base: dict
    algorithm: dict
    solver: dict
    runner: dict

    def __init__(self, d: Optional[MutableMapping] = None):
        """
        Initialize the Info object.

        Parameters
        ----------
        d : MutableMapping (optional)
            A dictionary to initialize the Info object.
        """
        if d is not None:
            self.from_dict(d)
        else:
            self._cleanup()

    def from_dict(self, d: MutableMapping) -> None:
        """
        Initialize the Info object from a dictionary.

        Parameters
        ----------
        d : MutableMapping
            A dictionary containing the information to initialize the Info object.

        Raises
        ------
        exception.InputError
            If any required section is missing in the input dictionary.
        """
        for section in ["base", "algorithm", "solver"]:
            if section not in d:
                raise exception.InputError(
                    f"ERROR: section {section} does not appear in input"
                )
        self._cleanup()
        self.base = d["base"]
        self.algorithm = d["algorithm"]
        self.solver = d["solver"]
        self.runner = d.get("runner", {})

        self.base["root_dir"] = (
            Path(self.base.get("root_dir", ".")).expanduser().absolute()
        )
        self.base["output_dir"] = (
            self.base["root_dir"]
            / Path(self.base.get("output_dir", ".")).expanduser()
        )

    def _cleanup(self) -> None:
        """
        Reset the Info object to its default state.
        """
        self.base = {}
        self.base["root_dir"] = Path(".").absolute()
        self.base["output_dir"] = self.base["root_dir"]
        self.algorithm = {}
        self.solver = {}
        self.runner = {}

    @classmethod
    def from_file(cls, file_name, fmt="", **kwargs):
        """
        Create an Info object from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to load the information from.
        fmt : str
            The format of the file (default is "").
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Info
            An Info object initialized with the data from the file.

        Raises
        ------
        ValueError
            If the file format is unsupported.
        """
        if fmt == "toml" or fnmatch(file_name.lower(), "*.toml"):
            inp = {}
            if mpi.rank() == 0:
                inp = toml.load(file_name)
            if mpi.size() > 1:
                inp = mpi.comm().bcast(inp, root=0)
            return cls(inp)
        else:
            raise ValueError("unsupported file format: {}".format(file_name))
