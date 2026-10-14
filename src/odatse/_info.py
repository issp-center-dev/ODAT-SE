# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections.abc import MutableMapping
from typing import Optional
from pathlib import Path
from fnmatch import fnmatch

from .util import io as shared_io
from . import exception


class Info:
    """
    A class to represent the information structure for the data-analysis software.
    """

    base: dict
    algorithm: dict
    solver: dict
    runner: dict

    # perf_counter() timestamp recorded by odatse.initialize(), used as the
    # start of the "init" phase in time.log. None when the Info object is
    # constructed directly without going through initialize().
    _start_time: Optional[float] = None

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
                    f"section {section} does not appear in input"
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
    def from_file(cls, file_name, **kwargs):
        """
        Create an Info object from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to load the information from.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Info
            An Info object initialized with the data from the file.

        Raises
        ------
        exception.LoadError
            On every rank, when the file could not be read or parsed on rank 0
            (the original exception is attached as ``__cause__`` there).

        Notes
        -----
        Only rank 0 reads the file; the result is distributed with
        ``odatse.util.io.load_toml()``, which shares the outcome of the read
        with every rank so that a failure on rank 0 does not leave the other
        ranks blocked on the data broadcast.
        """
        inp = shared_io.load_toml(file_name)
        return cls(inp)
