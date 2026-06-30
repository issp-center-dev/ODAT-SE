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
        TOMLDecodeError
            If the file is an invalid TOML document (raised on rank 0).
        exception.InputError
            On the other ranks, if the load failed on rank 0.

        Notes
        -----
        Only rank 0 reads the file. The load status is broadcast *before* the
        parsed data so that a failure on rank 0 does not leave the other ranks
        blocked forever on the data broadcast.
        """
        inp = {}
        if mpi.size() > 1:
            comm = mpi.comm()
            rank = mpi.rank()

            # Phase 1: rank 0 attempts the load; share the outcome with all
            # ranks. ``error_message`` is a plain (picklable) string so the
            # status broadcast itself can never fail and deadlock.
            error = None          # original exception, rank 0 only
            error_message = None  # status shared with every rank
            if rank == 0:
                try:
                    inp = toml.load(file_name)
                except Exception as e:
                    error = e
                    error_message = f"{type(e).__name__}: {e}"
            error_message = comm.bcast(error_message, root=0)

            if error_message is not None:
                if rank == 0:
                    raise error
                raise exception.InputError(
                    f"failed to load '{file_name}' on rank 0: {error_message}"
                )

            # Phase 2: broadcast the parsed data only when the load succeeded.
            inp = comm.bcast(inp, root=0)
        else:
            inp = toml.load(file_name)
        return cls(inp)
