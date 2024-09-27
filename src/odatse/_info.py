# -*- coding: utf-8 -*-

# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import MutableMapping, Optional
from pathlib import Path
from fnmatch import fnmatch

from .util import toml
from . import mpi
from . import exception


class Info:
    base: dict
    algorithm: dict
    solver: dict
    runner: dict

    def __init__(self, d: Optional[MutableMapping] = None):
        if d is not None:
            self.from_dict(d)
        else:
            self._cleanup()

    def from_dict(self, d: MutableMapping) -> None:
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
        self.base = {}
        self.base["root_dir"] = Path(".").absolute()
        self.base["output_dir"] = self.base["root_dir"]
        self.algorithm = {}
        self.solver = {}
        self.runner = {}

    @classmethod
    def from_file(cls, file_name, fmt="", **kwargs):
        if fmt == "toml" or fnmatch(file_name.lower(), "*.toml"):
            inp = {}
            if mpi.rank() == 0:
                inp = toml.load(file_name)
            if mpi.size() > 1:
                inp = mpi.comm().bcast(inp, root=0)
            return cls(inp)
        else:
            raise ValueError("unsupported file format: {}".format(file_name))
