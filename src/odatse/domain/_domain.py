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

from typing import List, Dict, Union, Any

from pathlib import Path
import numpy as np

import odatse

class DomainBase:
    def __init__(self, info: odatse.Info = None):
        if info:
            self.root_dir = info.base["root_dir"]
            self.output_dir = info.base["output_dir"]
        else:
            self.root_dir = Path(".")
            self.output_dir = Path(".")

        self.mpisize = odatse.mpi.size()
        self.mpirank = odatse.mpi.rank()
            