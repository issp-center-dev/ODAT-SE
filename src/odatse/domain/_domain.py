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
            
