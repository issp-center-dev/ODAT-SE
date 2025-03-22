# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Pay attention to the dependencies and the order of imports!
# For example, Runner depends on solver.

from ._info import Info
from . import solver
from ._runner import Runner
from . import algorithm
from ._main import main
from ._initialize import initialize

__version__ = "3.0.1"
