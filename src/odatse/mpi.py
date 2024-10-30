# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

try:
    from mpi4py import MPI
    Comm = MPI.Comm

    __comm = MPI.COMM_WORLD
    __size = __comm.size
    __rank = __comm.rank

    def comm() -> MPI.Comm:
        return __comm

    def size() -> int:
        return __size

    def rank() -> int:
        return __rank

    def enabled() -> bool:
        return True


except ImportError:
    Comm = None

    def comm() -> None:
        return None

    def size() -> int:
        return 1

    def rank() -> int:
        return 0

    def enabled() -> bool:
        return False
