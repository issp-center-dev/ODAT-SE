# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os

ODATSE_NOMPI = os.environ.get("ODATSE_NOMPI", "0")!="0"

if not ODATSE_NOMPI:
    try:
        from mpi4py import MPI
        ODATSE_NOMPI = False
    except ImportError:
        ODATSE_NOMPI = True

if ODATSE_NOMPI:
    Comm = None

    def setup(nalg, nsolve, nthreads):
        pass

    def comm():
        return None

    def size() -> int:
        return 1

    def rank() -> int:
        return 0

    def solcomm():
        return None

    def solsize() -> int:
        return 1

    def solrank() -> int:
        return 0

    def solthreads() -> int:
        return 1

    def algcomm():
        return None

    def algsize() -> int:
        return 1

    def algrank() -> int:
        return 0

    def color() -> int:
        return 0

    def enabled() -> bool:
        return False

else:
    global __comm, __size, __rank, __solcomm, __solsize, __solrank, __solthreads, __algcomm, __algsize, __algrank, __color

    Comm = MPI.Comm

    __comm = MPI.COMM_WORLD
    __size = __comm.size
    __rank = __comm.rank
    __solcomm = None
    __solsize = 1
    __solrank = 0
    __solthreads = 1
    __algcomm = None
    __algsize = 1
    __algrank = 0
    __color = 0
    
    def setup(nalg, nsolve, nthreads):
        global __comm, __size, __rank, __solcomm, __solsize, __solrank, __solthreads, __algcomm, __algsize, __algrank, __color

        if nthreads is not None:
            assert nthreads > 0
        if nalg is not None:
            assert nalg > 0
        if nsolve is not None:
            assert nsolve > 0
        if nalg is not None and nsolve is not None:
            assert nalg * nsolve == __comm.size
        elif nalg is not None and nsolve is None:
            assert __comm.size % nalg == 0
            nsolve = __comm.size // nalg
        elif nsolve is not None and nalg is None:
            assert __comm.size % nsolve == 0
            nalg = __comm.size // nsolve
        else: # both are None, default to parallelizing over search algorithm
            nalg = __comm.size
            nsolve = 1

        __color = __comm.rank // nsolve
        __solcomm = __comm.Split(color=__color, key=__comm.rank)
        __solsize = __solcomm.size
        __solrank = __solcomm.rank
        __solthreads = nthreads
        __algcomm = __comm.Create(__comm.Get_group().Incl([color * nsolve for color in range(nalg)]))
        __algsize = __algcomm.size if __algcomm != MPI.COMM_NULL else None
        __algrank = __algcomm.rank if __algcomm != MPI.COMM_NULL else None

    def comm():
        return __comm

    def size() -> int:
        return __size

    def rank() -> int:
        return __rank

    def solcomm():
        return __solcomm

    def solsize() -> int:
        return __solsize

    def solrank() -> int:
        return __solrank

    def solthreads() -> int:
        return __solthreads

    def algcomm():
        return __algcomm

    def algsize() -> int:
        return __algsize

    def algrank() -> int:
        return __algrank
    
    def color() -> int:
        return __color

    def enabled() -> bool:
        return True
