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
