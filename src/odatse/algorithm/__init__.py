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

from ._algorithm import AlgorithmBase

def choose_algorithm(name):
    alg_table = {
        "mapper": "mapper_mpi",
    }

    try:
        import importlib
        alg_name = "odatse.algorithm.{}".format(alg_table.get(name, name))
        alg_module = importlib.import_module(alg_name)
    except ModuleNotFoundError as e:
        print("ERROR: {}".format(e))
        import sys
        sys.exit(1)

    return alg_module
