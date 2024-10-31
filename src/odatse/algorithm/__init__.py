# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from ._algorithm import AlgorithmBase

def choose_algorithm(name):
    """
    Search for algorithm module by name

    Parameters
    ----------
    name : str
        name of the algorithm

    Returns
    -------
    module
        algorithm module
    """

    alg_table = {
        "mapper": "mapper_mpi",
        "minsearch": "min_search",
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
