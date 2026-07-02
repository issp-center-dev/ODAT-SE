# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional, Sequence

import sys
import odatse
from . import exception


def choose_solver(info):
    """
    Select the solver class for the given info.

    Parameters
    ----------
    info : odatse.Info
        Information object containing the solver configuration.

    Returns
    -------
    type
        The solver class.

    Raises
    ------
    odatse.exception.InputError
        If the solver name is not recognised.
    """
    solvername = info.solver["name"]
    if solvername == "analytical":
        from .solver.analytical import Solver
        return Solver
    raise exception.InputError(f"Unknown solver ({solvername})")


def main(argv: Optional[Sequence[str]] = None):
    """
    Command-line entry point for the data-analysis software.

    Parses command-line arguments, loads the input file, selects the algorithm
    and solver, and executes the analysis.

    This is the top-level boundary: user-facing errors
    (``odatse.exception.Error``) are reported and converted to a non-zero exit
    status. To handle these as exceptions instead, compose the lower-level API
    (``odatse.initialize`` / ``odatse.algorithm.choose_algorithm`` /
    ``Algorithm``) directly.
    """
    try:
        info, run_mode = odatse.initialize(argv)

        alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])

        Solver = choose_solver(info)
        solver = Solver(info)
        runner = odatse.Runner(solver, info)
        alg = alg_module.Algorithm(info, runner, run_mode=run_mode)

        return alg.main()
    except exception.Error as e:
        # Report on whichever rank raised the error, not only global rank 0.
        # Some failures are per-rank (e.g. a CheckpointError while resuming
        # from a corrupt status.pickle on one rank): gating on rank 0 would let
        # such a rank abort the job (nonzero exit) with no diagnostic anywhere,
        # while the other ranks exit quietly via OtherAlgorithmProcessError.
        # The rank prefix identifies the source under mpirun; for errors that
        # occur identically on all ranks (e.g. InputError) it simply repeats
        # the same message per rank.
        print(f"ERROR [rank {odatse.mpi.rank()}]: {e}", file=sys.stderr)
        sys.exit(1)
