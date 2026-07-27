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


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """
    Run an analysis specified by command-line style arguments.

    Parses the arguments, loads the input file, selects the algorithm and
    solver, executes the analysis, and returns the result of
    ``Algorithm.main()`` so that programmatic callers can use it as
    ``result = odatse.main(argv)``.

    This is the top-level boundary: user-facing errors
    (``odatse.exception.Error``) are reported and converted to a non-zero exit
    status via ``sys.exit(1)``. To handle these as exceptions instead, compose
    the lower-level API (``odatse.initialize`` /
    ``odatse.algorithm.choose_algorithm`` / ``Algorithm``) directly.

    Note: do not use this function as a console-script entry point; use
    ``cli()``, which converts the result to an exit status.
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
        # rank-local errors (see exception.Error.rank_local) exist only on the
        # rank that failed, so gating on rank 0 would silence them entirely
        if e.rank_local:
            prefix = f"[rank {odatse.mpi.rank()}] " if odatse.mpi.size() > 1 else ""
            print(f"{prefix}ERROR: {e}", file=sys.stderr)
        elif odatse.mpi.rank() == 0:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """
    Console-script entry point for the ``odatse`` command.

    Runs :func:`main` and returns exit status 0 on success, discarding the
    result object (``sys.exit()`` would interpret a non-integer return value
    as a failure). Errors are reported inside :func:`main` and terminate the
    process with exit status 1.
    """
    main(argv)
    return 0
