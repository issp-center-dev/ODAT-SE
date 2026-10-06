# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional, Sequence

import odatse
import odatse.exception

def initialize(argv: Optional[Sequence[str]] = None):
    """
    Initialize for main function by parsing commandline arguments and loading input files

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. If None (default), sys.argv[1:] is used.
        Pass an explicit list to avoid modifying sys.argv when embedding
        odatse in a larger script that has its own argument parser.

    Returns
    -------
    Tuple(Info, str)
        an Info object having parameter values, and a run_mode string
    """
    import argparse
    import time

    # Record the start of the "init" phase for time.log. Everything from here
    # up to Algorithm.main() (argument parsing, input loading, solver and
    # algorithm construction) is attributed to "init".
    time_sta = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Open framework for data analysis"
    )
    parser.add_argument("inputfile", help="input file with TOML format")
    parser.add_argument("--version", action="version", version=odatse.__version__)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--init", action="store_true", help="initial start (default)")
    mode_group.add_argument("--resume", action="store_true", help="resume interrupted run")
    mode_group.add_argument("--cont", action="store_true", help="continue from previous run")

    parser.add_argument("--reset_rand", action="store_true", default=False, help="new random number series in resume or continue mode")
    
    parser.add_argument("--nalg", type=int, default=None, help="# of processes for search algorithm")
    parser.add_argument("--nsolve", type=int, default=None, help="# of processes for solver")

    args = parser.parse_args(argv)

    # setup() may already have been called, e.g. by a host program that
    # embeds ODAT-SE and confines it to its own communicator with
    # setup(comm=...). setup() keeps that configuration when --nalg/--nsolve
    # are not given, and otherwise checks that the requested layout agrees
    # with it. Invalid or conflicting values are input errors, reported by
    # odatse.main() on one line with exit status 1.
    try:
        odatse.mpi.setup(nalg=args.nalg, nsolve=args.nsolve)
    except ValueError as e:
        raise odatse.exception.InputError(f"invalid --nalg/--nsolve: {e}") from e
    except odatse.mpi.SetupConflictError as e:
        # only the explicit conflict check; an MPI failure inside setup()
        # (mpi4py's MPI.Exception is a RuntimeError too) propagates as is
        raise odatse.exception.InputError(
            f"--nalg/--nsolve conflict with the MPI layout already set up: {e}"
        ) from e


    if args.init is True:
        run_mode = "initial"
    elif args.resume is True:
        run_mode = "resume"
        if args.reset_rand is True:
            run_mode = "resume-resetrand"
    elif args.cont is True:
        run_mode = "continue"
        if args.reset_rand is True:
            run_mode = "continue-resetrand"
    else:
        run_mode = "initial"  # default

    info = odatse.Info.from_file(args.inputfile)
    info._start_time = time_sta

    return info, run_mode
