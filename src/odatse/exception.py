# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

class Error(Exception):
    """Base class of exceptions in odatse

    Attributes
    ----------
    rank_local : bool
        True when the error occurred on this MPI rank specifically (e.g. a
        checkpoint I/O failure re-raised through the phase consensus
        protocol), as opposed to an error raised identically on every rank
        (e.g. a config error). The CLI boundary prints rank-local errors from
        the owning rank; all other errors are printed on rank 0 only.
    report_rank : int or None
        When the same error is raised on several ranks at once (a
        ``LoadError``, raised on every process of the scope of a read), the
        global rank that reports it. The CLI boundary prints the error from
        that rank only and stays silent on the others, whether or not they
        are marked ``rank_local``; ``None`` leaves the choice to
        ``rank_local``.
    """

    rank_local = False
    report_rank = None


class InputError(Error):
    """
    Exception raised for errors in inputs

    Parameters
    ----------
    message : str
        explanation
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CheckpointError(Error):
    """
    Exception raised for checkpoint save/restore failures

    Parameters
    ----------
    message : str
        explanation
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LoadError(InputError):
    """
    Exception raised when data read on one process for the others could not
    be read or distributed (``odatse.util.io``).

    Raised on *every* process of the scope of the read (the job, the
    algorithm ranks, or one solver group), so that no process is left
    waiting in the collective that distributes the data. On the process that
    performed the read the original exception is attached as ``__cause__``.
    ``report_rank`` is the global rank of that process, so that the
    ``odatse`` command reports the error once, from it, and not once per
    process of the scope. An ``InputError``, not a ``RuntimeError``:
    ``ignore_error`` never turns a missing data file into ``NaN``.

    Parameters
    ----------
    message : str
        explanation, naming what was read and the rank that failed
    """
