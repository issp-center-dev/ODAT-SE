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
    """

    rank_local = False


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


class SolverError(Error):
    """
    Exception raised on a solver-group controller when ``solver.evaluate()``
    failed on at least one worker rank of the group and at least one of the
    failing ranks (worker or controller) raised something other than a
    ``RuntimeError``.

    Failures that are ``RuntimeError`` on every failing rank are re-raised as
    a plain ``RuntimeError`` instead, so that ``ignore_error`` applies to
    them, and a failure on the controller alone is re-raised unchanged. This
    class is deliberately *not* a ``RuntimeError``: a rank that died with,
    e.g., ``ValueError`` or ``MemoryError`` must not be silently turned into
    ``NaN``.

    Parameters
    ----------
    message : str
        explanation, including the global rank(s) that failed
    """

    rank_local = True

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
