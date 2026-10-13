# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

class Error(Exception):
    """Base class of exceptions in odatse

    Parameters
    ----------
    message : str
        explanation

    Attributes
    ----------
    message : str
        the explanation passed to the constructor
    rank_local : bool
        True when the error occurred on this MPI rank specifically (e.g. a
        checkpoint I/O failure re-raised through the phase consensus
        protocol), as opposed to an error raised identically on every rank
        (e.g. a config error). The CLI boundary prints rank-local errors from
        the owning rank; all other errors are printed on rank 0 only.
    """

    rank_local = False

    _NO_MESSAGE = object()

    def __init__(self, message=_NO_MESSAGE, *args) -> None:
        # An omitted message keeps Exception's argument-less behaviour
        # (args == (), as when the constructor was inherited), so that
        # Error() and pickling of such an instance are unchanged; the
        # subclasses document a message as required.
        if message is Error._NO_MESSAGE:
            super().__init__(*args)
            self.message = ""
        else:
            super().__init__(message, *args)
            self.message = message


class InputError(Error):
    """
    Exception raised for errors in inputs

    Parameters
    ----------
    message : str
        explanation
    """


class CheckpointError(Error):
    """
    Exception raised for checkpoint save/restore failures

    Parameters
    ----------
    message : str
        explanation
    """


class SolverError(Error):
    """
    Exception raised on a solver-group controller when ``solver.evaluate()``
    failed in the group in a way that must never be ignored: at least one
    worker rank failed and at least one of the failing ranks (worker or
    controller) raised something other than a ``RuntimeError``, or the
    controller itself raised ``SystemExit`` / ``KeyboardInterrupt`` (which
    cannot travel through the algorithm-layer consensus as they are).

    Failures that are ``RuntimeError`` on every failing rank are raised as
    ``SolverRuntimeError`` instead, so that ``ignore_error`` applies to
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


class SolverRuntimeError(SolverError, RuntimeError):
    """
    Exception raised on a solver-group controller when ``solver.evaluate()``
    failed on at least one worker rank of the group and every failing rank
    raised a ``RuntimeError``.

    Being a ``RuntimeError``, it is covered by ``ignore_error`` (see
    ``is_ignorable``) exactly as a ``RuntimeError`` on the controller alone.
    Being a ``SolverError``, it is reported by ``odatse.main()`` on one line
    from the failing controller, like every other error of the framework,
    instead of surfacing as a raw traceback on each controller.

    Parameters
    ----------
    message : str
        explanation, including the global rank(s) that failed
    """


def is_ignorable(error: BaseException) -> bool:
    """
    Whether ``[runner] ignore_error`` may turn this ``solver.evaluate()``
    failure into ``NaN``: only a ``RuntimeError``. This is the single
    definition of the policy, used by the controller's decision, by the
    solver-group status exchange and by the worker-side reporting.
    """
    return isinstance(error, RuntimeError)


def describe_error(error: BaseException) -> str:
    """
    ``"ExceptionType: message"`` for messages that must never fail to be
    built (a rank that raised while formatting would skip a collective the
    other ranks are entering).
    """
    try:
        msg = str(error)
    except BaseException:
        # also SystemExit / KeyboardInterrupt from a broken __str__: nothing
        # may escape here, the collective must be entered
        msg = "<unprintable exception>"
    return f"{type(error).__name__}: {msg}"
