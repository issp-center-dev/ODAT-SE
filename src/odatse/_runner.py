# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABCMeta, abstractmethod
import sys
import traceback

import numpy as np

import odatse
import odatse.util.read_matrix
import odatse.util.mapping
import odatse.util.limitation
from odatse.util.logger import Logger
from odatse.exception import InputError, SolverError, SolverRuntimeError, is_ignorable, describe_error

# type hints
from pathlib import Path
from typing import Optional, Tuple
from . import mpi


class Run(metaclass=ABCMeta):
    def __init__(self, comm=None):
        """
        Initialize the Run class.

        Parameters
        ----------
        comm : MPI.Comm
            MPI Communicator.
        """
        self.comm = comm

    @abstractmethod
    def submit(self, solver):
        """
        Abstract method to submit a solver.

        Parameters
        ----------
        solver : object
            Solver object to be submitted.
        """
        pass


# Entries of the solver-group status exchange are tagged, so that data of a
# solver collective that was mistakenly paired with the exchange (see
# Runner._evaluate_group) is not taken for a status: a solver's own
# allgather(None) would otherwise look like "every rank succeeded".
_STATUS_TAG = "odatse.runner.status"


def _status_entry(is_ignorable_: bool = None, summary: str = None) -> tuple:
    """``(tag, None)`` for success, ``(tag, (is_ignorable, summary))`` otherwise."""
    return (_STATUS_TAG, None if summary is None else (is_ignorable_, summary))


def _is_status_entry(entry) -> bool:
    """True for an entry made by _status_entry(). Foreign data (anything a
    solver collective may carry, e.g. numpy arrays whose == is elementwise)
    must give False without raising."""
    try:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return False
        tag, status = entry
        if not (isinstance(tag, str) and tag == _STATUS_TAG):
            return False
        if status is None:
            return True
        return (isinstance(status, tuple) and len(status) == 2
                and isinstance(status[0], bool) and isinstance(status[1], str))
    except Exception:
        return False


class Runner(object):
    #solver: "odatse.solver.SolverBase"
    logger: Logger

    def __init__(self,
                 solver,
                 info: Optional[odatse.Info] = None,
                 mapping = None,
                 limitation = None) -> None:
        """
        Initialize the Runner class.

        Parameters
        ----------
        solver : odatse.solver.SolverBase
            Solver object.
        info : Optional[odatse.Info]
            Information object.
        mapping : object, optional
            Mapping object.
        limitation : object, optional
            Limitation object.
        """
        self.solver = solver
        self.solver_name = solver.name
        self.logger = Logger(info)
        self.ignore_error = info.runner.get("ignore_error", False)

        if mapping is not None:
            self.mapping = mapping
        elif "mapping" in info.runner:
            info_mapping = info.runner["mapping"]
            # N.B.: only Affine mapping is supported at present
            self.mapping = odatse.util.mapping.Affine.from_dict(info_mapping)
        else:
            # trivial mapping
            self.mapping = odatse.util.mapping.TrivialMapping()

        if limitation is not None:
            self.limitation = limitation
        elif "limitation" in info.runner:
            info_limitation = info.runner["limitation"]
            self.limitation = odatse.util.limitation.Inequality.from_dict(info_limitation)
        else:
            self.limitation = odatse.util.limitation.Unlimited()

    def prepare(self, proc_dir: Path):
        """
        Prepare the logger with the given process directory.

        Parameters
        ----------
        proc_dir : Path
            Path to the process directory.
        """
        self.logger.prepare(proc_dir)

    def submit(
            self, x: np.ndarray, args: tuple = ()) -> float:
        """
        Submit the solver with the given parameters.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        args : tuple, optional
            Additional arguments.

        Returns
        -------
        float
            Result of the solver evaluation.
        """
        if self.limitation.judge(x):
            xp = self.mapping(x)

            assert xp.ndim == 1
            assert xp.shape[0] == self.solver.dimension

            if odatse.mpi.solsize() > 1:
                msg = np.array([odatse.mpi.MSG_EVALUATE])
                odatse.mpi.solcomm().Bcast(msg, root=0)
                odatse.mpi.solcomm().Bcast(xp, root=0)
                odatse.mpi.solcomm().bcast(args, root=0) # args is not array, so we use bcast instead of Bcast

            result, error = self._evaluate_group(xp, args)
            if error is not None:
                if self.ignore_error and is_ignorable(error):
                    result = np.nan
                else:
                    raise error

        else:
            result = np.inf
        self.logger.count(x, args, result)
        return result

    def serve(self, xp: np.ndarray, args: tuple = ()) -> None:
        """
        Worker-side counterpart of ``submit()``.

        Called by the framework on solver-worker ranks (``solrank() > 0``)
        with the ``xp`` / ``args`` broadcast by the controller. It evaluates
        the solver on this rank and takes part in the status exchange of the
        solver group; whether a failure is ignored or propagated is decided on
        the controller, so nothing is raised here (not even ``SystemExit`` or
        ``KeyboardInterrupt`` from ``evaluate()``: they are reported to the
        controller, which terminates the job).

        Parameters
        ----------
        xp : np.ndarray
            Input array (already mapped by the controller).
        args : tuple, optional
            Additional arguments.
        """
        self._evaluate_group(xp, args)

    def _evaluate_group(
            self, xp: np.ndarray, args: tuple) -> Tuple[float, Optional[BaseException]]:
        """
        Call ``solver.evaluate()`` on this rank and agree on the outcome
        across the solver group.

        Every rank of the solver group (controller and workers) calls this
        for the same ``xp`` / ``args``. After the local ``evaluate()`` the
        ranks exchange their success status in one ``allgather`` on
        ``solcomm``, so that a failure on any rank is seen by all of them and
        no rank is left blocked in the next control-signal broadcast.

        Returns
        -------
        result : float
            The local return value of ``evaluate()`` (``NaN`` if it raised).
        error : BaseException or None
            ``None`` when every rank succeeded. Otherwise the exception to be
            raised on the controller:

            * this rank's own exception, if it is the only failure
              (``SystemExit`` / ``KeyboardInterrupt`` on the controller are
              wrapped into a ``SolverError`` first);
            * a ``SolverRuntimeError`` (a ``RuntimeError``) summarising all
              failures, when every failing rank raised a ``RuntimeError`` (so
              ``ignore_error`` applies);
            * a ``SolverError`` otherwise (not covered by ``ignore_error``).

        Notes
        -----
        The status exchange is a collective on ``solcomm`` and is placed
        *after* ``evaluate()``. It therefore cannot rescue a solver that
        raises on some ranks *before* a collective inside ``evaluate()`` that
        the other ranks still enter; such a mismatch of collectives is the
        solver's responsibility (see the parallel-solver tutorial). When the
        exchange receives entries that are not status tuples, which is how
        that mismatch shows up here, the job is aborted instead of hanging.
        """
        own_error: Optional[BaseException] = None
        result = np.nan
        try:
            result = self.solver.evaluate(xp, args)
        except BaseException as e:
            # Also SystemExit (sys.exit() inside evaluate) and
            # KeyboardInterrupt: they must take part in the status exchange
            # below, or the other ranks of the group wait in it forever. They
            # are never ignorable (not RuntimeError), so they terminate the job.
            own_error = e

        if odatse.mpi.solsize() == 1:
            # no group to agree with: SystemExit / KeyboardInterrupt keep
            # their usual meaning
            if own_error is not None and not isinstance(own_error, Exception):
                raise own_error
            return result, own_error

        # One tagged entry per rank: (tag, None) on success,
        # (tag, (is_ignorable, summary)) on failure. Only plain Python types
        # are exchanged, so that the collective cannot fail on an exception
        # object that does not pickle.
        if own_error is None:
            own_status = _status_entry()
        else:
            own_status = _status_entry(
                is_ignorable(own_error),
                f"[rank {odatse.mpi.rank()}] {describe_error(own_error)}",
            )
        statuses = odatse.mpi.solcomm().allgather(own_status)

        # Entries without the tag mean the allgather was paired with a
        # collective of the solver itself, i.e. evaluate() raised on some
        # ranks before a solcomm collective the others still entered. The
        # group is desynchronised beyond repair; abort rather than hang.
        if not all(_is_status_entry(s) for s in statuses):
            print(f"[rank {odatse.mpi.rank()}] ERROR: mismatched collectives inside "
                  "solver.evaluate(): some ranks raised before a solcomm collective "
                  "that the others entered; aborting the job",
                  file=sys.stderr, flush=True)
            odatse.mpi.comm().Abort(1)
            raise RuntimeError("mismatched collectives inside solver.evaluate()")

        failures = [s[1] for s in statuses if s[1] is not None]

        if not failures:
            return result, None

        ignorable = all(is_ign for is_ign, _ in failures)

        if odatse.mpi.solrank() > 0:
            # The controller decides whether the failure is ignored. Print the
            # traceback here only when it will not be, so that ignore_error does
            # not flood stderr on solvers that fail routinely in some regions.
            # The wording differs from the MPI_Abort path in AlgorithmBase.main()
            # ("solver worker failed") so that a log tells the two apart.
            if own_error is not None and not (ignorable and self.ignore_error):
                traceback.print_exception(type(own_error), own_error, own_error.__traceback__)
                print(f"[rank {odatse.mpi.rank()}] ERROR: solver worker raised in evaluate(), "
                      f"reported to the controller: {describe_error(own_error)}",
                      file=sys.stderr, flush=True)
            return result, own_error

        # controller
        if own_error is not None and not isinstance(own_error, Exception):
            # SystemExit / KeyboardInterrupt on the controller: the phase
            # wrappers and the algorithm-layer consensus handle Exceptions
            # only, so hand it over as a SolverError (never ignorable) to
            # terminate the job cleanly on every algorithm rank
            wrapped = SolverError(
                f"solver.evaluate() raised {describe_error(own_error)} on the "
                f"controller (global rank {odatse.mpi.rank()})"
            )
            wrapped.__cause__ = own_error
            own_error = wrapped

        if len(failures) == 1 and own_error is not None:
            return result, own_error

        summary = (
            f"solver.evaluate() failed on {len(failures)} rank(s) of the solver group:\n"
            + "\n".join(f"  {msg}" for _, msg in failures)
        )
        error: BaseException = SolverRuntimeError(summary) if ignorable else SolverError(summary)
        if own_error is not None:
            error.__cause__ = own_error
        return result, error

    def post(self) -> None:
        """
        Write the logger data.
        """
        if odatse.mpi.solrank() == 0:
            self.logger.write()
