# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import numpy as np
from typing import Optional, Tuple

_NOMPI = os.environ.get("ODATSE_NOMPI", "0") != "0"

if not _NOMPI:
    try:
        from mpi4py import MPI
        _NOMPI = False
    except ImportError:
        _NOMPI = True


# ------------------------------------------------------------------ #
#  Checkpoint mixin
# ------------------------------------------------------------------ #

class _CheckpointMixin:
    """Mixin that provides checkpoint save/restore via the pickle protocol.

    Subclasses must implement __getstate__ to return a dict of integer-valued
    parallelism parameters. __setstate__ verifies the saved state against the
    current module-level singleton (_ctx) that has already been re-initialised
    by setup(), then copies its attributes.
    """

    def __getstate__(self) -> dict:
        raise NotImplementedError("Subclasses must implement __getstate__")

    def __setstate__(self, state: dict) -> None:
        """Restore from a checkpoint snapshot.

        Assumes that odatse.mpi.setup() has already been called in the current
        run. Raises RuntimeError if setup() has not been called, and ValueError
        if any saved value does not match the current configuration.
        """
        import odatse.mpi as _mod
        current = _mod._ctx

        if not current.ready():
            raise RuntimeError(
                "odatse.mpi.setup() must be called before restoring state"
            )

        current_state = current.__getstate__()
        mismatches = {
            key: (saved, current_state[key])
            for key, saved in state.items()
            if saved != current_state[key]
        }
        if mismatches:
            lines = [
                f"  {k}: saved={v[0]}, current={v[1]}"
                for k, v in mismatches.items()
            ]
            raise ValueError(
                "Parallelism configuration mismatch:\n" + "\n".join(lines)
            )

        self.__dict__.update(current.__dict__)


# ------------------------------------------------------------------ #
#  No-MPI stub implementation
# ------------------------------------------------------------------ #

class _NoMPIContext(_CheckpointMixin):
    """Stub used when MPI is not available or disabled (ODATSE_NOMPI=1).

    All accessors return values consistent with single-process execution.
    setup() accepts nalg, nsolve and comm but ignores them, and ready() is
    always True because there is nothing to partition.
    """

    def setup(self, *, nalg: Optional[int] = None, nsolve: Optional[int] = None,
              comm=None) -> None:
        pass

    def ready(self) -> bool:                return True
    def comm(self):                         return None
    def size(self) -> int:                  return 1
    def rank(self) -> int:                  return 0
    def solcomm(self):                      return None
    def solsize(self) -> int:               return 1
    def solrank(self) -> int:               return 0
    def algcomm(self):                      return None
    def algsize(self) -> int:               return 1
    def algrank(self) -> int:               return 0
    def run_on_algorithm(self) -> bool:     return True
    def enabled(self) -> bool:              return False

    def __getstate__(self) -> dict:
        return {"algsize": 1, "algrank": 0, "solsize": 1, "solrank": 0}


# ------------------------------------------------------------------ #
#  MPI implementation
# ------------------------------------------------------------------ #

if not _NOMPI:

    class _MPIContext(_CheckpointMixin):
        """MPI-enabled implementation.

        Manages three sets of communicators:

        * Global MPI      : comm / size / rank
        * Algorithm layer : algcomm / algsize / algrank
        * Solver layer    : solcomm / solsize / solrank

        Call setup() after MPI_Init to partition the global communicator,
        which is MPI.COMM_WORLD unless another intracommunicator is passed to
        setup(). Solver-layer and algorithm-layer accessors (including
        run_on_algorithm()) raise RuntimeError if called before setup();
        ready() tells whether setup() has been called. Calling setup() again
        with the same effective configuration is a no-op, and with a
        different one raises RuntimeError.
        """

        def __init__(self) -> None:
            self._ready: bool = False
            self._comm = MPI.COMM_WORLD
            self._nalg: Optional[int] = None
            self._nsolve: Optional[int] = None

            self._solcomm = MPI.COMM_SELF
            self._solsize: int = 1
            self._solrank: int = 0

            self._algcomm = self._comm
            self._algsize: int = self._comm.size
            self._algrank: int = self._comm.rank

        @staticmethod
        def _resolve_comm(comm):
            """Return the communicator to partition (MPI.COMM_WORLD by default)."""
            if comm is None:
                return MPI.COMM_WORLD
            if isinstance(comm, MPI.Comm) and comm == MPI.COMM_NULL:
                # MPI.COMM_NULL itself, or a communicator that has been freed
                raise ValueError("comm must not be a null communicator "
                                 "(MPI.COMM_NULL or a freed communicator)")
            if not isinstance(comm, MPI.Intracomm):
                raise TypeError(
                    f"comm must be an MPI intracommunicator, got {type(comm).__name__}"
                )
            return comm

        @staticmethod
        def _resolve_layout(nalg: Optional[int], nsolve: Optional[int],
                            total: int) -> Tuple[int, int]:
            """Validate nalg/nsolve against the process count and fill in the
            missing one, returning the effective (nalg, nsolve)."""
            if nalg is not None and nalg <= 0:
                raise ValueError(f"nalg must be a positive integer, got {nalg}")
            if nsolve is not None and nsolve <= 0:
                raise ValueError(f"nsolve must be a positive integer, got {nsolve}")

            if nalg is not None and nsolve is not None:
                if nalg * nsolve != total:
                    raise ValueError(
                        f"nalg * nsolve must equal the total number of MPI processes, "
                        f"but {nalg} * {nsolve} = {nalg * nsolve} != {total}"
                    )
            elif nalg is not None:
                if total % nalg != 0:
                    raise ValueError(
                        f"Total MPI processes ({total}) must be divisible by nalg ({nalg})"
                    )
                nsolve = total // nalg
            elif nsolve is not None:
                if total % nsolve != 0:
                    raise ValueError(
                        f"Total MPI processes ({total}) must be divisible by nsolve ({nsolve})"
                    )
                nalg = total // nsolve
            else:
                nalg = total
                nsolve = 1
            return nalg, nsolve

        def setup(self, *, nalg: Optional[int] = None, nsolve: Optional[int] = None,
                  comm=None) -> None:
            """Partition the global communicator.

            Parameters
            ----------
            nalg:
                Number of MPI processes for the search algorithm.
            nsolve:
                Number of MPI processes per solver group.
            comm:
                Intracommunicator to partition. Defaults to MPI.COMM_WORLD.
                It becomes the global communicator returned by comm(); the
                caller keeps ownership (it is never freed here) and must keep
                it alive while ODAT-SE is in use. setup() is collective over
                this communicator: every rank of it must call setup() with
                the same arguments.

            Exactly one of nalg/nsolve may be None; the missing value is
            derived from the total process count of the communicator. If both
            are None, all processes are assigned to the algorithm layer
            (nsolve=1).

            setup() may be called again. If the effective configuration (the
            same communicator, and the same nalg/nsolve after the derivation
            above) equals the current one, the call does nothing; otherwise
            RuntimeError is raised. Communicators are compared as MPI handles
            (mpi4py's ``==``), so two Python objects wrapping the same handle
            count as the same communicator, while a duplicate (``Dup()``)
            does not. All checks are local and happen before any collective,
            so raising here cannot leave other ranks blocked.
            """
            comm = self._resolve_comm(comm)

            if self._ready and comm != self._comm:
                raise RuntimeError(
                    "setup() has already been called with a different communicator"
                )

            nalg, nsolve = self._resolve_layout(nalg, nsolve, comm.size)

            if self._ready:
                if (nalg, nsolve) != (self._nalg, self._nsolve):
                    raise RuntimeError(
                        "setup() has already been called with a different layout: "
                        f"current nalg={self._nalg}, nsolve={self._nsolve}; "
                        f"requested nalg={nalg}, nsolve={nsolve}"
                    )
                return

            # The new communicators are built in locals and stored only once
            # every collective has succeeded, so that a failure below leaves
            # the context untouched (still not ready, comm() unchanged).

            # Solver intracommunicator: nsolve processes per group
            color = comm.rank // nsolve
            solcomm = comm.Split(color=color, key=comm.rank)
            solsize = solcomm.size
            assert solsize == nsolve
            solrank = solcomm.rank

            # Algorithm intracommunicator: one representative per solver group (solrank==0)
            world_group = comm.Get_group()
            alg_group = world_group.Incl([c * nsolve for c in range(nalg)])
            algcomm = comm.Create(alg_group)
            alg_group.Free()
            world_group.Free()
            if algcomm != MPI.COMM_NULL:
                algsize = algcomm.size
                algrank = algcomm.rank
                sr = np.array([algsize, algrank])
                solcomm.bcast(sr, root=0)
            else:
                algcomm = None
                sr = np.array([0, 0])
                sr = solcomm.bcast(sr, root=0)
                algsize, algrank = int(sr[0]), int(sr[1])

            self._comm = comm
            self._solcomm, self._solsize, self._solrank = solcomm, solsize, solrank
            self._algcomm, self._algsize, self._algrank = algcomm, algsize, algrank
            self._nalg, self._nsolve = nalg, nsolve
            self._ready = True

            # self._print_status()

        def ready(self) -> bool:
            """Return True once setup() has been called."""
            return self._ready

        def _require_ready(self) -> None:
            if not self._ready:
                raise RuntimeError("odatse.mpi.setup() has not been called")

        # --- Global MPI (available before setup, when they refer to
        #     MPI.COMM_WORLD; after setup(comm=...) they refer to that comm) ---

        def comm(self):
            return self._comm

        def size(self) -> int:
            return self._comm.size

        def rank(self) -> int:
            return self._comm.rank

        def enabled(self) -> bool:
            return True

        # --- Solver layer ---

        def solcomm(self):
            self._require_ready()
            return self._solcomm

        def solsize(self) -> int:
            self._require_ready()
            return self._solsize

        def solrank(self) -> int:
            self._require_ready()
            return self._solrank

        # --- Algorithm layer ---

        def algcomm(self):
            """Return the algorithm communicator, or None for solver-worker processes."""
            self._require_ready()
            return self._algcomm

        def algsize(self) -> int:
            """Return the algorithm communicator size (0 for solver-worker processes)."""
            self._require_ready()
            return self._algsize

        def algrank(self) -> int:
            """Return this process's rank in the algorithm communicator (broadcast to all solver workers)."""
            self._require_ready()
            return self._algrank

        def run_on_algorithm(self) -> bool:
            """Return True on the controller of a solver group (solrank == 0)."""
            self._require_ready()
            return self._solrank == 0

        # --- debug ---

        def _print_status(self):
            print("DEBUG: "
                  + f"global: size={self._comm.size}, rank={self._comm.rank}"
                  + "; "
                  + f"alg: comm={self._algcomm}, size={self._algsize}, rank={self._algrank}"
                  + "; "
                  + f"sol: comm={self._solcomm}, size={self._solsize}, rank={self._solrank}"
            )

        # --- Checkpoint ---

        def __getstate__(self) -> dict:
            self._require_ready()
            return {
                "algsize": self._algsize,
                "algrank": self._algrank,
                "solsize": self._solsize,
                "solrank": self._solrank,
            }

    _ctx = _MPIContext()

else:

    _ctx = _NoMPIContext()


# ------------------------------------------------------------------ #
#  Exception and message constants
# ------------------------------------------------------------------ #

class OtherAlgorithmProcessError(Exception):
    """Raised when an error occurs in another algorithm process.

    After catching this, the algorithm process should signal solver workers
    to finish and exit without printing a message.
    """
    def __init__(self) -> None:
        super().__init__()

MSG_ABORT    = -1
MSG_FINISHED =  0
MSG_EVALUATE =  1


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

__all__ = [
    "setup", "ready",
    "comm", "size", "rank",
    "solcomm", "solsize", "solrank",
    "algcomm", "algsize", "algrank",
    "run_on_algorithm",
    "enabled",
    "OtherAlgorithmProcessError",
    "MSG_ABORT", "MSG_FINISHED", "MSG_EVALUATE",
]

def setup(*, nalg=None, nsolve=None, comm=None):
    _ctx.setup(nalg=nalg, nsolve=nsolve, comm=comm)
def ready() -> bool:                    return _ctx.ready()
def comm():                             return _ctx.comm()
def size() -> int:                      return _ctx.size()
def rank() -> int:                      return _ctx.rank()
def solcomm():                          return _ctx.solcomm()
def solsize() -> int:                   return _ctx.solsize()
def solrank() -> int:                   return _ctx.solrank()
def algcomm():                          return _ctx.algcomm()
def algsize() -> int:                   return _ctx.algsize()
def algrank() -> int:                   return _ctx.algrank()
def run_on_algorithm() -> bool:         return _ctx.run_on_algorithm()
def enabled() -> bool:                  return _ctx.enabled()
