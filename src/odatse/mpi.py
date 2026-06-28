# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import numpy as np
from typing import Optional

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

        if not getattr(current, "_ready", True):
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
    setup() accepts nalg and nsolve but ignores them.
    """

    def setup(self, *, nalg: Optional[int] = None, nsolve: Optional[int] = None) -> None:
        pass

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

        Manages two layers of parallelism:

        * Global MPI      : comm / size / rank
        * Algorithm layer : algcomm / algsize / algrank
        * Solver layer    : solcomm / solsize / solrank

        Call setup() exactly once after MPI_Init to partition the global
        communicator. Solver-layer and algorithm-layer accessors raise
        RuntimeError if called before setup().
        """

        def __init__(self) -> None:
            self._ready: bool = False
            self._comm = MPI.COMM_WORLD

            self._solcomm = MPI.COMM_SELF
            self._solsize: int = 1
            self._solrank: int = 0

            self._algcomm = MPI.COMM_WORLD
            self._algsize: int = self._comm.size
            self._algrank: int = self._comm.rank

        def setup(self, *, nalg: Optional[int] = None, nsolve: Optional[int] = None) -> None:
            """Partition the global communicator.

            Parameters
            ----------
            nalg:
                Number of MPI processes for the search algorithm.
            nsolve:
                Number of MPI processes per solver group.

            Exactly one of nalg/nsolve may be None; the missing value is
            derived from the total process count. If both are None, all
            processes are assigned to the algorithm layer (nsolve=1).
            """
            if self._ready:
                raise RuntimeError("setup() must be called only once")

            if nalg is not None and nalg <= 0:
                raise ValueError(f"nalg must be a positive integer, got {nalg}")
            if nsolve is not None and nsolve <= 0:
                raise ValueError(f"nsolve must be a positive integer, got {nsolve}")

            total = self._comm.size

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

            # Solver intracommunicator: nsolve processes per group
            color = self._comm.rank // nsolve
            self._solcomm = self._comm.Split(color=color, key=self._comm.rank)
            self._solsize = self._solcomm.size
            assert self._solsize == nsolve
            self._solrank = self._solcomm.rank

            # Algorithm intracommunicator: one representative per solver group (solrank==0)
            algcomm = self._comm.Create(
                self._comm.Get_group().Incl([c * nsolve for c in range(nalg)])
            )
            if algcomm != MPI.COMM_NULL:
                self._algcomm = algcomm
                self._algsize = algcomm.size
                self._algrank = algcomm.rank
                sr = np.array([self._algsize, self._algrank])
                self._solcomm.bcast(sr, root=0)
            else:
                self._algcomm = None
                self._algsize = 0
                self._algrank = 0
                sr = np.array([self._algsize, self._algrank])
                sr = self._solcomm.bcast(sr, root=0)
                self._algsize, self._algrank = int(sr[0]), int(sr[1])

            self._ready = True

            # self._print_status()

        def _require_ready(self) -> None:
            if not self._ready:
                raise RuntimeError("odatse.mpi.setup() has not been called")

        # --- Global MPI (available before setup) ---

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
    "setup",
    "comm", "size", "rank",
    "solcomm", "solsize", "solrank",
    "algcomm", "algsize", "algrank",
    "run_on_algorithm",
    "enabled",
    "OtherAlgorithmProcessError",
    "MSG_ABORT", "MSG_FINISHED", "MSG_EVALUATE",
]

def setup(*, nalg=None, nsolve=None):   _ctx.setup(nalg=nalg, nsolve=nsolve)
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
