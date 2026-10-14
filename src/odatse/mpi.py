# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABCMeta as _ABCMeta
import contextlib
import os
import pickle
import numpy as np
from typing import Optional, Tuple

_NOMPI = os.environ.get("ODATSE_NOMPI", "0") != "0"


class SetupConflictError(RuntimeError):
    """setup() was called again with a different communicator or layout.

    A RuntimeError, so that callers catching that still work; a distinct
    class so that a conflict can be told apart from an MPI failure
    (mpi4py's MPI.Exception is a RuntimeError as well).
    """

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
    setup() validates nalg/nsolve like the MPI build but otherwise ignores
    its arguments (a RuntimeWarning is emitted when a layout other than 1x1
    or a communicator of more than one rank is requested), and ready() is
    always True because there is nothing to partition.
    """

    def setup(self, *, nalg: Optional[int] = None, nsolve: Optional[int] = None,
              comm=None) -> None:
        # the same argument validation as the MPI build, so that the CLI
        # rejects --nalg 0 regardless of the backend
        if nalg is not None and nalg <= 0:
            raise ValueError(f"nalg must be a positive integer, got {nalg}")
        if nsolve is not None and nsolve <= 0:
            raise ValueError(f"nsolve must be a positive integer, got {nsolve}")
        requested = []
        if comm is not None and getattr(comm, "size", None) != 1:
            # a one-rank communicator loses nothing; anything else does
            requested.append("comm")
        if nalg is not None and nalg != 1:
            requested.append(f"nalg={nalg}")
        if nsolve is not None and nsolve != 1:
            requested.append(f"nsolve={nsolve}")
        if requested:
            # Neither a communicator nor a layout can be honoured here: every
            # process runs as an independent serial instance writing the same
            # files, which is what the caller should learn about.
            import warnings
            warnings.warn(
                f"odatse.mpi.setup({', '.join(requested)}) is ignored because MPI "
                "is disabled (ODATSE_NOMPI is set or mpi4py is not available); "
                "every process runs as an independent serial instance",
                RuntimeWarning, stacklevel=3,
            )

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

        def _resolve_comm(self, comm):
            """Return the communicator to partition. None means the current
            global communicator: MPI.COMM_WORLD before setup(), and the
            communicator given to setup() afterwards (so a bare setup() after
            setup(comm=sub) refers to sub, not to MPI.COMM_WORLD)."""
            if comm is None:
                return self._comm
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
                Intracommunicator to partition. None means the current global
                communicator (MPI.COMM_WORLD unless an earlier setup() was
                given another one). It becomes the global communicator
                returned by comm(); the
                caller keeps ownership (it is never freed here) and must keep
                it alive while ODAT-SE is in use. setup() is collective over
                this communicator: every rank of it must call setup() with
                the same arguments.

            Exactly one of nalg/nsolve may be None; the missing value is
            derived from the total process count of the communicator. If both
            are None, all processes are assigned to the algorithm layer
            (nsolve=1).

            ready() must be the same on every rank of comm when setup() is
            called: the no-op return for an already configured context is
            local, while the partitioning below is collective, so a rank that
            skips it leaves the others blocked.

            setup() may be called again. With both nalg and nsolve omitted
            it keeps the current layout, whatever it is; with a layout given,
            or on the first call, the effective configuration (the
            communicator, and nalg/nsolve after the derivation above) must
            equal the current one for the call to be a no-op, otherwise
            SetupConflictError (a RuntimeError) is raised. A library that
            only needs "some" partition of its communicator can therefore
            call setup(comm=my_comm) unconditionally. Communicators are compared as MPI handles
            (mpi4py's ``==``), so two Python objects wrapping the same handle
            count as the same communicator, while a duplicate (``Dup()``)
            does not. All checks are local and happen before any collective,
            so raising here cannot leave other ranks blocked.
            """
            comm = self._resolve_comm(comm)
            # argument errors (ValueError) take precedence over a conflict
            # with an earlier call (RuntimeError), whatever the state
            if self._ready and nalg is None and nsolve is None:
                nalg, nsolve = self._nalg, self._nsolve   # nothing requested: keep
            else:
                nalg, nsolve = self._resolve_layout(nalg, nsolve, comm.size)

            if self._ready:
                if comm != self._comm:
                    raise SetupConflictError(
                        "setup() has already been called with a different communicator"
                    )
                if (nalg, nsolve) != (self._nalg, self._nsolve):
                    raise SetupConflictError(
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
            handles = [solcomm]       # freed if a later step fails
            try:
                solsize = solcomm.size
                assert solsize == nsolve
                solrank = solcomm.rank

                # Algorithm intracommunicator: one representative per solver group (solrank==0)
                world_group = comm.Get_group()
                handles.append(world_group)
                alg_group = world_group.Incl([c * nsolve for c in range(nalg)])
                handles.append(alg_group)
                algcomm = comm.Create(alg_group)
                if algcomm != MPI.COMM_NULL:
                    handles.append(algcomm)
                    algsize = algcomm.size
                    algrank = algcomm.rank
                    sr = np.array([algsize, algrank])
                    solcomm.bcast(sr, root=0)
                else:
                    algcomm = None
                    sr = np.array([0, 0])
                    sr = solcomm.bcast(sr, root=0)
                    algsize, algrank = int(sr[0]), int(sr[1])
            except BaseException:
                for h in handles:
                    try:
                        h.Free()
                    except Exception:
                        pass
                raise
            alg_group.Free()
            world_group.Free()

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
    "setup", "ready", "SetupConflictError",
    "comm", "size", "rank",
    "solcomm", "solsize", "solrank",
    "algcomm", "algsize", "algrank",
    "run_on_algorithm",
    "enabled",
    "fail_together", "FailTogetherMeta",
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


# ------------------------------------------------------------------ #
#  Failing together
# ------------------------------------------------------------------ #

def _agree_on_failure(error: Optional[BaseException]) -> None:
    """Collectively agree on whether *error* (this process's exception, or
    ``None``) occurred on any process of the job, and raise on every process
    when it did.

    Three small collectives share the outcome:

    1. the processes of each solver group agree on ``solcomm``;
    2. the controllers agree across the groups on ``algcomm``;
    3. each controller broadcasts the verdict to its workers on ``solcomm``.

    Then a process that failed re-raises its own exception and every other
    process raises ``OtherAlgorithmProcessError``. Before ``setup()`` has been
    called there is no layout to agree on, and *error* is re-raised unchanged.
    """
    ready = getattr(_ctx, "_ready", True)
    if ready:
        ok = np.array([0 if error is not None else 1])

        if solsize() > 1:
            total = np.array([0])
            solcomm().Allreduce(ok, total)
            group_ok = bool(total[0] == solsize())
        else:
            group_ok = error is None

        all_ok = group_ok
        if run_on_algorithm() and algsize() > 1:
            flag = np.array([1 if group_ok else 0])
            total = np.array([0])
            algcomm().Allreduce(flag, total)
            all_ok = bool(total[0] == algsize())

        if solsize() > 1:
            # the workers are not in algcomm: they learn the verdict from
            # their controller, which has just taken part in the agreement
            all_ok = solcomm().bcast(all_ok, root=0)
    else:
        all_ok = error is None

    if error is not None:
        from odatse import exception
        if ready and isinstance(error, exception.Error):
            # this rank's own failure: have the CLI boundary report it
            # from this rank (rank 0 may have no error to print)
            error.rank_local = True
        raise error
    if not all_ok:
        raise OtherAlgorithmProcessError()


@contextlib.contextmanager
def fail_together():
    """Context manager: if the block raises on any process, raise on all.

    Use it around set-up code that every process of the job runs and that
    may fail on some processes only (a file that is missing on one node, for
    example)::

        with odatse.mpi.fail_together():
            solver = Solver(info)
            runner = odatse.Runner(solver, info)

    When the block completes everywhere, nothing happens. Otherwise, after
    every process has left the block, a process whose block raised re-raises
    its own exception (an ``odatse.exception.Error`` is marked
    ``rank_local``, so that the ``odatse`` command reports it from that
    process) and every other process raises ``OtherAlgorithmProcessError``.
    Without this, the processes that succeeded would go on into the next
    collective and wait forever for the one that failed.

    The exit of the block is itself a collective once ``setup()`` has been
    called: every process of the job, the solver workers included, must
    enter the same ``fail_together()`` blocks in the same order, and
    collective calls inside a block must stay balanced (a process that
    raises before such a call leaves the others waiting in it). Before
    ``setup()``, it only lets the exception propagate.

    The constructors of ``Solver``, ``Runner`` and ``Algorithm`` carry this
    agreement on their own (``FailTogetherMeta``). Blocks nest: only the
    outermost block of a process agrees, an inner block (an explicit one, or
    the one of a construction) only lets the exception through, so that a
    process failing anywhere in the outermost block, before or after an
    inner one, agrees exactly once with the others.
    """
    global _block_depth
    if _block_depth > 0:
        # an enclosing block agrees when it is left
        _block_depth += 1
        try:
            yield
        finally:
            _block_depth -= 1
        return
    _block_depth += 1
    try:
        yield
    except BaseException as e:
        # also SystemExit / KeyboardInterrupt: the other processes must be
        # released before it propagates
        _block_depth -= 1
        _agree_on_failure(e)
        raise  # not reached: _agree_on_failure() re-raises e
    _block_depth -= 1
    _agree_on_failure(None)


# Nesting depth of fail_together() blocks on this process (process-global,
# not per thread: ODAT-SE does not construct these objects from threads).
# Only the outermost block agrees (inner blocks include the constructions of
# FailTogetherMeta classes). Otherwise a process that fails in the outer block
# before an inner block the others reach would perform one agreement while
# they perform two (the inner block's and the outer one's), and the second
# would wait forever.
_block_depth = 0


class FailTogetherMeta(_ABCMeta):
    """Metaclass whose instances are constructed inside ``fail_together()``.

    It is the metaclass of ``SolverBase``, ``Runner`` and ``AlgorithmBase``,
    so that ``Solver(info)``, ``Runner(solver, info)`` and
    ``Algorithm(info, runner)`` are protected whatever code calls them: the
    ``odatse`` command, a user-written main, or a host program that embeds
    ODAT-SE. These constructors run on every process of the job, the solver
    workers included, and may fail on some of them only (a data file missing
    on one node, an invalid ``mesh_path`` detected by the algorithm ranks
    that read the mesh); a process that leaves the set-up through such an
    exception would otherwise leave the others waiting in the next collective
    (issues #101, #112). ``fail_together()`` remains available for the
    caller's own code between the constructions.

    Wrapping the construction here covers the whole constructor of every
    subclass, including the part that runs after ``super().__init__()``, and
    third-party subclasses without any change on their side (a subclass that
    declares a metaclass of its own must derive it from this one).
    ``Cls.__new__(Cls)`` does not go through ``__call__`` and is therefore
    not affected. The rules of ``fail_together()`` apply: every process must
    construct the same objects in the same order (constructing one on some
    processes only hangs), and collectives inside a constructor must stay
    balanced.

    Inside a ``fail_together()`` block the constructors do not agree on
    their own; the outermost block does, when it is left. The block is then
    the only agreement, however many objects it constructs and wherever in
    it a process fails (``odatse.main()`` and the parallel-solver example
    build the solver and the runner inside such a block).
    """

    def __call__(cls, *args, **kwargs):
        # inside an enclosing block this agrees nothing (see fail_together())
        with fail_together():
            return super().__call__(*args, **kwargs)


# ------------------------------------------------------------------ #
#  Reading on one process for the others
# ------------------------------------------------------------------ #

def agree(comm, problem=None, payload=None):
    """Share this rank's outcome with every rank of ``comm`` in one
    ``allgather``, so that a failure on one rank is seen by all of them.

    Parameters
    ----------
    comm
        The communicator (``None`` or size 1: no communication).
    problem : str, optional
        What went wrong on this rank, or ``None`` when it succeeded.
    payload
        Anything picklable to share alongside (the steps of the checkpoints
        this rank has, for instance).

    Returns
    -------
    list
        One ``(global rank, problem, payload)`` tuple per rank of ``comm``,
        in rank order.

    Notes
    -----
    Collective over ``comm``: every rank must call it, whether or not it
    failed. That is the point: the ranks that succeeded learn of a failure
    here instead of waiting for the failed rank in a later collective. The
    caller raises the same error on every rank when any ``problem`` is set
    (``exception.Error.report_rank`` names the rank that prints it).
    """
    report = (rank(), problem, payload)
    if comm is None or comm.size == 1:
        return [report]
    return comm.allgather(report)


def _describe(error: BaseException) -> str:
    """``"Type: message"``; never raises (a broken __str__ must not make the
    reading process skip the collective below)."""
    try:
        msg = str(error)
    except BaseException:
        msg = "<unprintable exception>"
    return f"{type(error).__name__}: {msg}"


# Bytes per Bcast when the pickled data is sent: below the 2**31 element
# count of MPI-3, so that data of any size can be sent on any MPI stack.
_CHUNK_BYTES = 1 << 30


def _alloc_buffer(nbytes: int) -> np.ndarray:
    """Receive buffer for the pickled data (a function, so that tests can
    make the allocation fail)."""
    return np.empty(nbytes, dtype=np.uint8)


def _distribute(loader, comm, *, root: int = 0, what: str = "data", distribute: bool = True):
    """Call ``loader()`` on rank ``root`` of ``comm`` and hand its result to
    every rank of ``comm``; if the read fails anywhere, every rank raises.

    The mechanism behind ``odatse.util.io.load()`` (the user-facing entry
    point, which also chooses the communicator from a scope name). Collective
    over ``comm``: every rank of it must call this in the same order.

    1. The root reads and pickles the data inside a ``try`` and broadcasts a
       small header, either ``("ok", number of bytes, what, root's global
       rank)`` or ``("error", message, root's global rank)``. An object that
       cannot be pickled is therefore reported as a failure, and a failing
       root (also on ``SystemExit`` / ``KeyboardInterrupt``) still enters
       the collective. The message is built on the root, so every rank
       reports the same ``what``; the root's global rank becomes
       ``report_rank`` of every ``LoadError`` raised below, so that the
       ``odatse`` command prints it once.
    2. The receivers allocate a buffer for the bytes (the root takes a view
       of its pickle), and all ranks agree on the outcome (one
       ``allgather``): a rank that cannot hold the data makes every rank
       raise instead of leaving the others in the transfer.
    3. The bytes are sent with ``Bcast`` in chunks of ``_CHUNK_BYTES``, from
       the root's pickle without another copy, so the size is not limited by
       the element count of an MPI call.
    4. The receivers unpickle the data, and all ranks agree on the outcome
       (one ``allgather``) before any of them returns: an object that pickles
       on the root but cannot be rebuilt on a receiver (a class that cannot be
       imported there, a ``__reduce__`` callable that raises) is a
       ``LoadError`` on every rank, not a raw exception on one.

    Every rank then raises the same ``odatse.exception.LoadError`` (the rank
    that failed with the original exception as ``__cause__``), or returns
    the data. With ``distribute=False`` only the outcome of the read is
    shared: the data is returned on the root and ``None`` on the other ranks
    (for data that the root will scatter itself), and steps 2 to 4 are
    skipped.

    A ``SystemExit`` or ``KeyboardInterrupt`` on the root becomes a
    ``LoadError`` (an ``Exception``) on every rank, the root included: the
    phase wrappers of ``AlgorithmBase`` agree on failures of type
    ``Exception`` only, so a root re-raising the original exception there
    would skip that agreement and leave the others waiting in it.

    ``comm`` of ``None`` (the non-MPI stub) or of size 1: a plain
    ``loader()`` call, wrapped in ``LoadError`` on failure for uniform
    reporting. ``root`` is validated against the size in every case, so a
    wrong value shows up in serial testing too.
    """
    from odatse import exception

    size = 1 if comm is None else comm.size
    if not 0 <= root < size:
        raise ValueError(f"root must be a rank of the communicator (0..{size - 1}), got {root}")

    if size == 1:
        try:
            return loader()
        except Exception as e:
            err = exception.LoadError(f"cannot read {what}: {_describe(e)}")
            err.report_rank = rank()
            raise err from e

    original = None
    root_rank = None   # global rank of the root, learned from the header

    def fail(message):
        # the same LoadError on every rank; the root of the read is the one
        # that reports it (odatse.exception.Error.report_rank), so that the
        # odatse command prints the line once and not once per process
        err = exception.LoadError(message)
        err.report_rank = root_rank
        if original is not None:
            raise err from original
        raise err

    def check(problem, message):
        # every rank learns whether the step failed anywhere; raise everywhere if so
        problems = [p for _, p, _ in agree(comm, problem) if p is not None]
        if problems:
            fail(f"{message}: " + "; ".join(problems))

    # 1. read and pickle on the root; share the outcome (and the root's
    # global rank, for the report)
    data = None
    blob = None
    header = None
    if comm.rank == root:
        try:
            data = loader()
            if distribute:
                blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            header = ("ok", 0 if blob is None else len(blob), what, rank())
        except BaseException as e:
            # also SystemExit / KeyboardInterrupt: the collective must be entered
            original = e
            header = ("error", f"cannot read {what} (failed on rank {rank()}): {_describe(e)}", rank())
    header = comm.bcast(header, root=root)
    root_rank = header[-1]

    if header[0] == "error":
        fail(header[1])

    if not distribute:
        return data   # None on the receivers

    _, nbytes, root_what, _ = header

    # 2. a buffer on every rank (a view of the pickle on the root), or nobody
    # enters the transfer
    problem = None
    try:
        if comm.rank == root:
            buf = np.frombuffer(blob, dtype=np.uint8)
        else:
            buf = _alloc_buffer(nbytes)
    except BaseException as e:
        original = e
        problem = f"rank {rank()}: {_describe(e)}"
    check(problem, f"cannot transfer {root_what} ({nbytes} bytes)")

    # 3. the bytes, in chunks
    for start in range(0, nbytes, _CHUNK_BYTES):
        comm.Bcast(buf[start:start + _CHUNK_BYTES], root=root)

    # 4. unpack on the receivers; agree on the outcome before anyone returns
    problem = None
    if comm.rank != root:
        try:
            data = pickle.loads(buf)
        except BaseException as e:
            original = e
            problem = f"rank {rank()}: {_describe(e)}"
        del buf
    check(problem, f"cannot unpack {root_what} on the receiving rank(s)")
    return data
