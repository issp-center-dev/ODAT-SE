# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
from abc import abstractmethod
from enum import IntEnum
import time
import os
import pathlib
import pickle
import copy
import traceback

import numpy as np

import odatse
import odatse.util.limitation
from odatse import exception, mpi

# for type hints
from pathlib import Path
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from mpi4py import MPI

class AlgorithmStatus(IntEnum):
    """Enumeration for the status of the algorithm."""
    INIT = 1
    PREPARE = 2
    RUN = 3

class AlgorithmBase(metaclass=odatse.mpi.FailTogetherMeta):
    """Base class for algorithms, providing common functionality and structure.

    Construction takes part in the agreement of ``odatse.mpi.fail_together()``
    (``FailTogetherMeta``): ``Algorithm(info, runner)`` is called on every
    process of the job, the solver workers included, and may fail on some of
    them only (an invalid ``mesh_path``, for example, is detected on the
    algorithm ranks, which read the mesh, and not on the workers). The phase
    wrappers below agree on failures through ``_reach_consensus()`` and
    ``main()`` releases the workers when a phase fails, but neither is reached
    when the constructor raises; the metaclass closes that gap (issue #101).
    Collectives inside a subclass constructor must stay balanced across the
    processes: a process that raises before such a collective leaves the
    others waiting in it. The one collective of ``AlgorithmBase.__init__`` is
    failure-safe in this sense.

    Lifecycle
    ---------
    ``main()`` drives the three-phase lifecycle by calling the framework wrappers::

        main()
          ├── prepare()   runner.prepare → dispatch(init/resume/continue) → _prepare()
          ├── run()       _run()
          └── post()      _post() → runner.post()

    Subclasses implement the hooks with underscore prefix:
    ``_prepare()`` (optional), ``_run()`` (required), ``_post()`` (required).
    The plain-named wrappers ``prepare``, ``run``, ``post`` are
    framework internals and must **not** be overridden in subclasses.

    Checkpoint
    ----------
    Each class declares ``_checkpoint_attrs`` (a list of attribute names).
    ``__getstate__()`` walks the MRO and collects them all automatically.
    Subclasses normally need only declare their own ``_checkpoint_attrs``
    and override ``_apply_state()`` to call ``super()`` and restore RNG /
    algorithm-specific state.  Override ``_save_state()`` / ``_load_state()``
    only when extra files (e.g. an external policy object) must be written.
    """

    rng: np.random.RandomState
    dimension: int
    label_list: list[str]
    runner: Optional[odatse.Runner]

    root_dir: Path
    output_dir: Path
    proc_dir: Path

    timer: dict[str, dict]

    status: AlgorithmStatus = AlgorithmStatus.INIT
    mode: Optional[str] = None

    # Fields saved/restored at every checkpoint for this class.
    # Each subclass declares only its own fields; ``__getstate__`` collects
    # them all by walking the MRO.
    _checkpoint_attrs: list[str] = []

    # Name of the progress counter of the run (the Monte Carlo step, the
    # number of evaluations) when the algorithm ranks advance in lockstep,
    # i.e. when they take part in collectives during the run.
    # ``_load_data()`` then makes every rank resume from a checkpoint of the
    # same progress (see there); ranks that restarted from different steps
    # would run different numbers of iterations and block in the first
    # collective.  By default it is the key of the snapshot field that
    # ``_checkpoint_step()`` reads; an algorithm whose counter is not a single
    # field overrides that method and sets the name used in messages here.
    # ``None`` (the default) for algorithms whose ranks do not communicate
    # during the run (mapper, random_search, bayes): a different position per
    # rank is then legitimate.
    _checkpoint_step_key: Optional[str] = None

    def _checkpoint_step(self, data: dict):
        """The progress recorded in the snapshot *data*, a value that is the
        same on every algorithm rank at a given checkpoint and increases from
        one checkpoint to the next (comparable, picklable). The default reads
        the field named by ``_checkpoint_step_key``; pamc overrides it
        (``istep`` is not in lockstep there when a rank runs out of walkers).
        """
        return data[self._checkpoint_step_key]

    # Number of previous checkpoints kept next to the current one
    # (``status.pickle.1`` ... ``status.pickle.<n>``).
    _checkpoint_generations: int = 3

    def __getstate__(self) -> dict:
        """Return a checkpoint snapshot of the full algorithm state.

        Saves the MPI configuration, RNG state, timer, and ``info``, then
        appends every field declared in ``_checkpoint_attrs`` by this class
        and all its subclasses (collected via MRO traversal).  Subclasses
        need only declare their own ``_checkpoint_attrs``; no override is
        needed unless extra non-attribute data must be saved (e.g. a global
        RNG or an external policy object).
        """
        state: dict = {
            "algsize": odatse.mpi.algsize(),
            "algrank": odatse.mpi.algrank(),
            "rng": self.rng.get_state(),
            "timer": self.timer,
            "info": self.info,
        }
        for cls in reversed(type(self).__mro__):
            for attr in cls.__dict__.get("_checkpoint_attrs", []):
                state[attr] = getattr(self, attr)
        return state

    def _apply_state(self, data: dict, mode: str = "resume", restore_rng: bool = True) -> None:
        """Restore the base algorithm state from a checkpoint snapshot.

        Validates the MPI configuration, restores the timer, checks that
        algorithm parameters are consistent, and restores the RNG state.
        Subclasses should call
        ``super()._apply_state(data, mode=mode, restore_rng=restore_rng)``
        and then handle their own subclass-specific fields
        (``_checkpoint_attrs``, continue-mode semantics, etc.).  The RNG state
        saved by ``__getstate__`` for every algorithm is restored here (guarded
        by ``restore_rng``), so subclasses need not repeat it.

        Parameters
        ----------
        data : dict
            Snapshot previously produced by ``__getstate__``.
        mode : str
            ``"resume"`` or ``"continue"``.  Passed through to subclass
            overrides so they can implement continue-mode semantics.
        restore_rng : bool
            When *True* (default) the RNG state is restored from *data*.
        """
        assert odatse.mpi.algsize() == data["algsize"]
        assert odatse.mpi.algrank() == data["algrank"]
        self.timer = data["timer"]
        self._check_parameters(data["info"])
        if restore_rng:
            # Restore in place rather than rebinding self.rng: collaborators
            # constructed in __init__ hold a reference to this object (e.g.
            # the Monte Carlo StateSpace draws its proposals from it, and
            # MeshGrid may capture it), and __init__ runs before the resume
            # dispatch in prepare(). Rebinding would leave those collaborators
            # on the stale un-restored RNG, silently splitting the random
            # stream after a resume.
            self.rng.set_state(data["rng"])

    @abstractmethod
    def __init__(
            self,
            info: odatse.Info,
            runner: Optional[odatse.Runner] = None,
            run_mode: str = "initial",
    ) -> None:
        """
        Initialize the algorithm with the given information and runner.

        Parameters
        ----------
        info : Info
            Information object containing algorithm and base parameters.
        runner : Runner (optional)
            Optional runner object to execute the algorithm.
        run_mode : str
            Mode in which the algorithm should run.
        """
        # Start of the "init" phase: prefer the timestamp recorded by
        # odatse.initialize() so that argument parsing, input loading and
        # solver construction are included; fall back to the construction
        # time of this object when the caller did not use initialize().
        # The phase is closed at the beginning of _main_algorithm().
        start_time = getattr(info, "_start_time", None)
        self._init_time_sta = start_time if start_time is not None else time.perf_counter()
        self.timer = {"init": {}, "prepare": {}, "run": {}, "post": {}}
        self.timer["init"]["total"] = 0.0
        self.status = AlgorithmStatus.INIT
        self.mode = run_mode.lower()

        # keep copy of parameters
        self.info = copy.deepcopy(info.algorithm)

        self.dimension = info.algorithm.get("dimension") or info.base.get("dimension")
        if not self.dimension:
            raise ValueError("ERROR: dimension is not defined")

        if "label_list" in info.algorithm:
            label = info.algorithm["label_list"]
            if len(label) != self.dimension:
                raise ValueError(f"ERROR: length of label_list and dimension do not match ({len(label)} != {self.dimension})")
            self.label_list = label
        else:
            self.label_list = [f"x{d+1}" for d in range(self.dimension)]

        # initialize random number generator
        self.__init_rng(info)

        # directories
        self.root_dir = info.base["root_dir"]
        self.output_dir = info.base["output_dir"]
        self.proc_dir = self.output_dir / str(odatse.mpi.algrank())
        mkdir_error: Optional[Exception] = None
        try:
            # create directory for each rank in case every rank has some output
            self.proc_dir.mkdir(parents=True, exist_ok=True)
            # Some cache of the filesystem may delay making a dictionary
            # especially when mkdir just after removing the old one
            while not self.proc_dir.is_dir():
                time.sleep(0.1)
        except Exception as e:
            mkdir_error = e

        # Synchronise the algorithm ranks. An Allreduce of the success flags
        # rather than a Barrier: a rank that failed to create its directory
        # (e.g. a PermissionError on one node only) must not leave the others
        # here while it goes on to the construction consensus. Every rank
        # raises instead, and all of them reach that consensus.
        if odatse.mpi.algcomm() is not None:
            self._reach_consensus(
                mkdir_error, np.array([0 if mkdir_error is not None else 1])
            )
        elif mkdir_error is not None:
            raise mkdir_error

        # checkpointing
        self.checkpoint = info.algorithm.get("checkpoint", False)
        _chk_file = info.algorithm.get("checkpoint_file", "status.pickle")
        # Resolve to an absolute path so _load_state() / _save_state() work
        # correctly regardless of the working directory at call time.
        # _prepare() runs from the original directory, not proc_dir, so a
        # relative path would resolve to the wrong location.
        self.checkpoint_file = str(
            Path(_chk_file) if Path(_chk_file).is_absolute()
            else self.proc_dir / _chk_file
        )
        self.checkpoint_steps = info.algorithm.get("checkpoint_steps", 65536*256)  # large number
        self.checkpoint_interval = info.algorithm.get("checkpoint_interval", 86400*360)  # longer enough

        # runner
        if runner is not None:
            self.set_runner(runner)

    def __init_rng(self, info: odatse.Info) -> None:
        """
        Initialize the random number generator.

        Parameters
        ----------
        info : Info
            Information object containing algorithm parameters.
        """
        seed = info.algorithm.get("seed", None)
        seed_delta = info.algorithm.get("seed_delta", 314159)

        if seed is None:
            self.rng = np.random.RandomState()
        else:
            # Offset the seed by the algorithm-layer rank, not the global MPI
            # rank.  When the solver runs in parallel (nsolve > 1) the global
            # rank differs from the algorithm rank, so seeding by rank() would
            # make the per-replica seeds depend on the solver parallelism and
            # break reproducibility.  algrank() identifies the replica.
            self.rng = np.random.RandomState(seed + odatse.mpi.algrank() * seed_delta)

    def set_runner(self, runner: odatse.Runner) -> None:
        """
        Set the runner for the algorithm.

        Parameters
        ----------
        runner : Runner
            Runner object to execute the algorithm.
        """
        self.runner = runner

    # ------------------------------------------------------------------
    # Framework wrappers – called by main().  Do NOT override in subclasses.
    # ------------------------------------------------------------------

    def _reach_consensus(self, error: Optional[BaseException], ok: "np.ndarray") -> None:
        """Collectively agree on whether *every* algorithm rank succeeded.

        Every algorithm rank must call this exactly once per phase, regardless
        of whether its phase body succeeded or raised. ``ok`` is ``[1]`` when
        this rank's phase succeeded and ``[0]`` otherwise; ``error`` is the
        exception this rank caught (or ``None``). ``SystemExit`` and
        ``KeyboardInterrupt`` take part like any other failure, so that the
        other ranks are released, and are then re-raised as they are.

        A single ``Allreduce`` shares the success flags, then:

        * if this rank failed, its own exception is re-raised;
        * else if any other rank failed, ``OtherAlgorithmProcessError`` is
          raised so this rank bails out too.

        Because the only collective on the failure path is this one
        ``Allreduce`` -- reached by all ranks whether they succeeded or failed
        -- a per-rank failure can no longer leave the other ranks blocked.
        (Collectives *inside* the ``_prepare``/``_run``/``_post`` hooks remain
        the responsibility of each algorithm to keep balanced across ranks.)
        """
        if odatse.mpi.algsize() > 1:
            total = np.array([0])
            odatse.mpi.algcomm().Allreduce(ok, total)
            all_ok = bool(total[0] == odatse.mpi.algsize())
        else:
            all_ok = error is None

        if error is not None:
            if isinstance(error, exception.Error):
                # this rank's own failure: have the CLI boundary report it
                # from this rank (rank 0 may have no error to print)
                error.rank_local = True
            raise error
        if not all_ok:
            raise odatse.mpi.OtherAlgorithmProcessError()

    def prepare(self) -> None:
        """Framework wrapper for the prepare phase.

        Calls ``runner.prepare()``, dispatches init/resume/continue, then
        calls the ``_prepare()`` hook.

        Do **not** override this method in subclasses.  Implement
        ``_prepare()`` instead.
        """
        if not odatse.mpi.run_on_algorithm():
            return

        if self.runner is None:
            raise RuntimeError("Runner is not assigned")

        restore_rng = not self.mode.endswith("-resetrand")

        # Everything that can fail per-rank (runner setup, checkpoint dispatch,
        # the _prepare() hook) is inside the try so that a failure on any rank
        # is shared through the single collective in _reach_consensus() instead
        # of leaving the other ranks blocked. The exception is captured here and
        # re-raised only after that collective.
        error = None
        ok = np.array([0])
        try:
            # runner lifecycle starts (commit 936e48: moved here from run())
            try:
                self.runner.prepare(self.proc_dir)
            except Exception as e:
                # The resume dispatch of a lockstep algorithm is itself a
                # collective (the agreement on the checkpoint step in
                # _load_common_data) that every rank must enter: a rank that
                # failed here takes part with its error and raises it after
                # the agreement, instead of skipping to _reach_consensus()
                # and leaving the others waiting in the agreement.
                if not self._resume_is_collective():
                    raise
                self._failed_before_load = e

            # checkpoint dispatch
            if self.mode.startswith("init"):
                self._initialize()
            elif self.mode.startswith("resume"):
                self._load_state(self.checkpoint_file, mode="resume", restore_rng=restore_rng)
            elif self.mode.startswith("continue"):
                self._load_state(self.checkpoint_file, mode="continue", restore_rng=restore_rng)
            else:
                raise RuntimeError(f"unknown mode {self.mode}")

            # algorithm-specific preparation
            self._prepare()
            ok = np.array([1])
        except BaseException as e:
            # also SystemExit / KeyboardInterrupt: they must reach the
            # consensus below, or the other algorithm ranks wait in it forever
            error = e

        self._reach_consensus(error, ok)

        # preparation step completed
        self.status = AlgorithmStatus.PREPARE

    def run(self) -> None:
        """Framework wrapper for the run phase.

        Calls the ``_run()`` hook.  Runner calls are handled by
        ``prepare()`` and ``post()``; this wrapper contains no runner
        invocations.

        Do **not** override this method in subclasses.  Implement
        ``_run()`` instead.
        """
        if not odatse.mpi.run_on_algorithm():
            return

        if self.runner is None:
            raise RuntimeError("Runner is not assigned")

        if self.status < AlgorithmStatus.PREPARE:
            raise RuntimeError("algorithm has not prepared yet")

        error = None
        ok = np.array([0])
        original_dir = os.getcwd()
        try:
            os.chdir(self.proc_dir)
            self._run()
            ok = np.array([1])
        except BaseException as e:
            # also SystemExit / KeyboardInterrupt (see prepare())
            error = e
        finally:
            os.chdir(original_dir)

        self._reach_consensus(error, ok)

        # run step completed
        self.status = AlgorithmStatus.RUN

    def post(self) -> dict:
        """Framework wrapper for the post phase.

        Calls the ``_post()`` hook then ``runner.post()``.

        Do **not** override this method in subclasses.  Implement
        ``_post()`` instead.
        """
        if not odatse.mpi.run_on_algorithm():
            return {}

        if self.status < AlgorithmStatus.RUN:
            raise RuntimeError("algorithm has not run yet")

        error = None
        ok = np.array([0])
        result = {}
        original_dir = os.getcwd()
        try:
            os.chdir(self.output_dir)
            result = self._post()
            ok = np.array([1])
        except BaseException as e:
            # also SystemExit / KeyboardInterrupt (see prepare())
            error = e
        finally:
            os.chdir(original_dir)

        self._reach_consensus(error, ok)

        # runner lifecycle ends (commit 936e48: moved here from run())
        self.runner.post()

        return result

    # ------------------------------------------------------------------
    # Hooks – override these in subclasses.
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize(self) -> None:
        """Set up initial algorithm state for a fresh run (init mode).

        Called by ``prepare()`` when ``mode`` starts with ``"init"``.
        Must not use the runner (evaluation happens later in ``_run()``).
        """
        pass

    @abstractmethod
    def _prepare(self) -> None:
        """Algorithm-specific preparation, called after dispatch.

        Override in subclasses to perform setup that must happen after the
        checkpoint state is established (e.g. initialising timer entries).
        """
        pass

    @abstractmethod
    def _run(self) -> None:
        """Execute the main algorithm loop.

        For ``init`` mode, perform the initial evaluation here before
        entering the main loop.  Call ``_save_state()`` at the appropriate
        points inside the loop.
        """
        pass

    @abstractmethod
    def _post(self) -> dict:
        """Perform post-processing and return results."""
        pass

    # ------------------------------------------------------------------
    # Checkpoint helpers – concrete implementations in the base class.
    # ------------------------------------------------------------------

    def _start_checkpoint_schedule(self, step: int) -> None:
        """Arm the next checkpoint: due after ``checkpoint_steps`` more steps
        or ``checkpoint_interval`` more seconds, counted from *step* and now.

        Call once before the main loop (``step`` is the restored position on
        resume); ``_checkpoint_due()`` re-arms it after every checkpoint.
        """
        self._next_checkpoint_step = step + self.checkpoint_steps
        self._next_checkpoint_time = time.time() + self.checkpoint_interval

    def _checkpoint_due(self, step: int) -> bool:
        """Whether a checkpoint is due at *step*; re-arms the schedule if so.

        Always ``False`` when checkpointing is off. For a lockstep algorithm
        (``_checkpoint_step_key`` set) in a parallel run the decision is taken
        on algorithm rank 0 and broadcast, so that every rank writes its
        checkpoint at the same step: the time condition evaluated against
        each rank's own clock would otherwise fire at different steps on
        different ranks, and the ranks could not be resumed together (see
        ``_load_data()``).

        Collective over the algorithm ranks in that case, so call it only at
        a point where the ranks synchronise anyway (just before the next
        replica exchange, at the end of a temperature, after a sweep), and
        with the same *step* on every rank. Called at every step it would
        turn a failure on one rank (a solver error) into a hang of the
        others, where they would otherwise finish the loop and meet the
        failed rank in the phase consensus.
        """
        if not self.checkpoint:
            return False
        due = bool(step >= self._next_checkpoint_step
                   or time.time() >= self._next_checkpoint_time)
        if self._checkpoint_step_key is not None and odatse.mpi.algsize() > 1:
            # rank 0's decision for everyone
            due = odatse.mpi.algcomm().bcast(due, root=0)
        if due:
            self._start_checkpoint_schedule(step)
        return due

    def _snapshot(self) -> dict:
        """The snapshot to write at a checkpoint: ``__getstate__()``.

        Override to do work that must precede the snapshot (flushing output
        files whose positions the snapshot records), calling
        ``super()._snapshot()`` last. A failure here is shared with the
        other ranks by ``_save_state()`` like a failed write.
        """
        return self.__getstate__()

    def _save_state(self, filename, *, agree=True) -> None:
        """Save a checkpoint snapshot to *filename*.

        Takes the snapshot (``_snapshot()``, which collects all fields
        declared in ``_checkpoint_attrs`` across the MRO), then delegates to
        ``_save_data()`` for versioned pickle storage. A failure to take the
        snapshot is handed to ``_save_data()`` as a problem, so that the
        rank still takes part in the agreement of lockstep ranks there.

        ``agree=False`` writes without that agreement: for the final
        checkpoint after the main loop, where the ranks are no longer in
        lockstep (a rank that failed in the loop has left for the phase
        consensus, and the others must not wait for it in the agreement).

        Override in subclasses **only** when extra files must be written
        alongside the pickle (e.g. an external policy object).  In that
        case call ``super()._save_state(filename)`` first.
        """
        data = None
        problem = None
        cause = None
        try:
            data = self._snapshot()
        except Exception as e:
            cause = e
            problem = f"failed to take the checkpoint snapshot for {filename}: {e}"
        self._save_data(data, filename, problem=problem, cause=cause, agree=agree)

    def _load_state(self, filename, mode="resume", restore_rng=True) -> None:
        """Load a checkpoint snapshot from *filename* and apply it.

        Delegates to ``_load_data()`` then ``_apply_state()``.

        Override in subclasses **only** when extra files must be read
        (e.g. an external policy object).  In that case call
        ``super()._load_state(filename, mode=mode, restore_rng=restore_rng)``
        first.

        Parameters
        ----------
        filename : str
            Path to the checkpoint file.
        mode : str
            ``"resume"`` or ``"continue"``, forwarded to ``_apply_state()``.
        restore_rng : bool
            Whether to restore the RNG state.
        """
        data = self._load_data(filename)
        self._apply_state(data, mode=mode, restore_rng=restore_rng)

    # ------------------------------------------------------------------
    # main() – orchestrates the three phases with timing and MPI barriers.
    # ------------------------------------------------------------------

    def _main_algorithm(self):
        # Close the "init" phase opened in __init__. On resume/continue this
        # value is later overwritten by the timer restored from the checkpoint
        # (in _apply_state), preserving the original run's value.
        self.timer["init"]["total"] = time.perf_counter() - self._init_time_sta

        time_sta = time.perf_counter()
        self.prepare()
        time_end = time.perf_counter()
        self.timer["prepare"]["total"] = time_end - time_sta
        if odatse.mpi.algsize() > 1:
            odatse.mpi.algcomm().Barrier()

        time_sta = time.perf_counter()
        self.run()
        time_end = time.perf_counter()
        self.timer["run"]["total"] = time_end - time_sta
        print("end of run")
        if odatse.mpi.algsize() > 1:
            odatse.mpi.algcomm().Barrier()

        time_sta = time.perf_counter()
        result = self.post()
        time_end = time.perf_counter()
        self.timer["post"]["total"] = time_end - time_sta

        if odatse.mpi.algrank() == 0:
            self.write_timer(self.proc_dir / "time.log")

        return result

    def __signal_workers(self, signal: int) -> None:
        if odatse.mpi.solsize() > 1:
            msg = np.array([signal])
            odatse.mpi.solcomm().Bcast(msg, root=0)

    def main(self):
        """
        Main method to execute the algorithm.
        """
        if odatse.mpi.run_on_algorithm():
            try:
                res = self._main_algorithm()

            except odatse.mpi.OtherAlgorithmProcessError:
                self.__signal_workers(odatse.mpi.MSG_ABORT)
                sys.exit(0)

            except BaseException:
                # Also SystemExit / KeyboardInterrupt (e.g. raised by
                # solver.evaluate() on this controller and re-raised after the
                # solver-group status exchange): the workers are back in the
                # control-signal Bcast and must be released.
                self.__signal_workers(odatse.mpi.MSG_ABORT)
                raise

            self.__signal_workers(odatse.mpi.MSG_FINISHED)
            return res
        else: # Worker process for solver
            assert odatse.mpi.solrank() > 0
            signal = np.array([0])
            xp = np.zeros(self.runner.solver.dimension)
            aborted = False
            try:
                while True:
                    odatse.mpi.solcomm().Bcast(signal, root=0)
                    if signal[0] == odatse.mpi.MSG_FINISHED:
                        break
                    elif signal[0] == odatse.mpi.MSG_ABORT:
                        aborted = True
                        break
                    elif signal[0] == odatse.mpi.MSG_EVALUATE:
                        odatse.mpi.solcomm().Bcast(xp, root=0)
                        args = odatse.mpi.solcomm().bcast(None, root=0)
                        # An exception raised by solver.evaluate() is caught
                        # inside serve() and reported to the controller through
                        # the solver-group status exchange (Runner._evaluate_group),
                        # where it becomes an ordinary evaluate failure subject
                        # to ignore_error. Nothing propagates out of serve().
                        self.runner.serve(xp, args)
                    else:
                        raise ValueError(f"Unknown signal: {signal[0]}")
            except BaseException as e:
                # Last resort for failures outside solver.evaluate() (a corrupt
                # control message, an unpicklable args, Ctrl-C while waiting
                # for the control signal, ...); a failure inside evaluate() is
                # reported through the status exchange in Runner.serve() and
                # never gets here. A solver worker is outside the
                # algorithm-layer consensus in _reach_consensus() (it is not a
                # member of algcomm) and the controller is, or will be, blocked
                # in a solcomm collective, so letting the exception propagate
                # (or the process exit) would hang the whole job. Report and
                # abort instead.
                if isinstance(e, KeyboardInterrupt):
                    print(f"[rank {odatse.mpi.rank()}] solver worker interrupted, aborting the job",
                          file=sys.stderr, flush=True)
                else:
                    traceback.print_exc()
                    print(f"[rank {odatse.mpi.rank()}] ERROR: solver worker failed: "
                          f"{exception.describe_error(e)}",
                          file=sys.stderr, flush=True)
                odatse.mpi.comm().Abort(1)
            if aborted:
                sys.exit(0)
            return None

    def write_timer(self, filename: Path):
        """
        Write the timing information to a file.

        Parameters
        ----------
        filename : Path
            Path to the file where timing information will be written.
        """
        with open(filename, "w") as fw:
            fw.write("#in units of seconds\n")

            def output_file(type):
                d = self.timer[type]
                fw.write("#{}\n total = {}\n".format(type, d["total"]))
                for key, t in d.items():
                    if key == "total":
                        continue
                    fw.write(" - {} = {}\n".format(key, t))

            output_file("init")
            output_file("prepare")
            output_file("run")
            output_file("post")

    def _save_data(self, data, filename="state.pickle", ngen=None, *,
                   problem=None, cause=None, agree=True) -> None:
        """
        Save data to a file with versioning.

        Parameters
        ----------
        data
            Data to be saved (ignored when *problem* is given).
        filename
            Name of the file to save the data.
        ngen : int, optional
            Number of generations for versioning; ``_checkpoint_generations``
            (3) by default.
        problem : str, optional
            A failure of this rank before the write (taking the snapshot):
            nothing is written, and the failure is shared with the other
            lockstep ranks, or raised, like a failed write.
        cause : Exception, optional
            The exception behind *problem*, chained to the error raised.
        agree : bool
            Whether the lockstep ranks agree on the outcome before and
            after publishing (see ``_save_state``).
        """
        if ngen is None:
            ngen = self._checkpoint_generations
        tmp = Path(filename + ".tmp")
        lockstep = agree and self._checkpoint_step_key is not None and odatse.mpi.algsize() > 1

        def fail(problems, original):
            # the same error on every rank, printed once (from the global rank
            # of algorithm rank 0; see exception.Error.report_rank)
            err = exception.CheckpointError("; ".join(problems))
            if lockstep:
                err.report_rank = reports[0][0]
            if original is not None:
                raise err from original
            raise err

        # 1. the new checkpoint, in a temporary file
        if problem is None:
            try:
                with open(tmp, "wb") as f:
                    pickle.dump(data, f)
                    # Make sure the new checkpoint is durable on disk before it
                    # replaces the current one below.
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                cause = e
                problem = f"failed to save checkpoint to {filename}: {e}"

        # 2. Two-phase commit for lockstep ranks: the new checkpoints are
        #    published (renamed into place below) only once every rank has
        #    finished writing its own, so a stop during the write of one rank
        #    (long for a large state) cannot leave the ranks with checkpoints
        #    from different steps; the window shrinks to the renames. The
        #    agreement is the barrier, and it also shares a failed write: it
        #    raises on every rank, so no rank goes on without the others.
        if lockstep:
            reports = odatse.mpi.agree(odatse.mpi.algcomm(), problem)
            problems = [p for _, p, _ in reports if p is not None]
        else:
            problems = [problem] if problem is not None else []
        if problems:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as e:
                # keep the agreed error, not this one
                print(f"WARNING: could not remove {tmp}: {e}")
            fail(problems, cause)

        # 3. publish: rotate the older backup generations
        #    .(ngen-1) -> .ngen, ..., .1 -> .2 (os.replace is an atomic O(1)
        #    rename on every platform; shutil.move would fall back to
        #    copy+delete on Windows when the target exists), move the current
        #    checkpoint aside to .1, then atomically swap in the new one.
        #    There is a tiny window between the two renames in which
        #    `filename` is absent; `_load_data` covers it by falling back to
        #    .1. The outcome is shared too (one more cheap agreement), so a
        #    failed rename on one rank does not leave the others waiting in
        #    the next collective of the run.
        problem = None
        cause = None
        try:
            for idx in range(ngen-1, 0, -1):
                fn_from = Path(filename + "." + str(idx))
                fn_to = Path(filename + "." + str(idx+1))
                if fn_from.exists():
                    os.replace(fn_from, fn_to)
            if ngen > 0 and Path(filename).exists():
                os.replace(Path(filename), Path(filename + "." + str(1)))
            os.replace(tmp, Path(filename))
        except OSError as e:
            cause = e
            problem = f"failed to publish checkpoint {filename}: {e}"
        if lockstep:
            reports = odatse.mpi.agree(odatse.mpi.algcomm(), problem)
            problems = [p for _, p, _ in reports if p is not None]
        else:
            problems = [problem] if problem is not None else []
        if problems:
            fail(problems, cause)
        print("save_state: write to {}".format(filename))

    def _load_data(self, filename="state.pickle") -> dict:
        """
        Load the checkpoint for this rank.

        Serially, or for an algorithm whose ranks do not advance in lockstep
        (``_checkpoint_step_key`` is ``None``): the current checkpoint
        ``filename``, or the previous generation ``filename.1`` when the
        current one is missing (a crash in the save window, or external
        removal).

        In a parallel run of a lockstep algorithm (exchange, pamc, ttopt):
        every rank writes its own checkpoint, so a run stopped between two
        ranks' writes leaves the ranks with checkpoints from different
        steps. Ranks restarted from different steps would run different
        numbers of iterations and block in the first collective. The ranks
        therefore agree on the step to restart from: each reads the progress
        (``_checkpoint_step()``) of every generation it has (``filename``,
        ``filename.1``, ...), one ``allgather`` shares them, and every rank
        loads the newest step that all of them have, with a warning on the
        ranks that skip a newer one. If there is no such step, or a rank has
        no readable checkpoint at all, every rank raises the same
        ``CheckpointError``. In continue mode there is no fallback: the
        latest checkpoint must be the same on every rank.

        Collective over the algorithm ranks in that case: every rank enters
        the ``allgather`` whether or not its own read succeeded, so a failure
        on one rank cannot leave the others waiting.

        Parameters
        ----------
        filename
            Name of the file to load the data from.

        Returns
        -------
        dict
            Dictionary containing the loaded data.
        """
        if self._resume_is_collective():
            return self._load_common_data(filename)
        return self._load_latest_data(filename)

    def _resume_is_collective(self) -> bool:
        """Whether loading the checkpoint is a collective over the algorithm
        ranks (``_load_common_data``): a resume or continue of a lockstep
        algorithm in a parallel run."""
        mode = getattr(self, "mode", None)
        return (self._checkpoint_step_key is not None
                and odatse.mpi.algsize() > 1
                and (mode is None or mode.startswith(("resume", "continue"))))

    def _load_latest_data(self, filename) -> dict:
        # Prefer the current checkpoint, but fall back to the previous
        # generation (.1) if it is missing -- e.g. a crash in an older version
        # during the save window, or external corruption of `filename`.
        fn = Path(filename)
        if not fn.exists() and Path(filename + "." + str(1)).exists():
            fn = Path(filename + "." + str(1))
            print("WARNING: {} not found, falling back to {}".format(filename, fn))

        if fn.exists():
            try:
                with open(fn, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                raise exception.CheckpointError(
                    f"failed to read checkpoint from {fn}: {e}"
                ) from e
            print("load_state: load from {}".format(fn))
        else:
            raise exception.CheckpointError(
                f"checkpoint file {filename} does not exist"
            )
        return data

    def _load_common_data(self, filename) -> dict:
        key = self._checkpoint_step_key
        comm = odatse.mpi.algcomm()
        myrank = odatse.mpi.algrank()
        # a failure of this rank before the dispatch (see prepare()): it is
        # shared like a missing checkpoint and raised after the agreement
        earlier = getattr(self, "_failed_before_load", None)

        # 1. the step of every generation this rank has: {step: file}, the
        #    newest file for a step written twice (the final checkpoint of
        #    pamc). Only the step is kept: a snapshot can be large, and the
        #    chosen one is read again below. Nothing raises before the
        #    agreement.
        files = [Path(filename)] + [
            Path(f"{filename}.{i}") for i in range(1, self._checkpoint_generations + 1)
        ]
        found = {}
        problems = []
        if earlier is not None:
            problems.append(f"rank {myrank}: {odatse.mpi._describe(earlier)}")
        else:
            for fn in files:
                try:
                    # inside the try: exists() raises OSError for anything
                    # but "not found" (e.g. PermissionError)
                    if not fn.exists():
                        continue
                    with open(fn, "rb") as f:
                        step = self._checkpoint_step(pickle.load(f))
                except Exception as e:
                    problems.append(f"{fn}: {e}")
                    continue
                found.setdefault(step, fn)
            if not found:
                problems.insert(0, f"no readable checkpoint {filename} on rank {myrank}")
            elif problems:
                print(f"WARNING: rank {myrank}: skipping unreadable checkpoint(s): " + "; ".join(problems))
                problems = []

        # 2. share the outcome and the available steps
        reports = odatse.mpi.agree(comm, "; ".join(problems) or None, sorted(found, reverse=True))

        def fail(message):
            # the same error on every rank, printed once (from the global rank
            # of algorithm rank 0; see exception.Error.report_rank), with the
            # original exception as the cause on the rank that failed
            err = exception.CheckpointError(message)
            err.report_rank = reports[0][0]
            if earlier is not None:
                raise err from earlier
            raise err

        failures = [problem for _, problem, _ in reports if problem is not None]
        if failures:
            fail("cannot resume: " + "; ".join(failures))

        common = set(reports[0][2])
        for _, _, steps in reports[1:]:
            common &= set(steps)
        if not common:
            fail(
                "cannot resume: the checkpoints of the algorithm ranks have no "
                f"{key} in common (they were written at different steps): "
                + ", ".join(f"rank {r}: {steps}" for r, (_, _, steps) in enumerate(reports))
            )

        # 3. the newest step every rank has
        step = max(common)
        fn = found[step]
        newest = max(found)
        if (getattr(self, "mode", None) or "").startswith("continue"):
            # --cont extends a finished run from its final checkpoint, and an
            # older generation is an intermediate one that the continuation
            # would misread (pamc, for instance, advances the temperature once
            # more): no fallback here, the final checkpoint must be on every
            # rank
            latest = [steps[0] for _, _, steps in reports]
            if any(s != step for s in latest):
                fail(
                    "cannot continue: the latest checkpoints of the algorithm "
                    f"ranks are from different steps ({key}): "
                    + ", ".join(f"rank {r}: {s}" for r, s in enumerate(latest))
                    + "; --cont needs the final checkpoint of the finished run on "
                    "every rank (--resume restarts from the newest checkpoint "
                    "common to all ranks)"
                )
        if step != newest:
            print(
                f"WARNING: rank {myrank}: the latest checkpoint ({key} = {newest}) "
                f"is not on every rank; resuming from {key} = {step} ({fn})"
            )
        if myrank == 0:
            behind = [r for r, (_, _, steps) in enumerate(reports) if max(steps) != step]
            if behind:
                print(
                    f"WARNING: resuming every rank from {key} = {step}, the newest "
                    f"checkpoint common to all ranks (newer ones on rank(s) {behind})"
                )
        try:
            with open(fn, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise exception.CheckpointError(
                f"failed to read checkpoint from {fn}: {e}"
            ) from e
        print("load_state: load from {}".format(fn))
        return data

    def _show_parameters(self):
        """
        Show the parameters of the algorithm.
        """
        if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
            info = flatten_dict(self.info)
            for k, v in info.items():
                print("{:16s}: {}".format(k, v))

    def _check_parameters(self, param=None):
        """
        Check the parameters of the algorithm against previous parameters.

        Parameters
        ----------
        param (optional)
            Previous parameters to check against.
        """
        info = flatten_dict(self.info)
        info_prev = flatten_dict(param)

        for k,v in info.items():
            w = info_prev.get(k, None)
            if v != w:
                if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
                    print("WARNING: parameter {} changed from {} to {}".format(k, w, v))
            if odatse.mpi.algrank() is not None and odatse.mpi.algrank() == 0:
                print("{:16s}: {}".format(k, v))

# utility
def flatten_dict(d, parent_key="", separator="."):
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d
        Dictionary to flatten.
    parent_key : str, default : ""
        Key for the parent dictionary.
    separator : str, default : "."
        Separator to use between keys.

    Returns
    -------
    dict
        Flattened dictionary.
    """
    items = []
    if d:
        for key_, val in d.items():
            key = parent_key + separator + key_ if parent_key else key_
            if isinstance(val, dict):
                items.extend(flatten_dict(val, key, separator=separator).items())
            else:
                items.append((key, val))
    return dict(items)
