# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
from abc import ABCMeta, abstractmethod
from enum import IntEnum
import time
import os
import pathlib
import pickle
import shutil
import copy

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

class AlgorithmBase(metaclass=ABCMeta):
    """Base class for algorithms, providing common functionality and structure.

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

        Validates the MPI configuration, restores the timer, and checks
        that algorithm parameters are consistent.  Subclasses should call
        ``super()._apply_state(data, mode=mode, restore_rng=restore_rng)``
        and then handle their own fields (RNG restore, subclass-specific
        ``_checkpoint_attrs``, continue-mode semantics, etc.).

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
        # create directory for each rank in case every rank has some output
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        # Some cache of the filesystem may delay making a dictionary
        # especially when mkdir just after removing the old one
        while not self.proc_dir.is_dir():
            time.sleep(0.1)

        if odatse.mpi.algcomm() is not None and odatse.mpi.algsize() > 1:
            odatse.mpi.algcomm().Barrier()

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

        # runner lifecycle starts (commit 936e48: moved here from run())
        self.runner.prepare(self.proc_dir)

        # checkpoint dispatch
        restore_rng = not self.mode.endswith("-resetrand")
        if self.mode.startswith("init"):
            self._initialize()
        elif self.mode.startswith("resume"):
            self._load_state(self.checkpoint_file, mode="resume", restore_rng=restore_rng)
        elif self.mode.startswith("continue"):
            self._load_state(self.checkpoint_file, mode="continue", restore_rng=restore_rng)
        else:
            raise RuntimeError(f"unknown mode {self.mode}")
        
        # algorithm-specific preparation
        try:
            self._prepare()
            res = np.array([1])
        except Exception:
            res = np.array([0])
            if odatse.mpi.algsize() > 1:
                resall = np.array([0])
                odatse.mpi.algcomm().Allreduce(res, resall)
            raise
        if odatse.mpi.algsize() > 1:
            resall = np.array([0])
            odatse.mpi.algcomm().Allreduce(res, resall)
            if resall[0] < odatse.mpi.algsize():
                raise odatse.mpi.OtherAlgorithmProcessError()

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

        try:
            original_dir = os.getcwd()
            os.chdir(self.proc_dir)
            self._run()
            os.chdir(original_dir)
        except Exception:
            if odatse.mpi.algsize() > 1:
                res = np.array([0])
                resall = np.array([0])
                odatse.mpi.algcomm().Allreduce(res, resall)
            raise

        if odatse.mpi.algsize() > 1:
            res = np.array([1])
            resall = np.array([0])
            odatse.mpi.algcomm().Allreduce(res, resall)
            if resall[0] < odatse.mpi.algsize():
                raise odatse.mpi.OtherAlgorithmProcessError()

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

        try:
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            result = self._post()
            os.chdir(original_dir)
        except Exception:
            if odatse.mpi.algsize() > 1:
                res = np.array([0])
                resall = np.array([0])
                odatse.mpi.algcomm().Allreduce(res, resall)
            raise

        if odatse.mpi.algsize() > 1:
            res = np.array([1])
            resall = np.array([0])
            odatse.mpi.algcomm().Allreduce(res, resall)
            if resall[0] < odatse.mpi.algsize():
                raise odatse.mpi.OtherAlgorithmProcessError()

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

    def _save_state(self, filename) -> None:
        """Save a checkpoint snapshot to *filename*.

        Uses ``__getstate__()`` to collect all fields declared in
        ``_checkpoint_attrs`` across the MRO, then delegates to
        ``_save_data()`` for versioned pickle storage.

        Override in subclasses **only** when extra files must be written
        alongside the pickle (e.g. an external policy object).  In that
        case call ``super()._save_state(filename)`` first.
        """
        self._save_data(self.__getstate__(), filename)

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
        if not data:
            print(f"ERROR: failed to load checkpoint from {filename}")
            sys.exit(1)
        self._apply_state(data, mode=mode, restore_rng=restore_rng)

    # ------------------------------------------------------------------
    # main() – orchestrates the three phases with timing and MPI barriers.
    # ------------------------------------------------------------------

    def _main_algorithm(self):
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

            except Exception:
                self.__signal_workers(odatse.mpi.MSG_ABORT)
                raise

            self.__signal_workers(odatse.mpi.MSG_FINISHED)
            return res
        else: # Worker process for solver
            assert odatse.mpi.solrank() > 0
            signal = np.array([0])
            xp = np.zeros(self.runner.solver.dimension)
            while True:
                odatse.mpi.solcomm().Bcast(signal, root=0)
                if signal[0] == odatse.mpi.MSG_FINISHED:
                    break
                elif signal[0] == odatse.mpi.MSG_ABORT:
                    sys.exit(0)
                elif signal[0] == odatse.mpi.MSG_EVALUATE:
                    odatse.mpi.solcomm().Bcast(xp, root=0)
                    args = odatse.mpi.solcomm().bcast(None, root=0)
                    self.runner.solver.evaluate(xp, args)
                else:
                    raise ValueError(f"Unknown signal: {signal[0]}")
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

    def _save_data(self, data, filename="state.pickle", ngen=3) -> None:
        """
        Save data to a file with versioning.

        Parameters
        ----------
        data
            Data to be saved.
        filename
            Name of the file to save the data.
        ngen : int, default: 3
            Number of generations for versioning.
        """
        try:
            fn = Path(filename + ".tmp")
            with open(fn, "wb") as f:
                pickle.dump(data, f)
                # Make sure the new checkpoint is durable on disk before it
                # replaces the current one below.
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print("ERROR: {}".format(e))
            sys.exit(1)

        # Rotate the older backup generations: .(ngen-1) -> .ngen, ..., .1 -> .2
        for idx in range(ngen-1, 0, -1):
            fn_from = Path(filename + "." + str(idx))
            fn_to = Path(filename + "." + str(idx+1))
            if fn_from.exists():
                shutil.move(fn_from, fn_to)
        # Keep the current checkpoint as .1 by *copying* it (not moving), so
        # that `filename` always points to a complete checkpoint -- there is no
        # window in which it is missing. Then atomically swap in the new one
        # with os.replace(), which is the only step that touches `filename`.
        if ngen > 0 and Path(filename).exists():
            shutil.copy2(Path(filename), Path(filename + "." + str(1)))
        os.replace(Path(filename + ".tmp"), Path(filename))
        print("save_state: write to {}".format(filename))

    def _load_data(self, filename="state.pickle") -> dict:
        """
        Load data from a file.

        Parameters
        ----------
        filename
            Name of the file to load the data from.

        Returns
        -------
        dict
            Dictionary containing the loaded data.
        """
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
                print("ERROR: {}".format(e))
                sys.exit(1)
            print("load_state: load from {}".format(fn))
        else:
            print("ERROR: file {} not exist.".format(filename))
            data = {}
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
