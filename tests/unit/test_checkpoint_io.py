import os
import sys
import pickle
from pathlib import Path

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.algorithm._algorithm import AlgorithmBase
from odatse.exception import CheckpointError


class _Dummy(AlgorithmBase):
    """Concrete subclass so we can build an instance (via __new__) and call the
    checkpoint I/O helpers, which only depend on their arguments."""
    def __init__(self):
        pass

    def _initialize(self):
        pass

    def _prepare(self):
        pass

    def _run(self):
        pass

    def _post(self):
        return {}


def _alg():
    return _Dummy.__new__(_Dummy)


def _read(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_save_load_roundtrip(tmp_path):
    alg = _alg()
    fn = str(tmp_path / "state.pickle")
    alg._save_data({"gen": 42}, filename=fn)
    assert Path(fn).exists()
    assert alg._load_data(filename=fn) == {"gen": 42}


def test_rotation_keeps_previous_generations(tmp_path):
    alg = _alg()
    fn = str(tmp_path / "state.pickle")
    for g in range(1, 5):
        alg._save_data({"gen": g}, filename=fn)

    # current file holds the newest generation ...
    assert alg._load_data(filename=fn) == {"gen": 4}
    # ... and the three backups hold the previous generations, newest first.
    assert _read(fn + ".1") == {"gen": 3}
    assert _read(fn + ".2") == {"gen": 2}
    assert _read(fn + ".3") == {"gen": 1}


def test_current_file_survives_crash_at_atomic_swap(tmp_path, monkeypatch):
    """Regression for the non-atomic rotation: a crash at the final swap must
    leave the previous checkpoint intact and loadable. Previously the current
    file was moved aside first, so a crash in that window destroyed it."""
    alg = _alg()
    fn = str(tmp_path / "state.pickle")
    alg._save_data({"gen": 1}, filename=fn)  # establish a valid checkpoint

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash at swap")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        alg._save_data({"gen": 2}, filename=fn)

    # the current checkpoint is still present and still the valid old one
    assert Path(fn).exists()
    assert alg._load_data(filename=fn) == {"gen": 1}


def test_load_falls_back_to_generation_1(tmp_path):
    """If the current file is missing, load recovers from the .1 backup."""
    alg = _alg()
    fn = str(tmp_path / "state.pickle")
    alg._save_data({"gen": 1}, filename=fn)
    alg._save_data({"gen": 2}, filename=fn)  # filename=gen2, .1=gen1

    Path(fn).unlink()  # lose the current checkpoint
    assert alg._load_data(filename=fn) == {"gen": 1}


def test_load_state_missing_file_raises_checkpoint_error(tmp_path):
    """A missing checkpoint must raise CheckpointError, not sys.exit."""
    alg = _alg()
    fn = str(tmp_path / "does_not_exist.pickle")
    with pytest.raises(CheckpointError):
        alg._load_state(fn, mode="resume")


def test_load_data_corrupt_file_raises_checkpoint_error(tmp_path):
    """An unreadable/corrupt checkpoint must raise CheckpointError."""
    alg = _alg()
    fn = tmp_path / "state.pickle"
    fn.write_bytes(b"this is not a pickle")
    with pytest.raises(CheckpointError):
        alg._load_data(filename=str(fn))


def test_save_data_failure_raises_checkpoint_error(tmp_path):
    """A failure while writing the checkpoint must raise CheckpointError."""
    alg = _alg()
    fn = str(tmp_path / "state.pickle")
    # a lambda cannot be pickled, so pickle.dump raises
    with pytest.raises(CheckpointError):
        alg._save_data({"bad": lambda x: x}, filename=fn)
