"""Resuming a lockstep algorithm in parallel loads the newest checkpoint
common to all ranks.

Every rank writes its own checkpoint, so a run stopped between two ranks'
writes leaves them with checkpoints from different steps; ranks restarted
from different steps block in the first collective. ``_load_data()`` of a
lockstep algorithm (``_checkpoint_step_key`` set) therefore shares the steps
of the generations each rank has and picks the newest one every rank has.

These tests run serially: the algorithm communicator is replaced by a stub
whose ``allgather`` returns this rank's report together with the reports
the test prescribes for the other ranks. The algorithm objects are made
without ``__init__`` (the construction consensus of ``FailTogetherMeta``
would otherwise run against the stub).
"""

import os
import pickle
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse.mpi as mpi
from odatse.algorithm._algorithm import AlgorithmBase, AlgorithmStatus
from odatse.exception import CheckpointError


class _Lockstep(AlgorithmBase):
    _checkpoint_step_key = "istep"
    mode = "resume"

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


class _Independent(_Lockstep):
    _checkpoint_step_key = None


class _Comm:
    """Stand-in for the algorithm communicator: ``allgather`` returns the
    local report followed by the prescribed reports of the other ranks;
    ``Allreduce`` (the phase consensus) adds the other ranks' success flags."""

    def __init__(self, others, others_ok=None):
        self.others = others
        self.size = len(others) + 1
        self.others_ok = len(others) if others_ok is None else others_ok
        self.calls = 0

    def allgather(self, report):
        self.calls += 1
        return [report] + list(self.others)

    def Allreduce(self, send, recv):
        recv[0] = send[0] + self.others_ok


def _write_generations(path, steps):
    """Write ``path`` and its generations so that ``path`` holds the first
    step of ``steps``, ``path.1`` the second, and so on."""
    for i, step in enumerate(steps):
        fn = path if i == 0 else f"{path}.{i}"
        with open(fn, "wb") as f:
            pickle.dump({"istep": step, "tag": f"gen{i}"}, f)


def _parallel(monkeypatch, others, others_ok=None):
    comm = _Comm(others, others_ok)
    monkeypatch.setattr(mpi, "algsize", lambda: comm.size)
    monkeypatch.setattr(mpi, "algrank", lambda: 0)
    monkeypatch.setattr(mpi, "rank", lambda: 0)
    monkeypatch.setattr(mpi, "algcomm", lambda: comm)
    return comm


def _report(steps, problem=None, grank=1):
    """What another rank contributes to the agreement (odatse.mpi.agree)."""
    return (grank, problem, sorted(steps, reverse=True))


@pytest.fixture
def ckpt(tmp_path):
    return str(tmp_path / "status.pickle")


# --- serial / independent ranks: the current file, unchanged behaviour ---

def test_serial_loads_the_current_checkpoint(ckpt):
    _write_generations(ckpt, [300, 200, 100])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 300


def test_serial_falls_back_to_previous_generation_when_current_is_missing(ckpt):
    _write_generations(ckpt, [300, 200])
    os.remove(ckpt)
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 200


def test_serial_missing_checkpoint_raises(ckpt):
    with pytest.raises(CheckpointError, match="does not exist"):
        _Lockstep.__new__(_Lockstep)._load_data(ckpt)


def test_independent_ranks_do_not_agree_on_a_step(ckpt, monkeypatch):
    _write_generations(ckpt, [300, 200])
    comm = _parallel(monkeypatch, [_report([250])])
    data = _Independent.__new__(_Independent)._load_data(ckpt)
    assert data["istep"] == 300
    assert comm.calls == 0


# --- lockstep ranks in parallel ---

def test_all_ranks_current_picks_the_newest(ckpt, monkeypatch, capsys):
    _write_generations(ckpt, [300, 200, 100])
    comm = _parallel(monkeypatch, [_report([300, 200, 100]), _report([300, 200])])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 300
    assert comm.calls == 1
    assert "WARNING" not in capsys.readouterr().out


def test_falls_back_to_the_newest_common_step(ckpt, monkeypatch, capsys):
    # this rank wrote 300 before the stop; the other rank only reached 200
    _write_generations(ckpt, [300, 200, 100])
    _parallel(monkeypatch, [_report([200, 100])])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 200
    assert data["tag"] == "gen1"
    out = capsys.readouterr().out
    assert "resuming from istep = 200" in out
    assert "status.pickle.1" in out


def test_other_rank_ahead_keeps_this_ranks_current(ckpt, monkeypatch):
    _write_generations(ckpt, [200, 100])
    _parallel(monkeypatch, [_report([300, 200, 100])])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 200


def test_a_step_written_twice_keeps_the_newest_file(ckpt, monkeypatch):
    # pamc writes its final checkpoint twice at the same step
    _write_generations(ckpt, [300, 300, 200])
    _parallel(monkeypatch, [_report([300, 200])])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["tag"] == "gen0"


def test_no_common_step_raises_on_every_rank(ckpt, monkeypatch):
    _write_generations(ckpt, [300, 200])
    _parallel(monkeypatch, [_report([301, 201])])
    with pytest.raises(CheckpointError, match="no istep in common") as info:
        _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert "rank 0: [300, 200]" in str(info.value)
    assert "rank 1: [301, 201]" in str(info.value)
    assert info.value.report_rank == 0


def test_missing_checkpoint_on_another_rank_raises_here_too(ckpt, monkeypatch):
    _write_generations(ckpt, [300])
    _parallel(monkeypatch, [_report([], problem="no readable checkpoint status.pickle on rank 1")])
    with pytest.raises(CheckpointError, match="no readable checkpoint .* on rank 1"):
        _Lockstep.__new__(_Lockstep)._load_data(ckpt)


def test_missing_checkpoint_here_still_enters_the_collective(ckpt, monkeypatch):
    comm = _parallel(monkeypatch, [_report([300])])
    with pytest.raises(CheckpointError, match="no readable checkpoint .* on rank 0"):
        _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert comm.calls == 1


def test_unreadable_generation_is_skipped_with_a_warning(ckpt, monkeypatch, capsys):
    _write_generations(ckpt, [300, 200])
    with open(ckpt, "wb") as f:
        f.write(b"not a pickle")
    _parallel(monkeypatch, [_report([300, 200])])
    data = _Lockstep.__new__(_Lockstep)._load_data(ckpt)
    assert data["istep"] == 200
    assert "skipping unreadable checkpoint" in capsys.readouterr().out


# --- a failure before the dispatch is carried through the agreement ---

class _FailingRunner:
    def prepare(self, proc_dir):
        raise RuntimeError("injected runner failure")

    def post(self):
        pass


def _resuming(ckpt, runner):
    alg = _Lockstep.__new__(_Lockstep)
    alg.mode = "resume"
    alg.runner = runner
    alg.proc_dir = os.path.dirname(ckpt)
    alg.checkpoint_file = ckpt
    return alg


def test_runner_failure_before_resume_is_shared_then_raised(ckpt, monkeypatch):
    """prepare() on a rank whose runner.prepare() failed still enters the
    agreement (the other ranks would otherwise wait in it), reports the
    failure there, and raises it afterwards with the original as the cause."""
    _write_generations(ckpt, [300])
    comm = _parallel(monkeypatch, [_report([300])], others_ok=0)
    alg = _resuming(ckpt, _FailingRunner())
    with pytest.raises(CheckpointError, match="injected runner failure") as info:
        alg.prepare()
    assert comm.calls == 1
    assert isinstance(info.value.__cause__, RuntimeError)
    assert info.value.report_rank == 0
    assert not hasattr(alg, "status") or alg.status < AlgorithmStatus.PREPARE


def test_runner_failure_on_another_rank_raises_here(ckpt, monkeypatch):
    _write_generations(ckpt, [300])
    _parallel(monkeypatch, [_report([], problem="rank 1: RuntimeError: injected runner failure")])
    with pytest.raises(CheckpointError, match="rank 1: RuntimeError"):
        _Lockstep.__new__(_Lockstep)._load_data(ckpt)


def test_runner_failure_in_serial_resume_is_raised_as_is(ckpt):
    """Without the agreement there is nothing to carry the failure through:
    the runner's exception comes out of prepare() unchanged."""
    _write_generations(ckpt, [300])
    alg = _resuming(ckpt, _FailingRunner())
    with pytest.raises(RuntimeError, match="injected runner failure"):
        alg.prepare()
