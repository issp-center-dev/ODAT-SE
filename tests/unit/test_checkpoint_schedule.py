"""The checkpoint decision of lockstep ranks is taken on rank 0, and their
checkpoints are published only once every rank has written its own.

``_checkpoint_due()`` evaluates the step and time conditions on algorithm
rank 0 and broadcasts the result, so that the ranks of exchange / pamc /
ttopt write at the same step (each rank's own clock would fire at different
steps). ``_save_data()`` agrees on the outcome of the temporary-file write
before any rank renames it into place, so a failed write on one rank raises
on every rank and no rank publishes a checkpoint the others do not have.

These tests do not communicate: the parallel cases use a stub algorithm
communicator, the serial cases pin ``algsize()`` to 1 (the unit tests also
run under ``mpirun -np 2`` / ``-np 4``). The algorithm objects are made
without ``__init__`` (the construction consensus of ``FailTogetherMeta``
would otherwise run against the stub).
"""

import os
import pickle
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse.mpi as mpi
from odatse.algorithm._algorithm import AlgorithmBase
from odatse.exception import CheckpointError


class _Lockstep(AlgorithmBase):
    _checkpoint_step_key = "istep"

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


def _make(cls=_Lockstep, checkpoint=True, steps=10, interval=1.0e9):
    alg = cls.__new__(cls)
    alg.checkpoint = checkpoint
    alg.checkpoint_steps = steps
    alg.checkpoint_interval = interval
    return alg


class _Comm:
    """Stand-in for the algorithm communicator of a 2-rank run seen from one
    rank: ``bcast`` hands back what rank 0 decided (``root_value`` when set,
    i.e. when this rank is not rank 0), ``allgather`` appends the prescribed
    other report (``(global rank, problem, payload)``, see odatse.mpi.agree)."""

    def __init__(self, root_value=None, other=None):
        self.root_value = root_value
        self.other = other
        self.size = 2
        self.bcasts = 0
        self.allgathers = 0

    def bcast(self, value, root=0):
        self.bcasts += 1
        return value if self.root_value is None else self.root_value

    def allgather(self, report):
        self.allgathers += 1
        return [report, self.other]


def _parallel(monkeypatch, rank, comm):
    monkeypatch.setattr(mpi, "algsize", lambda: 2)
    monkeypatch.setattr(mpi, "algrank", lambda: rank)
    monkeypatch.setattr(mpi, "rank", lambda: rank)
    monkeypatch.setattr(mpi, "algcomm", lambda: comm)


def _clock(monkeypatch, now):
    monkeypatch.setattr("odatse.algorithm._algorithm.time.time", lambda: now[0])


@pytest.fixture
def serial(monkeypatch):
    """A single algorithm rank, whatever the launcher (the unit tests also
    run under mpirun -np 2 / -np 4): the non-collective path."""
    monkeypatch.setattr(mpi, "algsize", lambda: 1)
    monkeypatch.setattr(mpi, "algrank", lambda: 0)


# --- _checkpoint_due ---

def test_serial_step_condition(serial):
    alg = _make(steps=10)
    alg._start_checkpoint_schedule(0)
    assert not alg._checkpoint_due(9)
    assert alg._checkpoint_due(10)
    # re-armed from the step of the checkpoint
    assert not alg._checkpoint_due(19)
    assert alg._checkpoint_due(20)


def test_schedule_starts_from_the_restored_step(serial):
    alg = _make(steps=10)
    alg._start_checkpoint_schedule(25)
    assert not alg._checkpoint_due(34)
    assert alg._checkpoint_due(35)


def test_serial_time_condition(monkeypatch, serial):
    now = [1000.0]
    _clock(monkeypatch, now)
    alg = _make(steps=10 ** 9, interval=5.0)
    alg._start_checkpoint_schedule(0)
    now[0] = 1004.0
    assert not alg._checkpoint_due(1)
    now[0] = 1005.0
    assert alg._checkpoint_due(2)
    now[0] = 1009.0
    assert not alg._checkpoint_due(3)


def test_checkpointing_off_is_never_due_and_not_collective(monkeypatch):
    comm = _Comm()
    _parallel(monkeypatch, 0, comm)
    alg = _make(checkpoint=False, steps=1)
    alg._start_checkpoint_schedule(0)
    assert not alg._checkpoint_due(5)
    assert comm.bcasts == 0


def test_rank0_decides_and_broadcasts(monkeypatch):
    comm = _Comm()
    _parallel(monkeypatch, 0, comm)
    alg = _make(steps=10)
    alg._start_checkpoint_schedule(0)
    assert not alg._checkpoint_due(9)
    assert alg._checkpoint_due(10)
    assert comm.bcasts == 2


def test_other_ranks_follow_rank0_not_their_own_clock(monkeypatch):
    # rank 1's own conditions say "due" (its clock is ahead), rank 0 says no
    now = [1000.0]
    _clock(monkeypatch, now)
    comm = _Comm(root_value=False)
    _parallel(monkeypatch, 1, comm)
    alg = _make(steps=10 ** 9, interval=5.0)
    alg._start_checkpoint_schedule(0)
    now[0] = 1100.0
    assert not alg._checkpoint_due(1)
    # and when rank 0 says yes, rank 1 writes even though its step is not due
    comm.root_value = True
    alg2 = _make(steps=10 ** 9)
    alg2._start_checkpoint_schedule(0)
    assert alg2._checkpoint_due(1)
    assert comm.bcasts == 2


def test_independent_ranks_decide_locally(monkeypatch):
    comm = _Comm(root_value=False)
    _parallel(monkeypatch, 1, comm)
    alg = _make(_Independent, steps=10)
    alg._start_checkpoint_schedule(0)
    assert alg._checkpoint_due(10)
    assert comm.bcasts == 0


# --- _save_data: two-phase publication ---

@pytest.fixture
def ckpt(tmp_path):
    return str(tmp_path / "status.pickle")


def _load(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def test_serial_write_publishes_and_rotates(ckpt, serial):
    alg = _make()
    alg._save_data({"istep": 1}, ckpt)
    alg._save_data({"istep": 2}, ckpt)
    assert _load(ckpt)["istep"] == 2
    assert _load(ckpt + ".1")["istep"] == 1
    assert not os.path.exists(ckpt + ".tmp")


def test_serial_write_failure_raises(tmp_path, serial):
    alg = _make()
    with pytest.raises(CheckpointError, match="failed to save checkpoint"):
        alg._save_data({"istep": 1}, str(tmp_path / "missing_dir" / "status.pickle"))


def test_parallel_write_agrees_before_publishing(ckpt, monkeypatch):
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    alg._save_data({"istep": 1}, ckpt)
    seen = []
    # at the first agreement (after the write) the previous checkpoint must
    # still be the published one, and the new one must still be the
    # temporary file; the second agreement follows the publication
    def allgather(report):
        seen.append((_load(ckpt)["istep"], os.path.exists(ckpt + ".tmp")))
        return [report, comm.other]
    comm.allgather = allgather
    alg._save_data({"istep": 2}, ckpt)
    assert seen == [(1, True), (2, False)]
    assert _load(ckpt)["istep"] == 2
    assert _load(ckpt + ".1")["istep"] == 1


def test_failure_on_another_rank_withholds_this_ranks_checkpoint(ckpt, monkeypatch):
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    alg._save_data({"istep": 1}, ckpt)   # the previous, published checkpoint
    comm.other = (1, "failed to save checkpoint to status.pickle: disk full", None)
    comm.allgathers = 0
    with pytest.raises(CheckpointError, match="disk full") as info:
        alg._save_data({"istep": 2}, ckpt)
    assert info.value.report_rank == 0
    # nothing published, no temporary file left behind, previous one intact
    assert _load(ckpt)["istep"] == 1
    assert not os.path.exists(ckpt + ".tmp")
    assert not os.path.exists(ckpt + ".1")


def test_failure_here_still_enters_the_agreement(tmp_path, monkeypatch):
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    with pytest.raises(CheckpointError, match="failed to save checkpoint") as info:
        alg._save_data({"istep": 1}, str(tmp_path / "missing_dir" / "status.pickle"))
    assert comm.allgathers == 1
    assert isinstance(info.value.__cause__, OSError)


def test_independent_ranks_write_without_agreement(ckpt, monkeypatch):
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make(_Independent)
    alg._save_data({"istep": 1}, ckpt)
    assert comm.allgathers == 0
    assert _load(ckpt)["istep"] == 1


def test_snapshot_failure_is_shared_like_a_failed_write(ckpt, monkeypatch):
    """A failure in _snapshot() (e.g. flushing an output file) on this rank
    still enters the agreement and is raised on every rank."""
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()

    def snapshot():
        raise OSError("flush failed")
    alg._snapshot = snapshot
    with pytest.raises(CheckpointError, match="snapshot .*flush failed") as info:
        alg._save_state(ckpt)
    assert comm.allgathers == 1
    assert isinstance(info.value.__cause__, OSError)
    assert not os.path.exists(ckpt) and not os.path.exists(ckpt + ".tmp")


def test_snapshot_failure_on_another_rank_withholds_this_ranks_checkpoint(ckpt, monkeypatch):
    comm = _Comm(other=(1, "failed to take the checkpoint snapshot for status.pickle: flush failed", None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    alg.__getstate__ = lambda: {"istep": 1}
    with pytest.raises(CheckpointError, match="flush failed"):
        alg._save_state(ckpt)
    assert not os.path.exists(ckpt) and not os.path.exists(ckpt + ".tmp")


def test_publish_failure_is_shared(ckpt, monkeypatch):
    """A failed rename after the first agreement is shared by a second one."""
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    real_replace = os.replace

    def replace(src, dst):
        if str(dst) == ckpt:
            raise OSError("rename failed")
        real_replace(src, dst)
    monkeypatch.setattr("odatse.algorithm._algorithm.os.replace", replace)
    with pytest.raises(CheckpointError, match="publish .*rename failed") as info:
        alg._save_data({"istep": 1}, ckpt)
    assert comm.allgathers == 2
    assert isinstance(info.value.__cause__, OSError)


def test_publish_failure_on_another_rank_raises_here(ckpt, monkeypatch):
    comm = _Comm(other=(1, None, None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    calls = []

    def allgather(report):
        calls.append(report)
        if len(calls) == 2:   # the publication agreement: the other rank failed
            return [report, (1, "failed to publish checkpoint status.pickle: rename failed", None)]
        return [report, comm.other]
    comm.allgather = allgather
    with pytest.raises(CheckpointError, match="rename failed"):
        alg._save_data({"istep": 1}, ckpt)
    assert _load(ckpt)["istep"] == 1   # this rank's own publication went through


def test_unlink_error_does_not_replace_the_agreed_error(ckpt, monkeypatch, capsys):
    comm = _Comm(other=(1, "failed to save checkpoint to status.pickle: disk full", None))
    _parallel(monkeypatch, 0, comm)
    alg = _make()
    monkeypatch.setattr("odatse.algorithm._algorithm.Path.unlink",
                        lambda self, missing_ok=False: (_ for _ in ()).throw(PermissionError("no")))
    with pytest.raises(CheckpointError, match="disk full"):
        alg._save_data({"istep": 1}, ckpt)
    assert "could not remove" in capsys.readouterr().out
