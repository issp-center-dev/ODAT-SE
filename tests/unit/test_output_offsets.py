"""The Monte Carlo output files are cut back to the checkpointed positions
on resume, and a failure to do so on one rank is raised on every rank.

These tests do not communicate: the parallel cases use a stub algorithm
communicator (the unit tests also run under ``mpirun -np 2`` / ``-np 4``,
so the serial cases pin ``algsize()`` to 1). The algorithm objects are made
without ``__init__``.
"""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse.mpi as mpi
from odatse.algorithm.montecarlo import AlgorithmBase
from odatse.exception import CheckpointError
from odatse.util.data_writer import DataWriter


class _MC(AlgorithmBase):
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


def _make(offsets):
    alg = _MC.__new__(_MC)
    alg.mode = "resume"
    alg._output_offsets = offsets
    return alg


class _Comm:
    def __init__(self, other=(1, None, None)):
        self.other = other
        self.size = 2
        self.allgathers = 0

    def allgather(self, report):
        self.allgathers += 1
        return [report, self.other]


def _parallel(monkeypatch, comm):
    monkeypatch.setattr(mpi, "algsize", lambda: 2)
    monkeypatch.setattr(mpi, "algrank", lambda: 0)
    monkeypatch.setattr(mpi, "rank", lambda: 0)
    monkeypatch.setattr(mpi, "algcomm", lambda: comm)


@pytest.fixture
def serial(monkeypatch):
    monkeypatch.setattr(mpi, "algsize", lambda: 1)
    monkeypatch.setattr(mpi, "algrank", lambda: 0)


def _written(path, lines):
    w = DataWriter(path, item_list=["step", "x"])
    for line in lines:
        w.write(*line)
    w.flush()
    position = w.tell()
    w.close()
    return position


def test_files_are_cut_back_to_the_checkpointed_positions(tmp_path, serial):
    os.chdir(tmp_path)
    pos = _written("trial.txt", [(0, 1.0), (1, 2.0)])
    with open("trial.txt", "a") as f:
        f.write("2 3.0\n3 4")   # written after the checkpoint, partial last line
    _written("result.txt", [(0, 1.0)])
    trial = DataWriter("trial.txt", mode="a", item_list=["step", "x"])
    result = DataWriter("result.txt", mode="a", item_list=["step", "x"])
    _make({"trial.txt": pos})._set_writer(trial, result)
    trial.write(2, 3.5)
    trial.close()
    result.close()
    assert open("trial.txt").read() == "# step x\n0 1.0\n1 2.0\n2 3.5\n"
    assert open("result.txt").read() == "# step x\n0 1.0\n"   # no offset recorded: untouched


def test_a_file_shorter_than_the_recorded_position_is_an_error(tmp_path, serial):
    os.chdir(tmp_path)
    pos = _written("trial.txt", [(0, 1.0), (1, 2.0)])
    os.remove("trial.txt")   # the output was removed between the runs
    trial = DataWriter("trial.txt", mode="a", item_list=["step", "x"])
    result = DataWriter("result.txt", mode="a", item_list=["step", "x"])
    with pytest.raises(CheckpointError, match="shorter .* than the position recorded") as info:
        _make({"trial.txt": pos})._set_writer(trial, result)
    assert isinstance(info.value.__cause__, ValueError)
    trial.close()
    result.close()
    assert os.path.getsize("trial.txt") == 0   # not padded


def test_failure_restoring_the_files_is_shared(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    pos = _written("trial.txt", [(0, 1.0)])
    os.remove("trial.txt")
    comm = _Comm()
    _parallel(monkeypatch, comm)
    trial = DataWriter("trial.txt", mode="a", item_list=["step", "x"])
    with pytest.raises(CheckpointError, match="rank 0: .*shorter") as info:
        _make({"trial.txt": pos})._set_writer(trial, None)
    assert comm.allgathers == 1
    assert info.value.report_rank == 0
    trial.close()


def test_failure_on_another_rank_raises_here(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    pos = _written("trial.txt", [(0, 1.0)])
    comm = _Comm(other=(1, "cannot restore the output files of rank 1: trial.txt is shorter", None))
    _parallel(monkeypatch, comm)
    trial = DataWriter("trial.txt", mode="a", item_list=["step", "x"])
    with pytest.raises(CheckpointError, match="rank 1: trial.txt is shorter"):
        _make({"trial.txt": pos})._set_writer(trial, None)
    trial.close()


def test_no_offsets_means_one_agreement_and_no_change(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    _written("trial.txt", [(0, 1.0)])
    comm = _Comm()
    _parallel(monkeypatch, comm)
    trial = DataWriter("trial.txt", mode="a", item_list=["step", "x"])
    _make({})._set_writer(trial, None)
    trial.close()
    assert comm.allgathers == 1
    assert open("trial.txt").read() == "# step x\n0 1.0\n"
