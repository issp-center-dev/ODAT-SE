"""A failure of one rank inside the main loop is shared at the next
synchronisation point, so that the other ranks raise instead of waiting
for it in their next collective.

These tests do not communicate: the parallel cases use a stub algorithm
communicator, the serial case pins ``algsize()`` to 1 (the unit tests also
run under ``mpirun -np 2`` / ``-np 4``). The algorithm objects are made
without ``__init__``.
"""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse.mpi as mpi
from odatse.algorithm._algorithm import AlgorithmBase


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


class _Comm:
    def __init__(self, other=(1, None, None)):
        self.other = other
        self.size = 2
        self.reports = []

    def allgather(self, report):
        self.reports.append(report)
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


def test_nothing_failed(monkeypatch):
    comm = _Comm()
    _parallel(monkeypatch, comm)
    _Lockstep.__new__(_Lockstep)._agree_or_fail(None)
    assert comm.reports == [(0, None, None)]


def test_own_failure_is_shared_then_reraised(monkeypatch):
    comm = _Comm()
    _parallel(monkeypatch, comm)
    err = ValueError("solver failed")
    with pytest.raises(ValueError) as info:
        _Lockstep.__new__(_Lockstep)._agree_or_fail(err)
    assert info.value is err
    assert comm.reports == [(0, "rank 0: ValueError: solver failed", None)]


def test_failure_on_another_rank_makes_this_rank_leave_quietly(monkeypatch):
    comm = _Comm(other=(1, "rank 1: ValueError: solver failed", None))
    _parallel(monkeypatch, comm)
    with pytest.raises(mpi.OtherAlgorithmProcessError):
        _Lockstep.__new__(_Lockstep)._agree_or_fail(None)


def test_own_failure_wins_over_another_ranks(monkeypatch):
    comm = _Comm(other=(1, "rank 1: ValueError: solver failed", None))
    _parallel(monkeypatch, comm)
    with pytest.raises(RuntimeError, match="mine"):
        _Lockstep.__new__(_Lockstep)._agree_or_fail(RuntimeError("mine"))


def test_serial_reraises_without_communication(serial):
    alg = _Lockstep.__new__(_Lockstep)
    alg._agree_or_fail(None)
    with pytest.raises(ValueError, match="solver failed"):
        alg._agree_or_fail(ValueError("solver failed"))


def test_independent_ranks_do_not_communicate(monkeypatch):
    comm = _Comm(other=(1, "rank 1: ValueError: solver failed", None))
    _parallel(monkeypatch, comm)
    alg = _Independent.__new__(_Independent)
    alg._agree_or_fail(None)
    assert comm.reports == []
    with pytest.raises(ValueError):
        alg._agree_or_fail(ValueError("solver failed"))
