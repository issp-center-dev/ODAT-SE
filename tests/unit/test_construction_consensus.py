"""Construction of an algorithm takes part in a consensus (see
_AlgorithmMeta in odatse.algorithm._algorithm): a constructor that fails on
some processes only must make every process leave Algorithm(...) with an
exception, instead of leaving the others waiting for the failed one."""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse.mpi as mpi
from odatse import exception
from odatse.algorithm._algorithm import AlgorithmBase


class _Alg(AlgorithmBase):
    """Minimal concrete algorithm whose constructor fails on a chosen rank."""
    fail_rank = None   # algrank whose __init__ raises
    exc = RuntimeError

    def __init__(self):
        # no super().__init__(): the base constructor needs an Info; the
        # consensus must cover the subclass constructor as a whole anyway
        if self.fail_rank is not None and mpi.algrank() == self.fail_rank:
            raise self.exc(f"injected construction failure on rank {self.fail_rank}")
        self.constructed = True

    def _initialize(self):
        pass

    def _prepare(self):
        pass

    def _run(self):
        pass

    def _post(self):
        pass


@pytest.fixture(autouse=True)
def _reset_alg():
    _Alg.fail_rank = None
    _Alg.exc = RuntimeError
    yield
    _Alg.fail_rank = None
    _Alg.exc = RuntimeError


def test_successful_construction_returns_the_instance():
    alg = _Alg()
    assert alg.constructed is True


def test_failure_on_every_rank_propagates_the_own_exception():
    """Symmetric failure: each rank re-raises what its constructor raised."""
    _Alg.fail_rank = mpi.algrank()
    with pytest.raises(RuntimeError, match=f"on rank {mpi.algrank()}"):
        _Alg()


def test_failure_marks_a_framework_error_rank_local():
    _Alg.fail_rank = mpi.algrank()
    _Alg.exc = exception.InputError
    with pytest.raises(exception.InputError) as excinfo:
        _Alg()
    assert excinfo.value.rank_local is True


def test_failure_on_one_rank_releases_the_others():
    """Asymmetric failure: the failing rank re-raises its exception, every
    other algorithm rank raises OtherAlgorithmProcessError instead of
    entering main() and waiting in the prepare-phase consensus. Serially
    this is the symmetric case (rank 0 fails)."""
    _Alg.fail_rank = mpi.algsize() - 1
    if mpi.algrank() == _Alg.fail_rank:
        with pytest.raises(RuntimeError, match="injected construction failure"):
            _Alg()
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            _Alg()


def test_system_exit_in_the_constructor_releases_the_others():
    _Alg.fail_rank = mpi.algsize() - 1
    _Alg.exc = SystemExit
    if mpi.algrank() == _Alg.fail_rank:
        with pytest.raises(SystemExit):
            _Alg()
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            _Alg()


def test_abstract_class_error_propagates():
    """A TypeError from abc (symmetric on every rank) is re-raised as is."""
    class _Incomplete(AlgorithmBase):
        def __init__(self):
            pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()


def test_new_bypasses_the_consensus():
    """Tests build bare instances with __new__; no collective is issued."""
    alg = _Alg.__new__(_Alg)
    assert not hasattr(alg, "constructed")


class _BaseAlg(_Alg):
    """Runs the real AlgorithmBase.__init__, which holds a collective
    (the algcomm synchronisation after creating the per-rank directory)."""

    def __init__(self, info):
        AlgorithmBase.__init__(self, info)
        self.constructed = True


def test_directory_failure_on_one_rank_before_the_base_synchronisation(monkeypatch):
    """A rank that cannot create its output directory must not leave the
    other algorithm ranks in the synchronisation of the base constructor:
    every rank leaves Algorithm(...), the failing one with its own error."""
    import pathlib
    import odatse

    info = odatse.Info({
        "base": {"dimension": 2, "output_dir": "output"},
        "algorithm": {"name": "test"},
        "solver": {},
    })
    fail_rank = mpi.algsize() - 1
    if mpi.algrank() == fail_rank:
        def _mkdir(self, *args, **kwargs):
            raise PermissionError(f"injected mkdir failure: {self}")
        monkeypatch.setattr(pathlib.Path, "mkdir", _mkdir)
        with pytest.raises(PermissionError, match="injected mkdir failure"):
            _BaseAlg(info)
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            _BaseAlg(info)
