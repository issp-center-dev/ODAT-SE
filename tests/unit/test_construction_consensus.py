"""Construction of an algorithm, a solver or a runner takes part in the
agreement of odatse.mpi.fail_together() (odatse.mpi.FailTogetherMeta): a
constructor that fails on some processes only must make every process leave
the constructor with an exception, instead of leaving the others waiting for
the failed one."""

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


# --- Solver and Runner are protected in the same way (issue #112) ----------

import odatse
import odatse.solver


class _Solver(odatse.solver.SolverBase):
    """Minimal solver whose constructor fails on a chosen rank; its ``name``
    property can be made to fail too, which is the first thing Runner.__init__
    touches, to inject a failure into the construction of the Runner."""
    fail_rank = None
    name_fails_on = None

    def __init__(self, info):
        if self.fail_rank is not None and mpi.algrank() == self.fail_rank:
            raise exception.InputError(f"injected solver failure on rank {self.fail_rank}")
        super().__init__(info)

    @property
    def name(self):
        if self.name_fails_on is not None and mpi.algrank() == self.name_fails_on:
            raise exception.InputError(f"injected runner failure on rank {self.name_fails_on}")
        return "test"

    def evaluate(self, x, args=()):
        return 0.0


@pytest.fixture
def info():
    return odatse.Info({
        "base": {"dimension": 2, "output_dir": "output"},
        "algorithm": {"name": "test"},
        "solver": {},
        "runner": {},
    })


@pytest.fixture(autouse=True)
def _reset_solver():
    _Solver.fail_rank = None
    _Solver.name_fails_on = None
    yield
    _Solver.fail_rank = None
    _Solver.name_fails_on = None


def test_solver_and_runner_use_the_metaclass():
    assert type(odatse.solver.SolverBase) is mpi.FailTogetherMeta
    assert type(odatse.Runner) is mpi.FailTogetherMeta
    assert type(AlgorithmBase) is mpi.FailTogetherMeta


def test_solver_and_runner_construct_normally(info):
    solver = _Solver(info)
    runner = odatse.Runner(solver, info)
    assert runner.solver is solver


def test_solver_failure_on_one_rank_releases_the_others(info):
    """No fail_together() block around the call: the constructor itself
    carries the agreement, as a user script or a host program relies on."""
    _Solver.fail_rank = mpi.algsize() - 1
    if mpi.algrank() == _Solver.fail_rank:
        with pytest.raises(exception.InputError, match="injected solver failure") as excinfo:
            _Solver(info)
        assert excinfo.value.rank_local is True
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            _Solver(info)


def test_runner_failure_on_one_rank_releases_the_others(info):
    solver = _Solver(info)
    _Solver.name_fails_on = mpi.algsize() - 1
    if mpi.algrank() == _Solver.name_fails_on:
        with pytest.raises(exception.InputError, match="injected runner failure") as excinfo:
            odatse.Runner(solver, info)
        assert excinfo.value.rank_local is True
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            odatse.Runner(solver, info)


def test_solver_new_bypasses_the_agreement():
    solver = _Solver.__new__(_Solver)
    assert not hasattr(solver, "dimension")


def test_constructor_inside_a_block_leaves_the_agreement_to_the_block(info):
    """A process that fails in a fail_together() block *before* a
    construction the others reach: with the constructors agreeing on their
    own inside the block, that process would agree once (the block's) while
    the others agree twice (the construction's, then the block's) and hang.
    Inside a block the constructors therefore do not agree; the block does
    when it is left, on every process."""
    fail_rank = mpi.algsize() - 1
    if mpi.algrank() == fail_rank:
        with pytest.raises(PermissionError, match="injected"):
            with mpi.fail_together():
                raise PermissionError("injected failure before the construction")
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            with mpi.fail_together():
                solver = _Solver(info)          # no agreement of its own here
                odatse.Runner(solver, info)


def test_constructor_outside_a_block_agrees_on_its_own(info):
    """Without an enclosing block a constructor failing on one process
    releases the others by itself (the embedding case)."""
    _Solver.fail_rank = mpi.algsize() - 1
    if mpi.algrank() == _Solver.fail_rank:
        with pytest.raises(exception.InputError):
            _Solver(info)
    else:
        with pytest.raises(mpi.OtherAlgorithmProcessError):
            _Solver(info)
    # and the depth counter is back to zero afterwards
    assert mpi._block_depth == 0
