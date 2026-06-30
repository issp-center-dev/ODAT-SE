"""Unit tests for odatse.mpi.

These exercise the two-layer (algorithm / solver) communicator partitioning
and the checkpoint mixin.  They run both serially (single rank) and under
``mpirun -n N``; tests that need several ranks skip uniformly on every rank so
that no MPI collective is left unmatched.

The _MPIContext tests build *fresh* context objects rather than touching the
module-level singleton (which conftest has already set up), so setup() can be
called and validated in isolation.
"""
import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

import odatse.mpi as mpi

needs_mpi = pytest.mark.skipif(
    not mpi.enabled(), reason="requires an MPI build (mpi4py)"
)


# --------------------------------------------------------------------------- #
#  Module-level constants
# --------------------------------------------------------------------------- #

def test_message_constants_are_distinct():
    assert len({mpi.MSG_ABORT, mpi.MSG_FINISHED, mpi.MSG_EVALUATE}) == 3


def test_other_algorithm_process_error_is_exception():
    assert issubclass(mpi.OtherAlgorithmProcessError, Exception)


# --------------------------------------------------------------------------- #
#  No-MPI stub context (logic available regardless of the build)
# --------------------------------------------------------------------------- #

def test_nompi_context_reports_serial_values():
    ctx = mpi._NoMPIContext()
    ctx.setup(nalg=8, nsolve=4)  # arguments are accepted but ignored
    assert ctx.size() == 1
    assert ctx.rank() == 0
    assert ctx.algsize() == 1
    assert ctx.algrank() == 0
    assert ctx.solsize() == 1
    assert ctx.solrank() == 0
    assert ctx.run_on_algorithm() is True
    assert ctx.enabled() is False
    assert ctx.comm() is None
    assert ctx.algcomm() is None
    assert ctx.solcomm() is None


def test_nompi_context_getstate():
    ctx = mpi._NoMPIContext()
    assert ctx.__getstate__() == {
        "algsize": 1, "algrank": 0, "solsize": 1, "solrank": 0,
    }


# --------------------------------------------------------------------------- #
#  MPI context: accessors and setup() validation
# --------------------------------------------------------------------------- #

@needs_mpi
def test_global_accessors_work_before_setup():
    from mpi4py import MPI
    ctx = mpi._MPIContext()
    assert ctx.size() == MPI.COMM_WORLD.size
    assert ctx.rank() == MPI.COMM_WORLD.rank
    assert ctx.enabled() is True


@needs_mpi
def test_layer_accessors_raise_before_setup():
    ctx = mpi._MPIContext()
    for accessor in (ctx.solsize, ctx.solrank, ctx.solcomm,
                     ctx.algsize, ctx.algrank, ctx.algcomm):
        with pytest.raises(RuntimeError):
            accessor()


@needs_mpi
@pytest.mark.parametrize("kwargs", [{"nalg": 0}, {"nsolve": 0}, {"nalg": -1}])
def test_setup_rejects_nonpositive(kwargs):
    # Validation happens before any collective, so raising here is safe.
    ctx = mpi._MPIContext()
    with pytest.raises(ValueError):
        ctx.setup(**kwargs)


@needs_mpi
def test_setup_rejects_inconsistent_product():
    total = mpi.size()
    ctx = mpi._MPIContext()
    with pytest.raises(ValueError):
        ctx.setup(nalg=total + 1, nsolve=total + 1)


@needs_mpi
def test_setup_rejects_nondivisible():
    total = mpi.size()
    ctx = mpi._MPIContext()
    with pytest.raises(ValueError):
        ctx.setup(nalg=total + 1)  # total is never divisible by total+1


# --------------------------------------------------------------------------- #
#  MPI context: partitioning
# --------------------------------------------------------------------------- #

@needs_mpi
def test_default_setup_assigns_all_to_algorithm_layer():
    from mpi4py import MPI
    total = MPI.COMM_WORLD.size
    ctx = mpi._MPIContext()
    ctx.setup()
    assert ctx.solsize() == 1
    assert ctx.solrank() == 0
    assert ctx.algsize() == total
    assert ctx.algrank() == MPI.COMM_WORLD.rank
    assert ctx.run_on_algorithm() is True
    assert ctx.algcomm() is not None


@needs_mpi
def test_setup_twice_raises():
    ctx = mpi._MPIContext()
    ctx.setup()
    with pytest.raises(RuntimeError):
        ctx.setup()


@needs_mpi
def test_solver_layer_split():
    from mpi4py import MPI
    total = MPI.COMM_WORLD.size
    if total % 2 != 0:
        pytest.skip("needs an even number of ranks")

    ctx = mpi._MPIContext()
    ctx.setup(nsolve=2)

    assert ctx.solsize() == 2
    assert ctx.algsize() == total // 2
    # exactly one process per solver group runs the algorithm layer
    assert ctx.run_on_algorithm() == (ctx.solrank() == 0)
    # algrank is broadcast to the solver workers, and is always a valid index
    assert 0 <= ctx.algrank() < ctx.algsize()
    # only the solver-group leaders own an algorithm communicator
    if ctx.solrank() == 0:
        assert ctx.algcomm() is not None
    else:
        assert ctx.algcomm() is None


# --------------------------------------------------------------------------- #
#  Checkpoint mixin (validates a saved snapshot against the live singleton)
# --------------------------------------------------------------------------- #

class _Dummy(mpi._CheckpointMixin):
    def __getstate__(self):
        return mpi._ctx.__getstate__()


def test_checkpoint_matching_state_restores():
    current = mpi._ctx.__getstate__()
    _Dummy().__setstate__(dict(current))  # identical config -> no error


def test_checkpoint_mismatch_raises():
    current = mpi._ctx.__getstate__()
    bad = dict(current)
    bad["algsize"] = current["algsize"] + 100
    with pytest.raises(ValueError):
        _Dummy().__setstate__(bad)
