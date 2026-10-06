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
    assert ctx.ready() is True   # nothing to partition: ready even before setup()
    ctx.setup(nalg=8, nsolve=4)  # arguments are accepted but ignored
    ctx.setup(nalg=2)            # ... and so is a repeated call
    # a communicator cannot be honoured without MPI: ignored with a warning
    with pytest.warns(RuntimeWarning, match="setup\\(comm=...\\) is ignored"):
        ctx.setup(comm=object())
    assert ctx.ready() is True
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


def test_module_singleton_matches_build():
    """The module singleton and public API must reflect whether MPI is active.

    With mpi4py installed, MPI is enabled automatically; this assertion only
    bites when ODATSE_NOMPI=1 is set explicitly (see the ``unit_nompi`` ctest
    entry), in which case the stub context is selected and every public
    accessor reports a single serial process.
    """
    if mpi.enabled():
        assert isinstance(mpi._ctx, mpi._MPIContext)
        assert mpi.ready() is True   # conftest has called setup()
    else:
        assert mpi.ready() is True
        assert isinstance(mpi._ctx, mpi._NoMPIContext)
        assert mpi.size() == 1
        assert mpi.rank() == 0
        assert mpi.algsize() == 1
        assert mpi.algrank() == 0
        assert mpi.solsize() == 1
        assert mpi.solrank() == 0
        assert mpi.run_on_algorithm() is True
        assert mpi.comm() is None
        assert mpi.algcomm() is None
        assert mpi.solcomm() is None


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
                     ctx.algsize, ctx.algrank, ctx.algcomm,
                     ctx.run_on_algorithm):
        with pytest.raises(RuntimeError):
            accessor()


@needs_mpi
def test_ready_reports_whether_setup_was_called():
    ctx = mpi._MPIContext()
    assert ctx.ready() is False
    ctx.setup()
    assert ctx.ready() is True


@needs_mpi
def test_failed_setup_leaves_context_not_ready():
    ctx = mpi._MPIContext()
    with pytest.raises(ValueError):
        ctx.setup(nalg=0)
    assert ctx.ready() is False


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


# --------------------------------------------------------------------------- #
#  MPI context: repeated setup()
# --------------------------------------------------------------------------- #

@needs_mpi
def test_setup_again_with_same_configuration_is_noop():
    from mpi4py import MPI
    total = MPI.COMM_WORLD.size
    ctx = mpi._MPIContext()
    ctx.setup()
    solcomm, algcomm = ctx.solcomm(), ctx.algcomm()

    # every spelling of the same effective configuration is accepted ...
    ctx.setup()
    ctx.setup(nalg=total)
    ctx.setup(nsolve=1)
    ctx.setup(nalg=total, nsolve=1)
    ctx.setup(comm=MPI.COMM_WORLD)

    # ... and nothing is re-partitioned
    assert ctx.solcomm() is solcomm
    assert ctx.algcomm() is algcomm
    assert ctx.algsize() == total


@needs_mpi
def test_setup_again_with_different_layout_raises():
    total = mpi.size()
    if total % 2 != 0:
        pytest.skip("needs an even number of ranks")
    ctx = mpi._MPIContext()
    ctx.setup()                       # nalg=total, nsolve=1
    with pytest.raises(RuntimeError, match="different layout"):
        ctx.setup(nsolve=2)
    assert ctx.solsize() == 1         # the first configuration is kept


@needs_mpi
def test_setup_again_with_different_communicator_raises():
    from mpi4py import MPI
    ctx = mpi._MPIContext()
    ctx.setup()
    dup = MPI.COMM_WORLD.Dup()        # congruent, but a different communicator
    try:
        with pytest.raises(RuntimeError, match="different communicator"):
            ctx.setup(comm=dup)
        assert ctx.comm() == MPI.COMM_WORLD
    finally:
        dup.Free()


@needs_mpi
def test_setup_again_still_validates_arguments():
    from mpi4py import MPI
    total = mpi.size()
    ctx = mpi._MPIContext()
    ctx.setup()
    with pytest.raises(ValueError):
        ctx.setup(nalg=0)
    with pytest.raises(ValueError):
        ctx.setup(nalg=total + 1)     # not a divisor: invalid, not "different"
    dup = MPI.COMM_WORLD.Dup()
    try:
        with pytest.raises(ValueError):
            ctx.setup(nalg=0, comm=dup)   # ... even combined with another communicator
    finally:
        dup.Free()


def test_module_setup_is_idempotent():
    """conftest has already called setup(); a library embedding ODAT-SE can
    call it again defensively with the same (default) configuration."""
    mpi.setup()
    assert mpi.ready() is True


# --------------------------------------------------------------------------- #
#  MPI context: external communicator
# --------------------------------------------------------------------------- #

@needs_mpi
def test_setup_with_duplicated_communicator():
    from mpi4py import MPI
    dup = MPI.COMM_WORLD.Dup()
    try:
        ctx = mpi._MPIContext()
        ctx.setup(comm=dup)
        assert ctx.comm() == dup
        assert ctx.comm() != MPI.COMM_WORLD
        assert ctx.size() == MPI.COMM_WORLD.size
        assert ctx.rank() == MPI.COMM_WORLD.rank
        assert ctx.algsize() == MPI.COMM_WORLD.size
        assert ctx.solsize() == 1
        ctx.setup(comm=dup)           # same communicator: no-op
        ctx.setup()                   # None now means dup, not COMM_WORLD: no-op
        assert ctx.comm() == dup
        with pytest.raises(RuntimeError, match="different communicator"):
            ctx.setup(comm=MPI.COMM_WORLD)
    finally:
        dup.Free()


@needs_mpi
def test_setup_with_sub_communicator():
    """ODAT-SE can be confined to a subset of the ranks of a larger job:
    sizes, ranks and the layout all refer to the communicator passed in."""
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    sub = world.Split(color=world.rank % 2, key=world.rank)
    try:
        ctx = mpi._MPIContext()
        ctx.setup(comm=sub)
        assert ctx.comm() == sub
        assert ctx.size() == sub.size
        assert ctx.rank() == sub.rank
        assert ctx.algsize() == sub.size
        assert ctx.algrank() == sub.rank
        assert ctx.algcomm().size == sub.size
        assert ctx.run_on_algorithm() is True
        # the layout is validated against the sub-communicator, not COMM_WORLD
        with pytest.raises(ValueError):
            mpi._MPIContext().setup(nalg=sub.size + 1, comm=sub)
    finally:
        sub.Free()


@needs_mpi
def test_setup_with_sub_communicator_and_solver_groups():
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    if world.size % 4 != 0:
        pytest.skip("needs a multiple of 4 ranks")
    sub = world.Split(color=world.rank % 2, key=world.rank)
    try:
        ctx = mpi._MPIContext()
        ctx.setup(nsolve=2, comm=sub)
        assert ctx.solsize() == 2
        assert ctx.algsize() == sub.size // 2
        assert ctx.run_on_algorithm() == (ctx.solrank() == 0)
        assert 0 <= ctx.algrank() < ctx.algsize()
    finally:
        sub.Free()


@needs_mpi
def test_setup_rejects_invalid_communicator():
    from mpi4py import MPI
    ctx = mpi._MPIContext()
    with pytest.raises(TypeError):
        ctx.setup(comm="COMM_WORLD")
    group = MPI.COMM_SELF.Get_group()
    try:
        with pytest.raises(TypeError):
            ctx.setup(comm=group)       # not a communicator at all
    finally:
        group.Free()
    # both spellings of a null handle are rejected with the same error
    with pytest.raises(ValueError):
        ctx.setup(comm=MPI.COMM_NULL)
    freed = MPI.COMM_WORLD.Dup()
    freed.Free()                        # an intracommunicator handle that is now null
    with pytest.raises(ValueError):
        ctx.setup(comm=freed)
    assert ctx.ready() is False
    assert ctx.comm() == MPI.COMM_WORLD  # a failed setup() leaves comm() unchanged


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
