# sys.path and odatse.mpi.setup() are handled by conftest.py
"""Unit tests for how odatse.initialize() drives odatse.mpi.setup().

A host program that embeds ODAT-SE may have called setup(comm=...) on its
own communicator before initialize(); initialize() must then keep that
configuration instead of partitioning MPI.COMM_WORLD again. The MPI layer is
replaced by recording stubs so that the decision logic can be checked on a
single process (the partitioning itself is covered by test_mpi.py).
"""
import pytest

import odatse
import odatse.mpi as mpi


class _Sentinel:
    """Stands in for a communicator object; only identity matters here."""


@pytest.fixture
def mpi_stub(monkeypatch):
    """Replace setup()/ready()/comm() by stubs and Info.from_file() by a
    dummy; returns the list of setup() calls as (nalg, nsolve, comm)."""
    calls = []
    state = {"ready": False, "comm": _Sentinel()}

    def setup(*, nalg=None, nsolve=None, comm=None):
        calls.append((nalg, nsolve, comm))
        state["ready"] = True

    monkeypatch.setattr(mpi, "setup", setup)
    monkeypatch.setattr(mpi, "ready", lambda: state["ready"])
    monkeypatch.setattr(mpi, "comm", lambda: state["comm"])
    dummy = {"base": {"dimension": 1}, "algorithm": {"name": "mapper"}, "solver": {"name": "analytical"}}
    monkeypatch.setattr(odatse.Info, "from_file", classmethod(lambda cls, f: cls(dummy)))
    state["calls"] = calls
    return state


def test_initialize_partitions_comm_world_when_not_ready(mpi_stub):
    odatse.initialize(["input.toml", "--nalg", "2", "--nsolve", "3"])
    assert mpi_stub["calls"] == [(2, 3, None)]


def test_initialize_without_layout_partitions_with_defaults(mpi_stub):
    odatse.initialize(["input.toml"])
    assert mpi_stub["calls"] == [(None, None, None)]


def test_initialize_keeps_existing_setup_when_no_layout_requested(mpi_stub):
    """setup(comm=sub) by the host, then initialize() without --nalg/--nsolve:
    the existing partition is kept and setup() is not called again (it would
    otherwise try to partition MPI.COMM_WORLD and raise)."""
    mpi_stub["ready"] = True
    odatse.initialize(["input.toml"])
    assert mpi_stub["calls"] == []


def test_initialize_checks_requested_layout_against_existing_setup(mpi_stub):
    """With --nalg/--nsolve after an earlier setup(), the request is passed
    to setup(), which refers to the current communicator and accepts the
    same layout."""
    mpi_stub["ready"] = True
    odatse.initialize(["input.toml", "--nsolve", "2"])
    assert mpi_stub["calls"] == [(None, 2, None)]


def test_initialize_reports_layout_conflict_as_input_error(mpi_stub, monkeypatch):
    """A conflicting --nalg/--nsolve is an input error, so that odatse.main()
    reports it on one line and exits with status 1 instead of dumping a raw
    RuntimeError traceback on every rank."""
    from odatse.exception import InputError

    def conflicting_setup(*, nalg=None, nsolve=None, comm=None):
        raise RuntimeError("setup() has already been called with a different layout")
    mpi_stub["ready"] = True
    monkeypatch.setattr(mpi, "setup", conflicting_setup)
    with pytest.raises(InputError, match="different layout") as excinfo:
        odatse.initialize(["input.toml", "--nsolve", "4"])
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_initialize_run_mode(mpi_stub):
    _, run_mode = odatse.initialize(["input.toml", "--resume", "--reset_rand"])
    assert run_mode == "resume-resetrand"
