import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse.mpi as mpi


@pytest.fixture(scope="session", autouse=True)
def mpi_setup():
    """Partition the MPI communicator once for the whole unit-test session.

    odatse.mpi accessors for the algorithm/solver layers (algcomm, solsize,
    ...) require setup() to have been called exactly once. Doing it here in a
    session-scoped, autouse fixture lets every unit test run both serially and
    under ``mpirun -n N`` without each test having to call setup() itself.

    With MPI disabled (ODATSE_NOMPI=1) setup() is a harmless no-op.
    """
    mpi.setup()
    yield


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Run each test in its own temporary working directory.

    Several unit tests write output files (sample.txt, combined.txt,
    SimplexData.txt, ...) into the current directory. Under ``mpirun -n N``
    every rank is a separate pytest process sharing the same starting cwd, so
    without isolation the ranks race on the same filenames. ``tmp_path`` is
    unique per test *and* per process, eliminating the collision.
    """
    monkeypatch.chdir(tmp_path)
    yield
