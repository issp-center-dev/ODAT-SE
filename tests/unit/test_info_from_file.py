import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

import odatse
from odatse import mpi
from odatse.exception import InputError


VALID_TOML = """
[base]
dimension = 2

[algorithm]
name = "minsearch"

[solver]
name = "analytical"
"""


def _all_ranks_raised(raised):
    """Collective check that every rank raised; returns True serially too."""
    if mpi.enabled() and mpi.size() > 1:
        return all(mpi.comm().allgather(raised))
    return raised


def test_from_file_loads_valid(tmp_path):
    """Happy path: rank 0 parses the file and broadcasts the data, so every
    rank ends up with the same parsed contents."""
    # Only rank 0 reads the file, but writing one per rank (each into its own
    # tmp_path) is harmless and keeps the call identical on every rank.
    path = tmp_path / "input.toml"
    path.write_text(VALID_TOML)

    info = odatse.Info.from_file(str(path))

    assert info.base["dimension"] == 2
    assert info.algorithm["name"] == "minsearch"
    assert info.solver["name"] == "analytical"


def test_from_file_missing_raises_on_all_ranks():
    """Regression for the rank-0-only load deadlock: when the load fails on
    rank 0, every rank must raise instead of the non-root ranks blocking on
    the data broadcast forever. If the bug were present, this test would hang
    under ``mpirun`` rather than fail."""
    missing = "/nonexistent/odatse_does_not_exist_abc123.toml"

    raised = False
    try:
        odatse.Info.from_file(missing)
    except Exception:
        raised = True

    assert raised
    assert _all_ranks_raised(raised)


def test_from_file_invalid_toml_raises_on_all_ranks(tmp_path):
    """A malformed TOML document on rank 0 must likewise raise everywhere."""
    path = tmp_path / "broken.toml"
    path.write_text("this is = = not valid toml [[[")

    raised = False
    try:
        odatse.Info.from_file(str(path))
    except Exception:
        raised = True

    assert raised
    assert _all_ranks_raised(raised)


def test_from_file_nonroot_error_is_input_error():
    """On the non-root ranks the failure is surfaced as an InputError that
    names the offending file. Only meaningful with more than one rank."""
    if not (mpi.enabled() and mpi.size() > 1):
        pytest.skip("needs more than one rank")

    missing = "/nonexistent/odatse_does_not_exist_def456.toml"
    with pytest.raises(Exception) as excinfo:
        odatse.Info.from_file(missing)

    if mpi.rank() != 0:
        assert isinstance(excinfo.value, InputError)
        assert missing in str(excinfo.value)
