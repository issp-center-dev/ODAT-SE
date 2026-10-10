"""odatse.util.io: one process reads, the others receive, and a failed read
raises the same LoadError on every process instead of leaving the others
waiting in the broadcast. Serially (and with the non-MPI stub) the functions
are plain reads."""

import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import json

import numpy as np
import pytest

import odatse.mpi as mpi
import odatse.util.io as oio
from odatse.exception import InputError, LoadError


def _parallel() -> bool:
    return mpi.enabled() and mpi.size() > 1


def test_load_hands_the_root_result_to_every_rank():
    # the loader runs on rank 0 only; every rank must end up with its result
    data = oio.load(lambda: {"from": mpi.rank(), "values": [1, 2, 3]})
    assert data == {"from": 0, "values": [1, 2, 3]}


def test_root_failure_raises_load_error_on_every_rank():
    def loader():
        raise FileNotFoundError("boom.dat is not here")

    with pytest.raises(LoadError, match="cannot read reference data") as excinfo:
        oio.load(loader, what="reference data")
    err = excinfo.value
    assert isinstance(err, InputError)        # never turned into NaN by ignore_error
    assert "FileNotFoundError: boom.dat is not here" in str(err)
    if mpi.rank() == 0:
        assert isinstance(err.__cause__, FileNotFoundError)
    else:
        assert err.__cause__ is None
        assert "failed on rank 0" in str(err)


def test_unpicklable_result_is_reported_not_hung():
    """The data is pickled on the root before the broadcast, inside the try:
    an object that cannot be pickled is a LoadError everywhere, not a failure
    of the root alone in the broadcast after it announced success. Serially
    nothing is pickled and the object comes back as it is."""
    loader = lambda: (lambda x: x)   # a function: not picklable
    if _parallel():
        with pytest.raises(LoadError, match="cannot read"):
            oio.load(loader)
    else:
        assert callable(oio.load(loader))


def test_distribute_false_keeps_the_data_on_the_root():
    data = oio.load(lambda: [10, 20], distribute=False)
    if mpi.rank() == 0:
        assert data == [10, 20]
    else:
        assert data is None


def test_distribute_false_still_raises_everywhere():
    def loader():
        raise ValueError("bad")

    with pytest.raises(LoadError, match="ValueError: bad"):
        oio.load(loader, distribute=False)


def test_invalid_scope():
    with pytest.raises(ValueError, match="scope"):
        oio.load(lambda: 1, scope="everyone")


def test_solver_scope_in_the_unit_tests_is_a_plain_read():
    """With nsolve = 1 the solver group is this process alone."""
    assert mpi.solsize() == 1
    assert oio.load(lambda: 7, scope="solver") == 7


def test_algorithm_scope_on_an_algorithm_rank():
    assert mpi.run_on_algorithm()
    assert oio.load(lambda: "mesh", scope="algorithm") == "mesh"


# --- the convenience functions (each rank writes its own copy of the file in
#     its own working directory; only the root reads it) -----------------------

def test_read_text_and_bytes():
    with open("note.txt", "w", encoding="utf-8") as fp:
        fp.write("héllo\n")
    assert oio.read_text("note.txt") == "héllo\n"
    assert oio.read_bytes("note.txt") == "héllo\n".encode("utf-8")


def test_load_toml():
    with open("in.toml", "w") as fp:
        fp.write("[base]\ndimension = 3\n")
    assert oio.load_toml("in.toml") == {"base": {"dimension": 3}}


def test_loadtxt_passes_keyword_arguments():
    with open("ref.txt", "w") as fp:
        fp.write("# header\n1 2\n")
    data = oio.loadtxt("ref.txt", ndmin=2)
    assert data.shape == (1, 2)
    assert data.tolist() == [[1.0, 2.0]]


def test_load_json():
    with open("cfg.json", "w") as fp:
        json.dump({"a": [1, 2]}, fp)
    assert oio.load_json("cfg.json") == {"a": [1, 2]}


def test_missing_file_names_the_file():
    with pytest.raises(LoadError, match="cannot read file does_not_exist.dat") as excinfo:
        oio.read_text("does_not_exist.dat")
    assert "FileNotFoundError" in str(excinfo.value)
