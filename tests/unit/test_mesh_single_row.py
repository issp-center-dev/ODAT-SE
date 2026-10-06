import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

from pathlib import Path

import pytest

import odatse
import odatse.mpi as mpi
import odatse.util.neighborlist as neighborlist
from odatse.algorithm.mapper_mpi import Algorithm
from odatse.domain.meshgrid import MeshGrid, load_mesh_file
from odatse.exception import InputError


def _write_mesh(text):
    # every rank writes its own copy: the working directory is per-process
    Path("mesh.txt").write_text(text)
    return "mesh.txt"


def test_meshgrid_reads_single_row_as_one_point():
    """numpy.loadtxt returns a 1-D array for a one-row file. It used to be
    reshaped column-wise, turning ``index x1 x2`` into three coordinate-less
    points; it must stay a single point."""
    mesh = MeshGrid(param={"mesh_path": _write_mesh("0 0.5 -0.25\n")})

    assert mesh.grid == [[0, 0.5, -0.25]]


def test_meshgrid_reads_multiple_rows():
    mesh = MeshGrid(param={"mesh_path": _write_mesh("0 0.5 -0.25\n1 1.0 2.0\n")})

    assert mesh.grid == [[0, 0.5, -0.25], [1, 1.0, 2.0]]


def _mapper():
    """A mapper Algorithm with only what _read_mesh_file() needs."""
    alg = Algorithm.__new__(Algorithm)
    alg.root_dir = Path(".")
    return alg


def test_mapper_reads_single_row_as_one_point():
    """Same one-row case through the mapper's own mesh file reader."""
    alg = _mapper()

    it = alg._read_mesh_file({"mesh_path": _write_mesh("0 0.5 -0.25\n")})

    # iterate through the public interface: (index, coordinates) pairs;
    # the point lands on exactly one rank, the others hold nothing
    points = [(idx, list(x)) for idx, x in it]
    assert points in ([], [(0, [0.5, -0.25])])
    if mpi.algsize() == 1:
        assert points == [(0, [0.5, -0.25])]


def test_meshgrid_accepts_more_coordinates_than_the_algorithm_dimension():
    """The coordinate count is not tied to the algorithm dimension: a mesh
    may hold the points in the solver's coordinates (tests/transform uses
    base.dimension = 1 with a two-coordinate mesh and a 2-D solver)."""
    info = odatse.Info({
        "base": {"dimension": 1},
        "algorithm": {"name": "mapper", "param": {"mesh_path": _write_mesh("0 0.5 -0.25\n")}},
        "solver": {"name": "analytical"},
    })
    mesh = MeshGrid(info)
    if mpi.run_on_algorithm():
        assert mesh.grid == [[0, 0.5, -0.25]]


@pytest.mark.parametrize("text, message", [
    ("0\n1\n", "at least 2 columns"),            # index only
    ("# header only\n", "no data rows"),          # empty
    ("0 abc 1.0\n", "cannot read mesh file"),     # not numeric
])
def test_load_mesh_file_rejects_bad_files_on_every_rank(text, message):
    """A bad mesh file is an InputError at load time, raised identically on
    every algorithm rank (the outcome of the read is broadcast before any
    other collective), rather than an AssertionError inside Runner.submit()
    or a hang of the ranks waiting in the broadcast / scatter."""
    if not mpi.run_on_algorithm():
        return   # solver workers do not read the mesh
    with pytest.raises(InputError, match=message):
        load_mesh_file(Path("."), {"mesh_path": _write_mesh(text)})


def test_load_mesh_file_missing_file_is_input_error():
    if not mpi.run_on_algorithm():
        return
    with pytest.raises(InputError, match="not found"):
        load_mesh_file(Path("."), {"mesh_path": "does_not_exist.txt"})


def test_meshgrid_rejects_index_only_file():
    if not mpi.run_on_algorithm():
        return
    with pytest.raises(InputError, match="at least 2 columns"):
        MeshGrid(param={"mesh_path": _write_mesh("0\n1\n")})


def test_mapper_rejects_index_only_file():
    """The mapper goes through the same reader, so every algorithm rank
    raises before entering the scatter in ListIterator."""
    with pytest.raises(InputError, match="at least 2 columns"):
        _mapper()._read_mesh_file({"mesh_path": _write_mesh("0\n1\n2\n")})


def test_neighborlist_main_accepts_single_row(monkeypatch):
    """The neighbor-list CLI indexed X.shape[1] on the 1-D array loadtxt
    returns for a one-row mesh file. With several ranks the single point
    lands on rank 0 and the other ranks hold an empty slice; every rank
    writes the gathered list to its own working directory."""
    _write_mesh("0 0.5 -0.25\n")
    # the session fixture has already partitioned the communicator
    monkeypatch.setattr(neighborlist.mpi, "setup", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["odatse_neighborlist", "mesh.txt", "-o", "nn.txt", "-q"])

    neighborlist.main()

    assert neighborlist.load_neighbor_list("nn.txt") == [[]]


@pytest.mark.parametrize("text", ["0\n1\n2\n", ""])
def test_neighborlist_main_rejects_file_without_coordinates(monkeypatch, capsys, text):
    """An index-only file (one column) used to give D = 0 and an all-pairs
    neighbor list; an empty file failed inside numpy. Both are now reported
    and the command exits with status 1 on every rank."""
    _write_mesh(text)
    monkeypatch.setattr(neighborlist.mpi, "setup", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["odatse_neighborlist", "mesh.txt", "-o", "nn.txt", "-q"])

    with pytest.raises(SystemExit) as excinfo:
        neighborlist.main()

    assert excinfo.value.code == 1
    assert not Path("nn.txt").exists()
    if mpi.rank() == 0:
        assert "at least one row and two columns" in capsys.readouterr().err
