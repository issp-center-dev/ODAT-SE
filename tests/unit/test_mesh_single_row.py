import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

from pathlib import Path

import pytest

import odatse.mpi as mpi
import odatse.util.neighborlist as neighborlist
from odatse.algorithm.mapper_mpi import Algorithm
from odatse.domain.meshgrid import MeshGrid


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


def test_mapper_reads_single_row_as_one_point():
    """Same one-row case through the mapper's own mesh file reader."""
    alg = Algorithm.__new__(Algorithm)
    alg.root_dir = Path(".")

    it = alg._read_mesh_file({"mesh_path": _write_mesh("0 0.5 -0.25\n")})

    assert it._total_points == 1
    # the point lands on exactly one rank; the others hold nothing
    points = [list(p) for p in it._data]
    assert points in ([], [[0, 0.5, -0.25]])
    if mpi.algsize() == 1:
        assert points == [[0, 0.5, -0.25]]


def test_neighborlist_main_accepts_single_row(monkeypatch):
    """The neighbor-list CLI indexed X.shape[1] on the 1-D array loadtxt
    returns for a one-row mesh file."""
    if mpi.algsize() != 1:
        pytest.skip("a single point cannot be distributed over several ranks")

    _write_mesh("0 0.5 -0.25\n")
    # the session fixture has already partitioned the communicator
    monkeypatch.setattr(neighborlist.mpi, "setup", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["odatse_neighborlist", "mesh.txt", "-o", "nn.txt", "-q"])

    neighborlist.main()

    assert neighborlist.load_neighbor_list("nn.txt") == [[]]
