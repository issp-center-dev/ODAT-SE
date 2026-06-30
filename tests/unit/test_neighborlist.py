import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

import numpy as np
import pytest

from odatse.util.neighborlist import (
    Cells,
    make_neighbor_list_cell,
    make_neighbor_list_naive,
)


# A deliberately non-cubic grid: the bug in cellcoord2cellindex only shows up
# when the per-dimension cell counts (Ns) differ from each other.
def _make_cells():
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([4.0, 6.0, 10.0])
    cells = Cells(mins, maxs, cellsize=1.0)
    # sanity: the grid is non-cubic so the strides really differ
    assert len(set(cells.Ns.tolist())) > 1
    return cells


def test_cellcoord2cellindex_matches_row_major_reference():
    """cellcoord2cellindex must equal numpy's C-order ravel_multi_index."""
    cells = _make_cells()
    Ns = cells.Ns
    for ns in np.ndindex(*Ns):
        ns = np.array(ns, dtype=np.int64)
        expected = int(np.ravel_multi_index(ns, Ns, order="C"))
        assert cells.cellcoord2cellindex(ns) == expected


def test_cellindex_roundtrip_is_bijective():
    """index -> coord -> index must be the identity over the whole range,
    and every index must stay within [0, ncell)."""
    cells = _make_cells()
    seen = set()
    for index in range(cells.ncell):
        coord = cells.cellindex2cellcoord(index)
        back = cells.cellcoord2cellindex(coord)
        assert back == index
        assert 0 <= back < cells.ncell
        seen.add(back)
    # bijective: no collisions, full coverage
    assert len(seen) == cells.ncell


def test_neighborcells_indices_in_range():
    """neighborcells round-trips index->coord->neighbor indices; with the old
    bug these went out of range and raised IndexError downstream."""
    cells = _make_cells()
    for index in range(cells.ncell):
        for nb in cells.neighborcells(index):
            assert 0 <= nb < cells.ncell


def test_cell_method_matches_naive_on_noncubic_grid():
    """The default (cell-based) neighbor list must agree with the brute-force
    all-pairs version on a non-cubic bounding box."""
    rng = np.random.RandomState(12345)
    # points spread over a non-cubic box so Ns differs per dimension
    X = rng.uniform(low=[0, 0, 0], high=[4, 6, 10], size=(200, 3))
    radius = 1.3

    nl_cell = make_neighbor_list_cell(X, radius, allow_selfloop=False, show_progress=False)
    nl_naive = make_neighbor_list_naive(X, radius, allow_selfloop=False, show_progress=False)

    assert len(nl_cell) == len(nl_naive)
    for a, b in zip(nl_cell, nl_naive):
        assert sorted(a) == sorted(b)
