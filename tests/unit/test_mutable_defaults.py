import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.domain.meshgrid import MeshGrid
from odatse.util.data_writer import DataWriter


def test_meshgrid_instances_do_not_share_grid():
    """grid/grid_local used to be class-level mutable lists shared by every
    instance. They must now be per-instance."""
    a = MeshGrid()
    b = MeshGrid()

    assert a.grid == [] and a.grid_local == []
    assert a.grid is not b.grid
    assert a.grid_local is not b.grid_local

    a.grid.append([1, 2, 3])
    a.grid_local.append([4, 5, 6])
    # mutating one instance must not leak into another
    assert b.grid == []
    assert b.grid_local == []


def test_datawriter_default_item_list_is_empty_header():
    """Default item_list (formerly a shared mutable []) yields an empty header
    and does not raise."""
    w = DataWriter()  # filename=None, item_list defaulted
    assert w.header == []
    w.close()


def test_datawriter_default_does_not_leak_between_instances():
    w1 = DataWriter(item_list=["A", "B"])
    w2 = DataWriter()  # uses the default
    assert [name for name, _, _ in w1.header] == ["A", "B"]
    assert w2.header == []
    w1.close()
    w2.close()
