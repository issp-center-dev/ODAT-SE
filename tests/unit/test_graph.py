import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.util.graph import is_connected, is_bidirectional


def test_empty_graph_does_not_crash():
    """is_connected used to index visited[0] on a zero-length array."""
    assert is_connected([]) is True


def test_single_node():
    assert is_connected([[]]) is True


def test_connected_chain():
    assert is_connected([[1], [0, 2], [1]]) is True


def test_disconnected_graph():
    # node 2 is isolated
    assert is_connected([[1], [0], []]) is False


def test_is_bidirectional():
    assert is_bidirectional([[1], [0]]) is True
    assert is_bidirectional([[1], []]) is False
