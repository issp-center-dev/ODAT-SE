import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.util.toml import _parse_version


@pytest.mark.parametrize("version,expected", [
    ("1.1.0", (1, 1, 0)),
    ("1.2.0", (1, 2, 0)),
    ("1.10.0", (1, 10, 0)),
    ("2.0.1", (2, 0, 1)),
    ("1.2", (1, 2, 0)),       # padded
    ("0.9", (0, 9, 0)),
    ("1.2.0rc1", (1, 2, 0)),  # non-numeric suffix ignored
])
def test_parse_version(version, expected):
    assert _parse_version(version) == expected


def test_version_threshold_numeric_not_lexicographic():
    """The 1.2.0 API threshold must compare numerically. The old string
    comparison wrongly classified 1.10.0 as < 1.2.0."""
    assert _parse_version("1.1.0") < (1, 2, 0)    # old API
    assert not _parse_version("1.2.0") < (1, 2, 0)  # new API
    assert not _parse_version("1.10.0") < (1, 2, 0)  # new API (was the bug)
    assert not _parse_version("2.0.1") < (1, 2, 0)   # new API
