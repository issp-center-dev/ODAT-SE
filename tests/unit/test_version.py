# sys.path is handled by conftest.py
from odatse.util.version import parse_version


def test_basic():
    assert parse_version("1.11.4") == (1, 11, 4)
    assert parse_version("2.0.0") == (2, 0, 0)


def test_prerelease_suffix_ignored():
    assert parse_version("1.16.0rc1") == (1, 16, 0)
    assert parse_version("1.2rc1") == (1, 2, 0)
    assert parse_version("1.26.0.dev0") == (1, 26, 0)


def test_short_version_padded():
    assert parse_version("1") == (1, 0, 0)
    assert parse_version("1.2") == (1, 2, 0)


def test_numeric_not_lexicographic():
    # a plain string compare gives "1.10.0" < "1.2.0", which is wrong
    assert parse_version("1.10.0") > parse_version("1.2.0")


def test_scipy_gate_semantics():
    # the min_search callback gate: modern branch for scipy >= 1.11
    assert parse_version("1.11.0") >= (1, 11, 0)
    assert parse_version("1.10.1") < (1, 11, 0)
    # the old gate (major >= 1 and minor >= 11) was False for 2.0
    assert parse_version("2.0.0") >= (1, 11, 0)


def test_non_numeric_component():
    assert parse_version("abc") == (0, 0, 0)
    assert parse_version("1.x.4") == (1, 0, 0)
