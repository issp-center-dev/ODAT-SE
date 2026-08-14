"""Tests for the reference comparator used by tests/*/do.sh.

``diff_res_mc.py`` compares ``best_result.txt`` against ``ref.txt`` with an
absolute tolerance.  A tolerance-based comparison silently accepts a NaN
(``abs(nan - x) > tol`` is False for every ``x``), so the parser must reject
non-finite values, duplicate keys, and malformed lines before comparing.
"""

import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "test_utilities", "diff_res_mc.py"
)

REF = "fx = 1.0\nx1 = 2.0\n"


def _run(tmp_path, res_text, ref_text=REF):
    resfile = tmp_path / "res.txt"
    reffile = tmp_path / "ref.txt"
    resfile.write_text(res_text)
    reffile.write_text(ref_text)
    return subprocess.run(
        [sys.executable, SCRIPT, str(resfile), str(reffile)],
        capture_output=True, text=True,
    )


def test_identical_files_pass(tmp_path):
    assert _run(tmp_path, REF).returncode == 0


def test_difference_within_tolerance_passes(tmp_path):
    assert _run(tmp_path, "fx = 1.0000005\nx1 = 2.0\n").returncode == 0


def test_difference_above_tolerance_fails(tmp_path):
    proc = _run(tmp_path, "fx = 1.001\nx1 = 2.0\n")
    assert proc.returncode != 0
    assert "differs" in proc.stdout


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_result_is_rejected(tmp_path, value):
    """Regression: nan slipped through the tolerance check and reported PASS."""
    proc = _run(tmp_path, f"fx = {value}\nx1 = 2.0\n")
    assert proc.returncode != 0
    assert "Non-finite" in proc.stdout


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_reference_is_rejected(tmp_path, value):
    proc = _run(tmp_path, REF, ref_text=f"fx = {value}\nx1 = 2.0\n")
    assert proc.returncode != 0
    assert "Non-finite" in proc.stdout


def test_duplicate_key_in_result_is_rejected(tmp_path):
    proc = _run(tmp_path, "fx = 1.0\nfx = 5.0\n")
    assert proc.returncode != 0
    assert "Duplicate key" in proc.stdout


def test_duplicate_key_in_reference_is_rejected(tmp_path):
    proc = _run(tmp_path, REF, ref_text="fx = 1.0\nfx = 5.0\n")
    assert proc.returncode != 0
    assert "Duplicate key" in proc.stdout


@pytest.mark.parametrize("line", ["fx != 1.0", "fx garbage 1.0", "fx 1.0"])
def test_malformed_line_is_rejected(tmp_path, line):
    proc = _run(tmp_path, f"{line}\nx1 = 2.0\n")
    assert proc.returncode != 0
    assert "Invalid line" in proc.stdout


def test_missing_key_is_rejected(tmp_path):
    proc = _run(tmp_path, "fx = 1.0\nx2 = 2.0\n")
    assert proc.returncode != 0


def test_comments_and_blank_lines_are_ignored(tmp_path):
    assert _run(tmp_path, "# a comment\n\nfx = 1.0\nx1 = 2.0\n").returncode == 0
