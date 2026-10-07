"""Tests for the standalone post-processing command
odatse.scripts.plt_model_evidence."""

import os
import sys

import numpy as np
import pytest

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

pytest.importorskip("matplotlib").use("Agg")

import odatse.scripts.plt_model_evidence as mod


def test_load_data_single_row(tmp_path):
    """numpy.loadtxt returns a 1-D array for a one-row file, which used to be
    rejected as having too few columns."""
    fx = tmp_path / "fx.txt"
    fx.write_text("# comment line\n1.0 0.1 0.2 0.3 -2.5\n")

    beta, logz = mod.load_data(str(fx))

    assert beta.tolist() == [1.0]
    assert logz.tolist() == [-2.5]


def test_load_data_multiple_rows(tmp_path):
    fx = tmp_path / "fx.txt"
    fx.write_text("1.0 0.1 0.2 0.3 -2.5\n2.0 0.1 0.2 0.3 -3.5\n")

    beta, logz = mod.load_data(str(fx))

    assert beta.tolist() == [1.0, 2.0]
    assert logz.tolist() == [-2.5, -3.5]


def test_load_data_rejects_too_few_columns(tmp_path):
    fx = tmp_path / "fx.txt"
    fx.write_text("1.0\n2.0\n3.0\n")

    # the message reports columns and rows of the file, not the unpacked shape
    with pytest.raises(ValueError, match=r"at least 5 columns.*got 1 column\(s\) in 3 row\(s\)"):
        mod.load_data(str(fx))


def test_load_data_rejects_empty_file(tmp_path):
    fx = tmp_path / "fx.txt"
    fx.write_text("# only a header\n")

    with pytest.raises(ValueError, match="no data rows"):
        mod.load_data(str(fx))


def test_auto_range_single_point():
    """np.gradient cannot take a single point; the range must still enclose it."""
    beta_min, beta_max, y_min, y_max = mod.auto_range(np.array([2.0]), np.array([-8.0]))

    assert beta_min < 2.0 < beta_max
    assert y_min < -8.0 < y_max


@pytest.mark.parametrize("auto_focus", [False, True])
def test_main_single_row(monkeypatch, auto_focus):
    with open("fx.txt", "w") as fp:
        fp.write("1.0 0.1 0.2 0.3 -2.5\n")
    argv = ["odatse_plt_model_evidence", "-n", "10", "fx.txt"]
    if auto_focus:
        argv.append("--auto-focus")
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    rows = np.loadtxt("model_evidence.txt", ndmin=2)
    assert rows.shape == (1, 3)
    assert rows[0, 1] == 1.0
    assert os.path.exists("model_evidence.png")
