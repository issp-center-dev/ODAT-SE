import os
import sys
from pathlib import Path

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import numpy as np
import pytest

from odatse.util.separateT import separateT


def test_separateT_rejects_duplicate_temperatures(tmp_path):
    """Duplicate temperatures cannot be separated (a result line records only
    the temperature, not the replica), so they must be rejected loudly instead
    of silently dropping data."""
    Ts = np.array([1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="distinct temperature"):
        separateT(Ts, nwalkers=3, output_dir=str(tmp_path), comm=None, use_beta=False)


def test_separateT_splits_by_temperature(tmp_path):
    """Happy path: distinct temperatures are routed to separate files, one per
    temperature index, preserving each replica's rows."""
    Ts = np.array([1.0, 2.0])
    proc = tmp_path / "0"
    proc.mkdir()
    (proc / "result.txt").write_text(
        "# step walker T fx x1 x2\n"
        "0 0 1.0 0.5 0.1 0.2\n"
        "0 1 2.0 0.7 0.3 0.4\n"
        "1 0 1.0 0.6 0.11 0.21\n"
        "1 1 2.0 0.8 0.31 0.41\n"
    )

    separateT(Ts, nwalkers=2, output_dir=str(tmp_path), comm=None, use_beta=False)

    t0 = (tmp_path / "result_T0.txt").read_text().splitlines()
    t1 = (tmp_path / "result_T1.txt").read_text().splitlines()

    # headers name the temperature
    assert t0[0] == "# T = 1.0"
    assert t1[0] == "# T = 2.0"

    # T=1.0 belongs to global walker 0; T=2.0 to global walker 1
    assert t0[1:] == ["0 0 0.5 0.1 0.2", "1 0 0.6 0.11 0.21"]
    assert t1[1:] == ["0 1 0.7 0.3 0.4", "1 1 0.8 0.31 0.41"]
