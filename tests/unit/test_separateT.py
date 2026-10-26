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
    tvals = np.array([1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="distinct temperature"):
        separateT(tvals, nwalkers=3, output_dir=str(tmp_path), comm=None, use_beta=False)


@pytest.mark.parametrize("use_beta", [False, True], ids=["T", "beta"])
def test_separateT_splits_by_recorded_value(tmp_path, use_beta):
    """Happy path: distinct temperatures are routed to separate files, one per
    temperature index, preserving each replica's rows.  The third column of
    result.txt records T or beta according to how the input gave them;
    separateT matches lines by that value and labels the headers after
    use_beta."""
    tvals = np.array([1.0, 2.0])
    label = "beta" if use_beta else "T"
    proc = tmp_path / "0"
    proc.mkdir()
    (proc / "result.txt").write_text(
        f"# step walker {label} fx x1 x2\n"
        "0 0 1.0 0.5 0.1 0.2\n"
        "0 1 2.0 0.7 0.3 0.4\n"
        "1 0 1.0 0.6 0.11 0.21\n"
        "1 1 2.0 0.8 0.31 0.41\n"
    )

    separateT(tvals, nwalkers=2, output_dir=str(tmp_path), comm=None, use_beta=use_beta)

    t0 = (tmp_path / "result_T0.txt").read_text().splitlines()
    t1 = (tmp_path / "result_T1.txt").read_text().splitlines()

    # headers name the recorded value with its label
    assert t0[0] == f"# {label} = 1.0"
    assert t1[0] == f"# {label} = 2.0"

    # recorded 1.0 belongs to global walker 0; 2.0 to global walker 1
    assert t0[1:] == ["0 0 0.5 0.1 0.2", "1 0 0.6 0.11 0.21"]
    assert t1[1:] == ["0 1 0.7 0.3 0.4", "1 1 0.8 0.31 0.41"]
