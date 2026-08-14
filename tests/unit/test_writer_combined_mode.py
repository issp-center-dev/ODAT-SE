import os
import sys
from pathlib import Path

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.util.data_writer import BaseWriter, DataWriter


@pytest.fixture
def fresh_combined(tmp_path, monkeypatch):
    """Reset the shared combined-file class state to an unconfigured default
    pointing at a temp file (monkeypatch restores it afterwards)."""
    monkeypatch.setattr(BaseWriter, "_fp", None)
    monkeypatch.setattr(BaseWriter, "_fp_count", 0)
    monkeypatch.setattr(BaseWriter, "_combined_filename", str(tmp_path / "combined.txt"))
    monkeypatch.setattr(BaseWriter, "_combined_filemode", None)  # not configured
    return Path(BaseWriter._combined_filename)


def test_combined_append_mode_does_not_truncate(fresh_combined):
    """With no explicit basicConfig mode, a writer opened with mode='a' must
    append to the combined file rather than truncating it (the resume case)."""
    fresh_combined.write_text("PRE\n")

    w = DataWriter("d.txt", mode="a", item_list=["A"], combined=True)
    w.write(1)
    w.close()

    content = fresh_combined.read_text()
    assert content.startswith("PRE\n")       # existing content preserved
    assert "<d.txt> 1" in content


def test_combined_write_mode_truncates(fresh_combined):
    fresh_combined.write_text("PRE\n")

    w = DataWriter("d.txt", mode="w", item_list=["A"], combined=True)
    w.write(1)
    w.close()

    content = fresh_combined.read_text()
    assert "PRE" not in content               # truncated
    assert "<d.txt>" in content


def test_basicconfig_mode_overrides_instance_mode(fresh_combined, monkeypatch):
    """An explicit basicConfig combined_mode wins over the writer's mode."""
    monkeypatch.setattr(BaseWriter, "_combined_filemode", "a")  # explicit
    fresh_combined.write_text("PRE\n")

    w = DataWriter("d.txt", mode="w", item_list=["A"], combined=True)
    w.write(1)
    w.close()

    content = fresh_combined.read_text()
    assert content.startswith("PRE\n")        # basicConfig "a" wins over "w"
