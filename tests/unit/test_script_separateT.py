"""Tests for the standalone post-processing command odatse.scripts.separateT
(not the library odatse.util.separateT)."""

import os
import resource
import sys

import pytest

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)


def _load_script():
    import odatse.scripts.separateT
    return odatse.scripts.separateT


def _write_result(path, temperatures, steps):
    """Write a result.txt with interleaved temperatures (as in exchange MC):
    every step has one line per temperature."""
    with open(path, "w") as fp:
        fp.write("# step walker T fx x1\n")
        for step in range(steps):
            for iw, t in enumerate(temperatures):
                fp.write(f"{step} {iw} {t} 0.5 0.1\n")


@pytest.mark.parametrize("buffer_limit", [200_000, 5])
def test_separate_routes_interleaved_temperatures(tmp_path, monkeypatch, buffer_limit):
    """buffer_limit=5 forces several mid-stream flushes (append path);
    the large value keeps everything in one final flush."""
    monkeypatch.chdir(tmp_path)
    mod = _load_script()

    temperatures = ["1.0", "2.5", "0.125"]
    _write_result("result.txt", temperatures, steps=4)

    mod.do_separate("result.txt", buffer_limit=buffer_limit)

    for index, t in enumerate(temperatures):
        lines = open(f"result_T{index}.txt").read().splitlines()
        # original header, then the marker line, then only this T's data
        assert lines[0] == "# step walker T fx x1"
        assert lines[1] == f"# T (or beta) = {t}"
        data = [l for l in lines if not l.startswith("#")]
        assert len(data) == 4
        assert all(l.split()[2] == t for l in data)


def test_separate_many_temperatures_under_low_fd_limit(tmp_path, monkeypatch):
    """Regression test for issue #61: one file descriptor was kept open per
    distinct temperature, so more temperatures than the soft RLIMIT_NOFILE
    (256 by default on macOS) aborted with 'Too many open files'."""
    monkeypatch.chdir(tmp_path)
    mod = _load_script()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, soft), hard))
    try:
        ntemp = 300
        temperatures = [f"{0.1 * (i + 1):.4f}" for i in range(ntemp)]
        _write_result("result.txt", temperatures, steps=2)

        mod.do_separate("result.txt")
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    for index in (0, ntemp // 2, ntemp - 1):
        lines = open(f"result_T{index}.txt").read().splitlines()
        data = [l for l in lines if not l.startswith("#")]
        assert len(data) == 2
        assert all(l.split()[2] == temperatures[index] for l in data)
