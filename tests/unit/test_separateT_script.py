import os
import sys
import glob

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '../../script')
sys.path.insert(0, SCRIPT_PATH)

import separateT


def _write_result(path, ntemp, nsweep):
    """A result.txt with `ntemp` distinct temperatures, each appearing once per
    sweep so the temperatures are interleaved (as in exchange MC)."""
    with open(path, "w") as fp:
        fp.write("# step walker T value\n")
        for sweep in range(nsweep):
            for t in range(ntemp):
                # columns: step walker T value ; temperature is column index 2
                fp.write(f"{sweep} 0 {float(t)} {sweep * 1000 + t}\n")


def test_one_file_per_temperature_even_when_interleaved(tmp_path):
    """Each distinct temperature gets exactly one file collecting all of its
    (non-contiguous) lines, with the header preserved."""
    src = str(tmp_path / "result.txt")
    _write_result(src, ntemp=5, nsweep=3)

    separateT.do_separate(src)

    outs = glob.glob(str(tmp_path / "result_T*.txt"))
    assert len(outs) == 5

    for idx in range(5):
        with open(tmp_path / f"result_T{idx}.txt") as f:
            lines = f.readlines()
        data = [l for l in lines if not l.startswith("#")]
        # header line preserved
        assert lines[0] == "# step walker T value\n"
        # all three sweeps routed here, and to the correct temperature
        assert len(data) == 3
        assert all(float(l.split()[2]) == float(idx) for l in data)


def test_does_not_leak_file_descriptors(tmp_path, monkeypatch):
    """More distinct temperatures than the open-file cap must not keep them all
    open at once (regression: the previous version held one FD per temperature
    for the whole run and exhausted the FD limit for large numT)."""
    monkeypatch.setattr(separateT, "MAX_OPEN_FILES", 8)

    real_open = open
    live = set()
    peak = [0]

    class _Tracked:
        def __init__(self, f):
            self._f = f
        def write(self, *a, **k):
            return self._f.write(*a, **k)
        def writelines(self, *a, **k):
            return self._f.writelines(*a, **k)
        def close(self):
            live.discard(id(self))
            return self._f.close()

    def counting_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        # only track the per-temperature output files
        if "result_T" in str(path):
            t = _Tracked(f)
            live.add(id(t))
            peak[0] = max(peak[0], len(live))
            return t
        return f

    monkeypatch.setattr("builtins.open", counting_open)

    src = str(tmp_path / "result.txt")
    _write_result(src, ntemp=50, nsweep=3)

    separateT.do_separate(src)

    # never more than the cap open simultaneously, and all closed at the end
    assert peak[0] <= 8
    assert len(live) == 0
    # correctness preserved despite eviction/reopen
    assert len(glob.glob(str(tmp_path / "result_T*.txt"))) == 50
