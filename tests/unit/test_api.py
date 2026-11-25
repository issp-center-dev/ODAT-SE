import asyncio
import copy
import json
import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse import api
from odatse.util import toml

MINSEARCH = {
    "base": {"dimension": 2, "output_dir": "output"},
    "solver": {"name": "analytical", "function_name": "himmelblau"},
    "algorithm": {"name": "minsearch", "seed": 12345,
                  "param": {"min_list": [-6.0, -6.0], "max_list": [6.0, 6.0], "initial_list": [0.0, 0.0]}},
}
MAPPER = {
    "base": {"dimension": 2, "output_dir": "output"},
    "solver": {"name": "analytical", "function_name": "himmelblau"},
    "algorithm": {"name": "mapper", "seed": 12345,
                  "param": {"min_list": [-6.0, -6.0], "max_list": [6.0, 6.0], "num_list": [13, 13]}},
}


def codes(issues):
    return {i["code"] for i in issues}


def test_schema_lists_algorithms_and_functions():
    s = api.schema()
    json.dumps(s)
    assert set(s["algorithms"]) == {"minsearch", "mapper", "random_search", "exchange", "pamc", "bayes"}
    assert "himmelblau" in s["runnable_solvers"]["analytical"]["function_name"]


def test_validate_accepts_samples_and_reports_structured_issues():
    assert api.validate(MINSEARCH) == []
    bad = copy.deepcopy(MINSEARCH)
    bad["algorithm"]["param"]["max_list"] = [6.0]
    bad["base"]["dimension"] = 3
    bad["base"]["output_dir"] = "../escape"
    del bad["solver"]
    assert codes(api.validate(bad)) == {"SECTION_MISSING"}
    bad["solver"] = {"name": "analytical", "function_name": "himmelblau"}
    c = codes(api.validate(bad))
    assert {"LIST_INVALID", "UNSAFE_PATH", "DIMENSION_INVALID"} <= c
    assert all(set(i) == {"code", "message", "field", "hint"} for i in api.validate(bad))


def test_validate_algorithm_specific_rules():
    ex = copy.deepcopy(MINSEARCH)
    ex["algorithm"]["name"] = "exchange"
    c = codes(api.validate(ex))
    assert "KEY_MISSING" in c  # step_list, numsteps, numsteps_exchange
    ex["algorithm"]["param"]["step_list"] = [0.3, 0.3]
    ex["algorithm"]["exchange"] = {"numsteps": 10, "numsteps_exchange": 2, "Tmin": 1.0, "Tmax": 10.0, "bmin": 0.1}
    assert "TEMPERATURE_INVALID" in codes(api.validate(ex))
    pa = copy.deepcopy(ex)
    pa["algorithm"]["name"] = "pamc"
    pa["algorithm"]["pamc"] = {"Tmin": 1.0, "Tmax": 100.0, "Tnum": 21}
    assert "SCHEDULE_INVALID" in codes(api.validate(pa))
    pa["algorithm"]["pamc"]["numsteps_annealing"] = 100
    assert api.validate(pa) == []
    user = copy.deepcopy(MINSEARCH)
    user["solver"] = {"name": "my_program"}
    assert api.validate(user) == []
    assert codes(api.validate(user, for_run=True)) == {"SOLVER_NOT_RUNNABLE"}


def test_to_toml_round_trip(tmp_path):
    inp = copy.deepcopy(MINSEARCH)
    inp["runner"] = {"limitation": {"co_a": [[1, -1], [1, 1]], "co_b": [[0], [-1]]}}
    out = api.prepare(inp, tmp_path)
    assert out["status"] == "ok"
    assert toml.load(str(tmp_path / "input.toml")) == inp


@pytest.mark.parametrize("inp,expected", [(MINSEARCH, (3.0, 2.0)), (MAPPER, None)])
def test_run_analytical_and_parse(tmp_path, inp, expected):
    r = api.run(inp, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r
    best = r["best"]
    assert best["fx"] < 1e-4 if expected else best["fx"] >= 0
    if expected:
        assert abs(best["x"]["x1"] - expected[0]) < 1e-3 and abs(best["x"]["x2"] - expected[1]) < 1e-3
    assert r["provenance"]["versions"]["odatse"]
    assert any(f.endswith(("res.txt", "ColorMap.txt")) for f in r["files"])
    assert api.parse(tmp_path)["best"] == best


def test_run_refuses_user_solvers(tmp_path):
    inp = copy.deepcopy(MINSEARCH)
    inp["solver"] = {"name": "my_program"}
    r = api.run(inp, tmp_path)
    assert r["status"] == "failed" and codes(r["issues"]) == {"SOLVER_NOT_RUNNABLE"}
    assert not (tmp_path / "input.toml").exists()


def _call(server, name, args):
    out = asyncio.run(server.call_tool(name, args))
    payload = out[1] if isinstance(out, tuple) else out
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    if not isinstance(payload, dict):
        payload = json.loads(payload[0].text)
    return payload


def test_mcp_server_run_is_opt_in(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from odatse import mcp_server

    monkeypatch.setenv("ODATSE_MCP_WORKDIR", str(tmp_path))
    monkeypatch.delenv("ODATSE_MCP_ALLOW_RUN", raising=False)
    server = mcp_server.build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"get_schema", "versions", "validate_input", "render_input_toml", "run_calculation",
                     "get_result"}
    assert _call(server, "validate_input", {"input": MINSEARCH})["valid"] is True
    assert "[algorithm.param]" in _call(server, "render_input_toml", {"input": MINSEARCH})["input_toml"]
    r = _call(server, "run_calculation", {"input": MINSEARCH})
    assert r["status"] == "failed" and r["issues"][0]["code"] == "RUN_DISABLED"
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv("ODATSE_MCP_ALLOW_RUN", "1")
    r = _call(server, "run_calculation", {"input": MINSEARCH, "timeout_s": 120})
    assert r["status"] == "ok", r
    assert (tmp_path / r["run_id"] / "input.toml").exists()
    assert _call(server, "get_result", {"run_id": r["run_id"]})["best"] == r["best"]
    assert _call(server, "get_result", {"run_id": "../etc"})["issues"][0]["code"] == "NOT_FOUND"
    for bad in (float("nan"), float("inf"), 10**400, -10**400):
        r = _call(server, "run_calculation", {"input": MINSEARCH, "timeout_s": bad})
        assert r["issues"][0]["code"] == "TIMEOUT_INVALID"


# --- regression tests for the review of the JSON API -------------------------------------------------

EXCHANGE = {
    "base": {"dimension": 2, "output_dir": "output"},
    "solver": {"name": "analytical", "function_name": "himmelblau"},
    "algorithm": {"name": "exchange", "seed": 1,
                  "param": {"min_list": [-6.0, -6.0], "max_list": [6.0, 6.0], "step_list": [0.3, 0.3]},
                  "exchange": {"numsteps": 10, "numsteps_exchange": 2, "Tmin": 1.0, "Tmax": 10.0}},
}


def test_toml_keys_cannot_inject_entries(tmp_path):
    inp = copy.deepcopy(MINSEARCH)
    del inp["base"]["output_dir"]
    inp["base"]["ignored = 0\noutput_dir"] = "/outside"
    assert "KEY_INVALID" in codes(api.validate(inp))
    assert api.prepare(inp, tmp_path)["status"] == "failed"
    assert not (tmp_path / "input.toml").exists()
    # to_toml on its own quotes such keys, so the text parses back to the same dictionary
    text = api.to_toml(inp)
    loaded = api._toml_loads(text)
    if loaded is not None:
        assert loaded == inp and "output_dir" not in loaded["base"]


def test_toml_strings_are_escaped(tmp_path):
    inp = copy.deepcopy(MINSEARCH)
    inp["solver"]["note"] = 'a"\n[base]\noutput_dir = "/x' + "b\\\t\x01\x7f"
    out = api.prepare(inp, tmp_path)
    assert out["status"] == "ok", out
    assert toml.load(str(tmp_path / "input.toml")) == inp


@pytest.mark.parametrize("path,value", [
    ("algorithm.checkpoint_file", "../../../outside"),
    ("algorithm.checkpoint_file", "/tmp/outside.pickle"),
    ("algorithm.param.neighborlist_path", "../nn.txt"),
    ("algorithm.param.neighborlist_path", "~/nn.txt"),
    ("runner.log.filename", "../log.txt"),
    ("algorithm.colormap", "C:/x.txt"),
    ("base.output_dir", ""),
])
def test_all_path_keys_are_confined(path, value):
    inp = copy.deepcopy(MAPPER)
    inp["algorithm"]["checkpoint"] = True
    d = inp
    *parents, leaf = path.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value
    assert "UNSAFE_PATH" in codes(api.validate(inp))


def test_symlink_escape_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "output").symlink_to(outside, target_is_directory=True)
    out = api.prepare(MAPPER, wd)
    assert out["status"] == "failed" and codes(out["issues"]) == {"UNSAFE_PATH"}
    assert not (wd / "input.toml").exists()
    assert api.run(MAPPER, wd)["status"] == "failed"
    # a planted input.toml symlink is not followed either
    wd2 = tmp_path / "wd2"
    wd2.mkdir()
    (wd2 / "input.toml").symlink_to(outside / "victim.toml")
    out = api.prepare(MAPPER, wd2)
    assert out["status"] == "failed" and not (outside / "victim.toml").exists()


def test_parse_refuses_paths_leaving_workdir(tmp_path):
    (tmp_path / "input.toml").write_text(api.to_toml({**MAPPER, "base": {"dimension": 2, "output_dir": "../x"}}))
    r = api.parse(tmp_path)
    assert r["status"] == "failed" and codes(r["issues"]) == {"UNSAFE_PATH"}


@pytest.mark.parametrize("mutate", [
    lambda i: i["solver"].__setitem__("function_name", []),
    lambda i: i["solver"].__setitem__("name", 3),
    lambda i: i["algorithm"].__setitem__("name", {"x": 1}),
    lambda i: i["algorithm"].__setitem__("extra", {"nested": None}),
    lambda i: i["algorithm"].__setitem__("seed", 2**70),
    lambda i: i["algorithm"].__setitem__("rows", [{"a": 1}]),
    lambda i: i["algorithm"]["param"].__setitem__("min_list", [-6.0, "a"]),
    lambda i: i["algorithm"].__setitem__("label_list", "x1"),
    lambda i: i.__setitem__("runner", []),
])
def test_malformed_values_give_structured_issues(tmp_path, mutate):
    inp = copy.deepcopy(MINSEARCH)
    mutate(inp)
    issues = api.validate(inp)
    assert issues and all(set(i) == {"code", "message", "field", "hint"} for i in issues)
    assert api.prepare(inp, tmp_path)["status"] == "failed"
    assert api.render(inp)["status"] == "failed"


def test_parse_reports_malformed_input_files(tmp_path):
    (tmp_path / "input.toml").write_text("[base\n")
    assert codes(api.parse(tmp_path)["issues"]) == {"INPUT_INVALID"}
    (tmp_path / "input.toml").write_text('[algorithm]\nname = "mapper"\n')
    assert codes(api.parse(tmp_path)["issues"]) == {"INPUT_INVALID"}


@pytest.mark.parametrize("section,changes,code", [
    ("exchange", {"numsteps_exchange": 0}, "VALUE_INVALID"),
    ("exchange", {"numsteps": 0}, "VALUE_INVALID"),
    ("exchange", {"numsteps": 2.5}, "VALUE_INVALID"),
    ("exchange", {"Tmax": None}, "TEMPERATURE_INVALID"),
    ("exchange", {"Tmin": -1.0}, "TEMPERATURE_INVALID"),
    ("exchange", {"Tmin": 20.0}, "TEMPERATURE_INVALID"),
    ("exchange", {"Tmin": None, "Tmax": None, "bmin": 0.0, "bmax": 1.0}, "TEMPERATURE_INVALID"),
    ("exchange", {"Tmin": None, "Tmax": None, "bmin": 2.0, "bmax": 1.0}, "TEMPERATURE_INVALID"),
    ("exchange", {"Tinvspace": True}, "TEMPERATURE_INVALID"),
    ("exchange", {"nreplica_per_proc": 0}, "VALUE_INVALID"),
])
def test_monte_carlo_counts_and_temperatures(section, changes, code):
    inp = copy.deepcopy(EXCHANGE)
    s = inp["algorithm"]["exchange"]
    for k, v in changes.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    assert code in codes(api.validate(inp))


def test_temperature_variants_that_odatse_accepts():
    inp = copy.deepcopy(EXCHANGE)
    inp["algorithm"]["exchange"].update({"Tmin": 1.0, "Tmax": 1.0})
    assert api.validate(inp) == []
    s = inp["algorithm"]["exchange"]
    del s["Tmin"], s["Tmax"]
    s.update({"bmin": 0.0, "bmax": 1.0, "Tlogspace": False})
    assert api.validate(inp) == []


def test_pamc_needs_its_section_and_random_search_needs_points():
    pa = copy.deepcopy(EXCHANGE)
    pa["algorithm"]["name"] = "pamc"
    del pa["algorithm"]["exchange"]
    assert "KEY_MISSING" in codes(api.validate(pa))
    pa["algorithm"]["pamc"] = {"numsteps_annealing": 10, "Tnum": 5}
    assert "TEMPERATURE_INVALID" in codes(api.validate(pa))
    pa["algorithm"]["pamc"].update({"bmin": 0.1, "bmax": 1.0})
    assert api.validate(pa) == []
    pa["algorithm"]["pamc"]["Tnum"] = -3
    assert {"VALUE_INVALID", "SCHEDULE_INVALID"} <= codes(api.validate(pa))
    rs = copy.deepcopy(MAPPER)
    rs["algorithm"]["name"] = "random_search"
    del rs["algorithm"]["param"]["num_list"]
    rs["algorithm"]["param"]["num_points"] = 0
    assert "VALUE_INVALID" in codes(api.validate(rs))
    rs["algorithm"]["param"]["num_points"] = 5
    assert api.validate(rs) == []


def test_nested_initial_list_is_accepted():
    inp = copy.deepcopy(MINSEARCH)
    inp["algorithm"]["param"]["initial_list"] = [[0.0, 0.0], [1.0, 1.0]]
    assert api.validate(inp) == []
    inp["algorithm"]["param"]["initial_list"] = [[0.0, 0.0], [1.0]]
    assert "LIST_INVALID" in codes(api.validate(inp))
    inp["algorithm"]["param"]["initial_list"] = [[0.0, 0.0], [9.0, 0.0]]
    assert "INITIAL_OUT_OF_RANGE" in codes(api.validate(inp))
    inp["algorithm"]["param"]["initial_list"] = []
    assert "LIST_INVALID" in codes(api.validate(inp))


def test_analytical_functions_match_the_solver():
    import re
    src = open(os.path.join(SOURCE_PATH, "odatse", "solver", "analytical.py"), encoding="utf-8").read()
    names = set(re.findall(r'function_name == "([a-z_0-9]+)"', src))
    assert names == set(api.ANALYTICAL_FUNCTIONS)
    for fn, need in api.ANALYTICAL_FUNCTIONS.items():
        inp = copy.deepcopy(MAPPER)
        inp["solver"]["function_name"] = fn
        if need is not None:
            inp["base"]["dimension"] = need
            for k in ("min_list", "max_list"):
                inp["algorithm"]["param"][k] = [-1.0 if k == "min_list" else 1.0] * need
            inp["algorithm"]["param"]["num_list"] = [3] * need
        assert api.validate(inp) == [], fn


@pytest.mark.parametrize("fn", sorted(set(api.ANALYTICAL_FUNCTIONS) - {"himmelblau", "linear_regression_test"}))
def test_every_analytical_function_runs(tmp_path, fn):
    inp = copy.deepcopy(MAPPER)
    inp["solver"]["function_name"] = fn
    inp["algorithm"]["param"].update({"min_list": [-1.0, -1.0], "max_list": [1.0, 1.0], "num_list": [3, 3]})
    r = api.run(inp, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r


def test_dimension_resolution_follows_odatse(tmp_path):
    # dimension only in [algorithm] plus solver.dimension: valid, and parse() must not raise
    inp = copy.deepcopy(MAPPER)
    del inp["base"]["dimension"]
    inp["algorithm"]["dimension"] = 2
    inp["solver"]["dimension"] = 2
    assert api.validate(inp) == []
    r = api.run(inp, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r
    assert api.parse(tmp_path)["status"] == "ok"
    # the analytical solver needs solver.dimension or base.dimension
    del inp["solver"]["dimension"]
    assert "DIMENSION_INVALID" in codes(api.validate(inp))


def test_mapped_dimensions_are_checked_against_the_solver(tmp_path):
    inp = copy.deepcopy(MAPPER)
    inp["base"]["dimension"] = 1
    inp["algorithm"]["dimension"] = 1
    inp["algorithm"]["param"].update({"min_list": [-1.0], "max_list": [1.0], "num_list": [5]})
    inp["solver"]["dimension"] = 2  # himmelblau needs 2 at the solver, reached through the mapping
    assert "DIMENSION_MISMATCH" in codes(api.validate(inp))
    inp["runner"] = {"mapping": {"A": [[1.0], [0.5]], "b": [0.0, 0.0]}}
    assert "MAPPING_INVALID" in codes(api.validate(inp))  # b must be a column vector
    inp["runner"]["mapping"]["b"] = [[0.0], [0.0]]
    assert api.validate(inp) == []
    r = api.run(inp, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r
    inp["runner"]["mapping"]["A"] = [[1.0, 0.0]]
    assert "MAPPING_INVALID" in codes(api.validate(inp))
    inp["runner"]["mapping"] = {"b": [[0.0], [0.0]]}
    assert "MAPPING_INVALID" in codes(api.validate(inp))
    inp["runner"] = {}
    inp["solver"]["dimension"] = 3
    assert "DIMENSION_INVALID" in codes(api.validate(inp))


@pytest.mark.parametrize("link", ["output/res.txt", "output/ColorMap.txt", "output/0/status.pickle.tmp",
                                  "input.toml", "odatse.log"])
def test_file_symlinks_anywhere_in_workdir_are_refused(tmp_path, link):
    outside = tmp_path / "outside"
    outside.mkdir()
    wd = tmp_path / "wd"
    (wd / link).parent.mkdir(parents=True, exist_ok=True)
    (wd / link).symlink_to(outside / "victim")
    assert codes(api.prepare(MINSEARCH, wd)["issues"]) == {"UNSAFE_PATH"}
    assert codes(api.run(MINSEARCH, wd)["issues"]) == {"UNSAFE_PATH"}
    assert codes(api.parse(wd)["issues"]) == {"UNSAFE_PATH"}
    assert not (outside / "victim").exists()


def test_symlink_loop_gives_structured_issue(tmp_path):
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    assert codes(api.prepare(MINSEARCH, tmp_path)["issues"]) == {"UNSAFE_PATH"}
    assert codes(api.parse(tmp_path)["issues"]) == {"UNSAFE_PATH"}


@pytest.mark.parametrize("labels", [["fx", "y"], ["a", "a"], ["a=b", "c"], ["a\nfx", "c"], ["1a", "b"],
                                    ["index", "y"]])
def test_labels_that_break_result_files_are_rejected(labels):
    inp = copy.deepcopy(MINSEARCH)
    inp["algorithm"]["label_list"] = labels
    assert "LIST_INVALID" in codes(api.validate(inp))


def test_labels_are_used_in_results(tmp_path):
    inp = copy.deepcopy(MINSEARCH)
    inp["algorithm"]["label_list"] = ["alpha", "beta_2"]
    r = api.run(inp, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r
    assert set(r["best"]["x"]) == {"alpha", "beta_2"} and r["best"]["fx"] < 1e-4


@pytest.mark.parametrize("content", ["", "fx = 1.0\n", "fx = 1.0\nx1 = 2.0\n", "fx = nan\nx1 = 1\nx2 = 2\n",
                                     "x1 = 1.0\nx2 = 2.0\n"])
def test_incomplete_result_files_are_not_ok(tmp_path, content):
    (tmp_path / "input.toml").write_text(api.to_toml(MINSEARCH))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "res.txt").write_text(content)
    r = api.parse(tmp_path)
    assert r["status"] == "failed" and codes(r["issues"]) == {"PARSE_FAILED"}


def test_truncated_colormap_row_is_not_ok(tmp_path):
    (tmp_path / "input.toml").write_text(api.to_toml(MAPPER))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "ColorMap.txt").write_text("0.0 0.0 1.0\n0.5\n")
    assert codes(api.parse(tmp_path)["issues"]) == {"PARSE_FAILED"}


def test_grid_domains_follow_montecarlo_and_bayes(tmp_path):
    mesh = {"mesh_path": "mesh.txt"}
    # mapper / bayes: mesh_path replaces min/max/num_list
    for name in ("mapper", "bayes"):
        inp = copy.deepcopy(MAPPER)
        inp["algorithm"]["name"] = name
        inp["algorithm"]["param"] = dict(mesh)
        assert codes(api.validate(inp)) <= {"DEPENDENCY_MISSING"}, name
    # exchange on a mesh: no step_list, needs radius unless a neighbour list file is given
    ex = copy.deepcopy(EXCHANGE)
    ex["algorithm"]["param"] = dict(mesh)
    assert codes(api.validate(ex)) == {"KEY_MISSING"}
    ex["algorithm"]["param"]["neighborlist_path"] = "nn.txt"
    assert api.validate(ex) == []
    ex["algorithm"]["param"] = {**mesh, "radius": 0.5}
    assert api.validate(ex) == []
    # pamc with use_grid: min/max/num_list plus radius, no step_list
    pa = copy.deepcopy(EXCHANGE)
    pa["algorithm"]["name"] = "pamc"
    del pa["algorithm"]["exchange"]
    pa["algorithm"]["pamc"] = {"numsteps_annealing": 5, "Tnum": 3, "Tmin": 1.0, "Tmax": 2.0}
    pa["algorithm"]["param"] = {"use_grid": True, "min_list": [-1.0, -1.0], "max_list": [1.0, 1.0]}
    assert {i["field"] for i in api.validate(pa)} == {"algorithm.param.num_list", "algorithm.param.radius"}
    pa["algorithm"]["param"].update({"num_list": [5, 5], "radius": 0.6})
    assert api.validate(pa) == []
    r = api.run(pa, tmp_path, timeout_s=120)
    assert r["status"] == "ok", r


def test_trailing_newline_in_labels_and_keys_is_rejected():
    inp = copy.deepcopy(MINSEARCH)
    inp["algorithm"]["label_list"] = ["alpha\n", "beta"]
    assert "LIST_INVALID" in codes(api.validate(inp))
    inp = copy.deepcopy(MINSEARCH)
    inp["base"]["output_dir\n"] = "x"
    assert "KEY_INVALID" in codes(api.validate(inp))


@pytest.mark.parametrize("rows,ok", [
    ("0.0 0.0 nan\n0.5 0.5 2.0\n", True),        # failed evaluation (runner.ignore_error), then a valid row
    ("0.0 0.0 nan\n", False),                     # only failed evaluations
    ("nan 0.0 0.0\n0.0 0.0 1.0\n", False),       # NaN coordinate in the first row
    ("0.0 0.0 -inf\n0.5 0.5 2.0\n", True),       # rejected evaluation (-inf), then a valid row
    ("0.0 nan 1.0\n", False),                     # NaN coordinate
    ("inf 0.0 1.0\n", False),                     # infinite coordinate
    ("0.0 0.0 inf\n0.5 0.5 2.0\n", True),         # rejected evaluation, then a valid row
    ("0.0 0.0 inf\n", False),                     # no valid candidate
])
def test_non_finite_table_values(tmp_path, rows, ok):
    (tmp_path / "input.toml").write_text(api.to_toml(MAPPER))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "ColorMap.txt").write_text(rows)
    r = api.parse(tmp_path)
    if ok:
        assert r["status"] == "ok" and r["best"] == {"fx": 2.0, "x": {"x1": 0.5, "x2": 0.5}}
    else:
        assert r["status"] == "failed" and codes(r["issues"]) <= {"PARSE_FAILED", "RESULT_MISSING"}
        json.dumps(r, allow_nan=False)


def test_incomplete_best_result_and_bayes_rows(tmp_path):
    ex = copy.deepcopy(EXCHANGE)
    (tmp_path / "input.toml").write_text(api.to_toml(ex))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "best_result.txt").write_text("nprocs = 1\nrank = 0\nfx = 0.5\nx1 = 1.0\n")
    assert codes(api.parse(tmp_path)["issues"]) == {"PARSE_FAILED"}
    by = copy.deepcopy(MAPPER)
    by["algorithm"]["name"] = "bayes"
    (tmp_path / "input.toml").write_text(api.to_toml(by))
    (tmp_path / "output" / "BayesData.txt").write_text("#step x1 x2 fx\n0 1.0\n")
    assert codes(api.parse(tmp_path)["issues"]) == {"PARSE_FAILED"}


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1.0, "10", True, 10**400,
                                     -10**400])
def test_run_rejects_invalid_timeouts_before_writing(tmp_path, timeout):
    r = api.run(MINSEARCH, tmp_path / "wd", timeout_s=timeout)
    assert codes(r["issues"]) == {"TIMEOUT_INVALID"}
    assert not (tmp_path / "wd").exists()
