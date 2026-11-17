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
