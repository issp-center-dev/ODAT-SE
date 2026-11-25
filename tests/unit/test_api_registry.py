import asyncio
import copy
import json
import os
import sys
import types

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse import api
from odatse.solver import registry

PLUGINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
MOD = "odatse_test_solvers"


@pytest.fixture(autouse=True)
def plugin_path(monkeypatch):
    """Make the test plugins importable here and in the subprocess of api.run; no installed entry points."""
    monkeypatch.syspath_prepend(PLUGINS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([PLUGINS, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(registry, "entry_point_solvers", lambda: {})
    registry.describe.cache_clear()
    yield
    registry.describe.cache_clear()


def reg(**names):
    return {n: f"{MOD}:{attr}" for n, attr in names.items()}


def inp_for(solver, algorithm="minsearch", dim=2, **alg):
    param = {"min_list": [-3.0] * dim, "max_list": [3.0] * dim}
    if algorithm == "minsearch":
        param["initial_list"] = [1.0] * dim
    elif algorithm == "mapper":
        param["num_list"] = [7] * dim
    out = {"base": {"dimension": dim, "output_dir": "output"}, "solver": solver,
           "algorithm": {"name": algorithm, "seed": 7, "param": param, **alg}}
    return out


def codes(issues):
    return {i["code"] for i in issues}


# --- registry parsing ---------------------------------------------------------------------------

def test_parse_spec_accepts_names_and_targets():
    assert registry.parse_spec("") == {}
    assert registry.parse_spec(" a=pkg.mod:f , b = pkg:obj.method ,") == {"a": "pkg.mod:f", "b": "pkg:obj.method"}


@pytest.mark.parametrize("text, words", [
    ("a", "expected name="),
    ("1a=pkg:f", "identifier"),
    ("analytical=pkg:f", "reserved"),
    ("function=pkg:f", "reserved"),
    ("a=pkg", "package.module:attribute"),
    ("a=pkg:f;rm -rf", "package.module:attribute"),
    ("a=../x:f", "package.module:attribute"),
    ("a=pkg:f,a=pkg:g", "twice"),
])
def test_parse_spec_rejects_malformed_entries(text, words):
    with pytest.raises(ValueError, match=words):
        registry.parse_spec(text)


def test_entry_points_are_read_and_explicit_entries_replace_them(monkeypatch):
    import importlib.metadata as md

    dist = types.SimpleNamespace(name="mydist", version="1.2.3")
    eps = [types.SimpleNamespace(name="sphere", value=f"{MOD}:shifted_sphere", dist=dist),
           types.SimpleNamespace(name="bad name", value=f"{MOD}:plane2", dist=dist),
           types.SimpleNamespace(name="plane", value="not a target", dist=None)]

    def fake_entry_points(**kw):
        assert kw == {"group": "odatse.solvers"}
        return eps

    monkeypatch.undo()  # drop the autouse stub of entry_point_solvers, keep nothing else
    monkeypatch.syspath_prepend(PLUGINS)
    monkeypatch.setattr(md, "entry_points", fake_entry_points)
    r, problems = registry.registry()
    assert r == {"sphere": {"source": "entry_point", "target": f"{MOD}:shifted_sphere", "distribution": "mydist",
                            "version": "1.2.3"}}
    assert len(problems) == 2
    r, _ = registry.registry({"sphere": f"{MOD}:plane2"})
    assert r["sphere"]["source"] == "explicit" and r["sphere"]["target"] == f"{MOD}:plane2"


def test_registry_reports_invalid_explicit_mappings():
    r, problems = registry.registry({"ok": f"{MOD}:plane2", "analytical": f"{MOD}:plane2", "x": "nope"})
    assert set(r) == {"ok"} and len(problems) == 2
    r, problems = registry.registry(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert r == {} and problems


def test_describe_reads_signature_and_attributes():
    d = registry.describe(f"{MOD}:shifted_sphere")
    assert d["description"] == "sphere centred at `center`"
    assert d["dimension"] is None and d["accepts_any_params"] is False
    assert d["params"] == {"center": {"required": False, "default": 0.0},
                           "scale": {"required": False, "default": 1.0}}
    assert registry.describe(f"{MOD}:needs_target")["params"]["target"]["required"] is True
    assert registry.describe(f"{MOD}:plane2")["dimension"] == 2
    assert registry.describe(f"{MOD}:any_kwargs")["accepts_any_params"] is True
    for attr, words in [("bad_dimension_attr", "odatse_dimension"), ("positional_only", "positional-only"),
                        ("not_callable", "not callable"), ("missing", "cannot import")]:
        assert words in registry.describe(f"{MOD}:{attr}")["error"]
    assert "cannot import" in registry.describe("odatse_broken_plugin:f")["error"]


# --- schema and validate --------------------------------------------------------------------------

def test_schema_lists_registered_solvers():
    s = api.schema(solvers=reg(sphere="shifted_sphere", broken="not_callable"))
    json.dumps(s)
    assert s["registered_solvers"]["sphere"]["params"]["center"]["default"] == 0.0
    assert s["registered_solvers"]["sphere"]["source"] == "explicit"
    assert "error" in s["registered_solvers"]["broken"]
    assert api.schema()["registered_solvers"] == {}


def test_validate_registered_solver_params():
    solvers = reg(sphere="shifted_sphere", target="needs_target", plane="plane2", anykw="any_kwargs")
    ok = inp_for({"name": "sphere", "params": {"center": 1.0}})
    assert api.validate(ok, for_run=True, solvers=solvers) == []
    assert api.validate(inp_for({"name": "anykw", "params": {"whatever": [1, 2]}}), for_run=True,
                        solvers=solvers) == []
    c = codes(api.validate(inp_for({"name": "sphere", "params": {"centre": 1.0}}), solvers=solvers))
    assert c == {"PARAM_UNKNOWN"}
    assert codes(api.validate(inp_for({"name": "target"}), solvers=solvers)) == {"PARAM_MISSING"}
    assert codes(api.validate(inp_for({"name": "sphere", "params": {"center": {"a": 1}}}),
                              solvers=solvers)) == {"PARAMS_INVALID"}
    assert codes(api.validate(inp_for({"name": "sphere", "params": 3}), solvers=solvers)) == {"PARAMS_INVALID"}
    assert codes(api.validate(inp_for({"name": "sphere", "params": {"center": None}}),
                              solvers=solvers)) == {"VALUE_INVALID"}
    assert codes(api.validate(inp_for({"name": "sphere", "function_name": "x"}),
                              solvers=solvers)) == {"SOLVER_KEY_UNKNOWN"}
    assert codes(api.validate(inp_for({"name": "plane"}, dim=3), solvers=solvers)) == {"DIMENSION_INVALID"}
    assert api.validate(inp_for({"name": "plane"}, dim=2), solvers=solvers) == []


def test_validate_for_run_requires_a_registered_or_analytical_solver():
    inp = inp_for({"name": "sphere"})
    assert api.validate(inp) == []  # rendering for the user's own script is fine
    assert codes(api.validate(inp, for_run=True)) == {"SOLVER_NOT_RUNNABLE"}
    assert codes(api.validate(inp, for_run=True, solvers={"sphere": f"{MOD}:not_callable"})) == \
        {"SOLVER_UNAVAILABLE"}
    c = codes(api.validate(inp, for_run=True, solvers={"x": "bad target"}))
    assert c == {"REGISTRY_INVALID", "SOLVER_NOT_RUNNABLE"}


def test_render_includes_params_table():
    out = api.render(inp_for({"name": "sphere", "params": {"center": 1.5, "scale": 2}}),
                     solvers=reg(sphere="shifted_sphere"))
    assert out["status"] == "ok"
    assert "[solver.params]" in out["input_toml"] and "center = 1.5" in out["input_toml"]


# --- runs -----------------------------------------------------------------------------------------

def test_run_registered_solver_minsearch(tmp_path):
    inp = inp_for({"name": "sphere", "params": {"center": 1.5}})
    r = api.run(inp, tmp_path, timeout_s=120, solvers=reg(sphere="shifted_sphere"))
    assert r["status"] == "ok", r
    assert r["best"]["x"]["x1"] == pytest.approx(1.5, abs=1e-3)
    assert r["best"]["fx"] == pytest.approx(0.0, abs=1e-5)
    prov = r["provenance"]["solver"]
    assert prov["name"] == "sphere" and prov["target"] == f"{MOD}:shifted_sphere"
    assert prov["params"] == {"center": 1.5} and prov["source"] == "explicit"
    assert "distribution" in prov and "version" in prov
    assert "odatse._api_runner" in r["provenance"]["command"]


def test_run_registered_solver_mapper_with_inf_rejections(tmp_path):
    inp = inp_for({"name": "half"}, algorithm="mapper", dim=1)
    r = api.run(inp, tmp_path, timeout_s=120, solvers=reg(half="returns_inf_right"))
    assert r["status"] == "ok", r
    assert r["best"]["fx"] == pytest.approx(0.0)


@pytest.mark.parametrize("attr, words", [("raises", "ValueError"), ("returns_nan", "nan"),
                                         ("returns_text", "not a number")])
def test_solver_failures_are_structured(tmp_path, attr, words):
    r = api.run(inp_for({"name": "bad"}), tmp_path, timeout_s=120, solvers=reg(bad=attr))
    assert r["status"] == "failed" and codes(r["issues"]) == {"SOLVER_FAILED"}
    assert any(words in line for line in r["log_tail"]), r["log_tail"]


def test_ignore_error_records_nan_and_continues(tmp_path):
    inp = inp_for({"name": "half"}, algorithm="mapper", dim=1)
    inp["runner"] = {"ignore_error": True}
    r = api.run(inp, tmp_path, timeout_s=120, solvers=reg(half="raises_at_positive"))
    assert r["status"] == "ok", r
    assert r["best"] == {"fx": 0.0, "x": {"x1": 0.0}}
    rows = (tmp_path / "output" / "ColorMap.txt").read_text().split("\n")
    assert sum("nan" in row for row in rows) == 3  # x = 1, 2, 3 failed and were recorded as NaN


def test_solver_that_changes_directory_and_mutates_inputs(tmp_path):
    r = api.run(inp_for({"name": "away"}, algorithm="mapper"), tmp_path / "a", timeout_s=120,
                solvers=reg(away="chdir_away"))
    assert r["status"] == "ok", r
    assert (tmp_path / "a" / "output" / "ColorMap.txt").exists()
    r = api.run(inp_for({"name": "mut", "params": {"values": [1.0, 2.0]}}, algorithm="mapper"), tmp_path / "b",
                timeout_s=120, solvers=reg(mut="mutates"))
    assert r["status"] == "ok", r
    assert r["best"]["fx"] == 3.0  # the params list was not grown by earlier evaluations
    assert r["best"]["x"]["x1"] == -3.0  # the vector change inside the solver did not reach ODAT-SE


def test_run_reports_import_failure_in_the_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")  # importable here, not in the run
    r = api.run(inp_for({"name": "sphere"}), tmp_path, timeout_s=120, solvers=reg(sphere="shifted_sphere"))
    assert r["status"] == "failed" and codes(r["issues"]) == {"SOLVER_UNAVAILABLE"}


def test_run_refuses_unregistered_names_without_writing(tmp_path):
    r = api.run(inp_for({"name": "sphere"}), tmp_path)
    assert codes(r["issues"]) == {"SOLVER_NOT_RUNNABLE"}
    assert not (tmp_path / "input.toml").exists()


# --- MCP ------------------------------------------------------------------------------------------

def _call(server, name, args):
    out = asyncio.run(server.call_tool(name, args))
    payload = out[1] if isinstance(out, tuple) else out
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    if not isinstance(payload, dict):
        payload = json.loads(payload[0].text)
    return payload


def test_mcp_server_lists_and_runs_registered_solvers(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from odatse import mcp_server

    monkeypatch.setenv("ODATSE_MCP_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ODATSE_MCP_SOLVERS", f"sphere={MOD}:shifted_sphere")
    monkeypatch.delenv("ODATSE_MCP_ALLOW_RUN", raising=False)
    server = mcp_server.build_server()
    s = _call(server, "get_schema", {})
    assert "sphere" in s["registered_solvers"]
    inp = inp_for({"name": "sphere", "params": {"center": 0.5}})
    assert _call(server, "validate_input", {"input": inp, "for_run": True})["valid"] is True
    assert _call(server, "run_calculation", {"input": inp})["issues"][0]["code"] == "RUN_DISABLED"
    monkeypatch.setenv("ODATSE_MCP_ALLOW_RUN", "1")
    r = _call(server, "run_calculation", {"input": inp, "timeout_s": 120})
    assert r["status"] == "ok", r
    assert r["best"]["x"]["x1"] == pytest.approx(0.5, abs=1e-3)


def test_mcp_server_refuses_malformed_solver_config(monkeypatch):
    pytest.importorskip("mcp")
    from odatse import mcp_server

    monkeypatch.setenv("ODATSE_MCP_SOLVERS", "analytical=pkg:f")
    with pytest.raises(SystemExit, match="ODATSE_MCP_SOLVERS"):
        mcp_server.build_server()


# --- review round 1 ------------------------------------------------------------------------------

def test_params_cannot_collide_with_the_vector_argument():
    solvers = reg(kw="kwargs_only", pos="vector_positional_only", sphere="shifted_sphere")
    assert codes(api.validate(inp_for({"name": "kw", "params": {"x": 1}}), solvers=solvers)) == {"PARAM_UNKNOWN"}
    assert api.validate(inp_for({"name": "pos", "params": {"x": 1.0}}), for_run=True, solvers=solvers) == []


def test_params_named_like_odatse_path_keys_are_function_arguments(tmp_path):
    solvers = reg(p="path_named_params")
    inp = inp_for({"name": "p", "params": {"output_dir": "/abs/scratch", "colormap": [1, 2],
                                           "checkpoint_file": "../x"}}, algorithm="mapper")
    assert api.validate(inp, for_run=True, solvers=solvers) == []
    r = api.run(inp, tmp_path, timeout_s=120, solvers=solvers)
    assert r["status"] == "ok", r
    bad = copy.deepcopy(inp)
    bad["base"]["output_dir"] = "/abs"
    assert codes(api.validate(bad, solvers=solvers)) == {"UNSAFE_PATH"}  # ODAT-SE's own settings still checked


@pytest.mark.parametrize("shadow", ["odatse.py", f"{MOD}.py", "runpy.py"])
def test_files_in_the_working_directory_cannot_shadow_modules(tmp_path, shadow):
    (tmp_path / shadow).write_text("import sys\nsys.exit(97)\n")
    r = api.run(inp_for({"name": "sphere"}, algorithm="mapper"), tmp_path, timeout_s=120,
                solvers=reg(sphere="shifted_sphere"))
    assert r["status"] == "ok", r


def test_bootstrap_drops_the_empty_path_entry_on_every_python(tmp_path):
    import subprocess

    code = api._BOOTSTRAP
    probe = tmp_path / "probe_mod.py"
    probe.write_text("import sys\nprint(repr(sys.path[:1]), sys.argv[1:])\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "elsewhere"))
    env.pop("PYTHONSAFEPATH", None)
    out = subprocess.run([sys.executable, "-c", code, "probe_mod", "a"], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert out.returncode != 0 and "No module named probe_mod" in out.stderr


# --- review round 2 ------------------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["returns_bad_float", "returns_unprintable", "raises_unprintable"])
def test_conversion_errors_are_solver_failures(tmp_path, attr):
    r = api.run(inp_for({"name": "b"}), tmp_path / "a", timeout_s=120, solvers=reg(b=attr))
    assert codes(r["issues"]) == {"SOLVER_FAILED"}, r
    assert any("ERROR: solver 'b'" in line for line in r["log_tail"])
    inp = inp_for({"name": "b"}, algorithm="mapper", dim=1)
    inp["runner"] = {"ignore_error": True}
    r = api.run(inp, tmp_path / "b", timeout_s=120, solvers=reg(b=attr))
    assert codes(r["issues"]) == {"RESULT_MISSING"}  # every point failed and was recorded as NaN
    assert (tmp_path / "b" / "output" / "ColorMap.txt").read_text().count("nan") == 7


def test_duplicate_entry_point_names_are_ambiguous(monkeypatch):
    import importlib.metadata as md

    d1 = types.SimpleNamespace(name="dist1", version="1")
    d2 = types.SimpleNamespace(name="dist2", version="2")
    eps = [types.SimpleNamespace(name="loss", value=f"{MOD}:shifted_sphere", dist=d1),
           types.SimpleNamespace(name="loss", value=f"{MOD}:plane2", dist=d2),
           types.SimpleNamespace(name="same", value=f"{MOD}:plane2", dist=d1),
           types.SimpleNamespace(name="same", value=f"{MOD}:plane2", dist=d2)]
    monkeypatch.undo()
    monkeypatch.syspath_prepend(PLUGINS)
    monkeypatch.setattr(md, "entry_points", lambda **kw: eps)
    r, problems = registry.registry()
    assert set(r) == {"same"} and len(problems) == 1 and "loss" in problems[0]
    r, _ = registry.registry({"loss": f"{MOD}:plane2"})
    assert r["loss"]["target"] == f"{MOD}:plane2" and r["loss"]["source"] == "explicit"
    inp = inp_for({"name": "loss"})
    assert codes(api.validate(inp, for_run=True)) == {"REGISTRY_INVALID", "SOLVER_NOT_RUNNABLE"}
