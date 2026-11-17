# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""JSON-in / JSON-out interface for tools and AI agents (shared core of :mod:`odatse.mcp_server`).

Functions
---------
schema()                       supported algorithms, required keys, built-in analytical functions
validate(input)                check an input dictionary (same structure as input.toml) without running
prepare(input, workdir)        write ``input.toml`` into ``workdir`` (nothing is executed)
run(input, workdir, timeout_s) run the **built-in analytical solver only** in a subprocess
parse(workdir)                 read the best result and list output files of a finished run
versions()                     ODAT-SE / NumPy / SciPy versions

Safety
------
* ``run`` executes ``python -m odatse input.toml`` with a fixed argument list, ``cwd=workdir`` and a
  timeout, in a separate process (ODAT-SE changes the working directory while running). Only
  ``solver.name = "analytical"`` is accepted; problems that need user-written solvers (Python
  functions, external programs) must be run by the user directly.
* ``base.root_dir`` / ``base.output_dir`` and every path-like value must be relative paths without
  ``..``, so that all files stay inside ``workdir``.
* Functions return issues ``{code, message, field, hint}`` instead of raising on bad input.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"

#: algorithm name -> required keys ("param.x" means [algorithm.param] x) and main result files
ALGORITHMS: dict[str, dict[str, Any]] = {
    "minsearch": {
        "description": "Nelder-Mead local minimisation inside a box (SciPy).",
        "required": ["param.min_list", "param.max_list"],
        "optional": ["param.initial_list", "param.unit_list", "minimize.initial_scale_list", "minimize.xatol",
                     "minimize.fatol", "minimize.maxiter", "minimize.maxfev"],
        "result_files": ["res.txt", "SimplexData.txt", "History_FunctionCall.txt"],
        "needs": ["scipy"],
    },
    "mapper": {
        "description": "Evaluate every point of a grid (num_list) or of a mesh file (mesh_path).",
        "required": ["param.min_list", "param.max_list", "param.num_list"],
        "optional": ["param.mesh_path", "colormap"],
        "result_files": ["ColorMap.txt"],
        "needs": [],
    },
    "random_search": {
        "description": "Evaluate num_points uniformly random points in the box.",
        "required": ["param.min_list", "param.max_list", "param.num_points"],
        "optional": ["colormap"],
        "result_files": ["ColorMap.txt"],
        "needs": [],
    },
    "exchange": {
        "description": "Replica-exchange Monte Carlo (sampling / global search).",
        "required": ["param.min_list", "param.max_list", "param.step_list", "exchange.numsteps",
                     "exchange.numsteps_exchange"],
        "optional": ["exchange.Tmin", "exchange.Tmax", "exchange.bmin", "exchange.bmax", "exchange.Tlogspace",
                     "exchange.nreplica_per_proc", "exchange.numsteps_thermalization"],
        "rules": ["give either Tmin/Tmax or bmin/bmax"],
        "result_files": ["best_result.txt", "result.txt", "trial.txt"],
        "needs": [],
    },
    "pamc": {
        "description": "Population annealing Monte Carlo (sampling, free energy / partition function).",
        "required": ["param.min_list", "param.max_list", "param.step_list"],
        "optional": ["pamc.Tnum", "pamc.numsteps", "pamc.numsteps_annealing", "pamc.Tmin", "pamc.Tmax",
                     "pamc.bmin", "pamc.bmax", "pamc.Tlogspace", "pamc.nreplica_per_proc",
                     "pamc.resampling_interval"],
        "rules": ["exactly two of pamc.numsteps, pamc.numsteps_annealing, pamc.Tnum must be positive",
                  "give either Tmin/Tmax or bmin/bmax"],
        "result_files": ["best_result.txt", "fx.txt", "result.txt", "trial.txt"],
        "needs": [],
    },
    "bayes": {
        "description": "Bayesian optimisation on a grid via PHYSBO.",
        "required": ["param.min_list", "param.max_list", "param.num_list"],
        "optional": ["bayes.random_max_num_probes", "bayes.bayes_max_num_probes", "bayes.score",
                     "bayes.interval", "bayes.num_rand_basis"],
        "result_files": ["BayesData.txt"],
        "needs": ["physbo"],
    },
}

#: built-in analytical test functions (odatse.solver.analytical) and dimension restrictions
ANALYTICAL_FUNCTIONS: dict[str, int | None] = {
    "quadratics": None,
    "quartics": None,
    "ackley": None,
    "rosenbrock": None,
    "himmelblau": 2,
    "linear_regression_test": 3,
}

_PATH_KEYS = {"root_dir", "output_dir", "mesh_path", "colormap", "filename"}


def _issue(code: str, message: str, field: str | None = None, hint: str | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "field": field, "hint": hint}


def schema() -> dict[str, Any]:
    """Supported algorithms with their required/optional keys and result files, plus analytical functions."""
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sections": {
            "base": {"required": ["dimension"], "optional": ["output_dir", "root_dir"]},
            "algorithm": {"required": ["name"], "optional": ["seed", "label_list", "param", "<algorithm section>"]},
            "solver": {"required": ["name"], "optional": ["function_name"]},
            "runner": {"optional": ["mapping", "limitation", "log", "ignore_error"]},
        },
        "algorithms": ALGORITHMS,
        "runnable_solvers": {"analytical": {"function_name": list(ANALYTICAL_FUNCTIONS)}},
        "notes": [
            "The input dictionary has the same structure as input.toml.",
            "Only solver.name='analytical' can be run through this API; user solvers must be run directly.",
        ],
    }


def versions() -> dict[str, str]:
    """Versions of ODAT-SE and its numerical dependencies (``not-installed`` when absent)."""
    import importlib

    import odatse

    out = {"odatse": odatse.__version__}
    for mod in ("numpy", "scipy", "physbo"):
        try:
            out[mod] = str(importlib.import_module(mod).__version__)
        except ImportError:
            out[mod] = "not-installed"
    return out


def _get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _num_list(v: Any, n: int) -> bool:
    return (isinstance(v, list) and len(v) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v))


def _safe_relpath(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    p = PurePosixPath(v.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and not v.startswith("~")


def validate(inp: dict[str, Any], *, for_run: bool = False) -> list[dict[str, Any]]:
    """Check an input dictionary (structure of input.toml) without running anything.

    ``for_run=True`` additionally requires the built-in analytical solver.
    """
    issues: list[dict[str, Any]] = []
    if not isinstance(inp, dict):
        return [_issue("INVALID_INPUT", "input must be a JSON object with base/algorithm/solver sections")]
    for sec in ("base", "algorithm", "solver"):
        if not isinstance(inp.get(sec), dict):
            issues.append(_issue("SECTION_MISSING", f"section [{sec}] is required", sec))
    if issues:
        return issues
    unknown = set(inp) - {"base", "algorithm", "solver", "runner"}
    for k in sorted(unknown):
        issues.append(_issue("UNKNOWN_SECTION", f"unknown section [{k}]", k))

    dim = inp["algorithm"].get("dimension", inp["base"].get("dimension"))
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        issues.append(_issue("DIMENSION_INVALID", "base.dimension must be a positive integer", "base.dimension"))
        dim = None

    name = inp["algorithm"].get("name")
    spec = ALGORITHMS.get(name) if isinstance(name, str) else None
    if spec is None:
        issues.append(_issue("ALGORITHM_UNKNOWN", f"algorithm.name must be one of {sorted(ALGORITHMS)}",
                             "algorithm.name"))
    else:
        alg = inp["algorithm"]
        has_mesh = _get(alg, "param.mesh_path") is not None
        for key in spec["required"]:
            if name == "mapper" and has_mesh and key in ("param.min_list", "param.max_list", "param.num_list"):
                continue
            if key == "param.step_list" and _get(alg, "param.unit_list") is not None:
                continue  # obsolete alias still accepted by odatse.algorithm.state (prints a WARNING)
            if _get(alg, key) is None:
                issues.append(_issue("KEY_MISSING", f"algorithm.{key} is required for {name}", f"algorithm.{key}"))
        if dim is not None:
            for key in ("param.min_list", "param.max_list", "param.initial_list", "param.unit_list",
                        "param.step_list", "param.num_list", "minimize.initial_scale_list"):
                v = _get(alg, key)
                if v is not None and not _num_list(v, dim):
                    issues.append(_issue("LIST_INVALID", f"algorithm.{key} must be {dim} finite numbers",
                                         f"algorithm.{key}"))
            lo, hi = _get(alg, "param.min_list"), _get(alg, "param.max_list")
            if _num_list(lo, dim) and _num_list(hi, dim) and not all(a < b for a, b in zip(lo, hi)):
                issues.append(_issue("BOUNDS_INVALID", "every min_list entry must be < max_list", "algorithm.param"))
            ini = _get(alg, "param.initial_list")
            if name == "minsearch" and _num_list(ini, dim) and _num_list(lo, dim) and _num_list(hi, dim) \
                    and not all(a <= x <= b for a, x, b in zip(lo, ini, hi)):
                issues.append(_issue("INITIAL_OUT_OF_RANGE", "initial_list must lie inside [min_list, max_list]",
                                     "algorithm.param.initial_list"))
            nl = _get(alg, "param.num_list")
            if _num_list(nl, dim) and not all(isinstance(n, int) and n >= 1 for n in nl):
                issues.append(_issue("LIST_INVALID", "num_list entries must be positive integers",
                                     "algorithm.param.num_list"))
        for sec in ("exchange", "pamc"):
            if name == sec and isinstance(alg.get(sec), dict):
                s = alg[sec]
                has_t = "Tmin" in s or "Tmax" in s
                has_b = "bmin" in s or "bmax" in s
                if has_t == has_b:
                    issues.append(_issue("TEMPERATURE_INVALID", f"give either Tmin/Tmax or bmin/bmax in [{sec}]",
                                         f"algorithm.{sec}"))
        if name == "pamc":
            s = alg.get("pamc") if isinstance(alg.get("pamc"), dict) else {}
            positive = [k for k in ("numsteps", "numsteps_annealing", "Tnum")
                        if isinstance(s.get(k, 0), (int, float)) and s.get(k, 0) > 0]
            if len(positive) != 2:
                issues.append(_issue("SCHEDULE_INVALID", "exactly two of numsteps, numsteps_annealing and Tnum "
                                     f"must be positive in [pamc] (positive: {positive})", "algorithm.pamc"))
        missing = [m for m in spec["needs"] if _missing_module(m)]
        if missing:
            issues.append(_issue("DEPENDENCY_MISSING", f"{name} needs {missing}", "algorithm.name",
                                 f"pip install {' '.join(missing)}"))

    sname = inp["solver"].get("name")
    if sname == "analytical":
        fn = inp["solver"].get("function_name", "quadratics")
        if fn not in ANALYTICAL_FUNCTIONS:
            issues.append(_issue("FUNCTION_UNKNOWN", f"function_name must be one of {list(ANALYTICAL_FUNCTIONS)}",
                                 "solver.function_name"))
        elif ANALYTICAL_FUNCTIONS[fn] is not None and dim is not None and dim != ANALYTICAL_FUNCTIONS[fn]:
            issues.append(_issue("DIMENSION_INVALID", f"{fn} requires dimension={ANALYTICAL_FUNCTIONS[fn]}",
                                 "base.dimension"))
    elif for_run:
        issues.append(_issue("SOLVER_NOT_RUNNABLE", f"solver '{sname}' cannot be run through this API",
                             "solver.name", "only the built-in 'analytical' solver can be run here; run user "
                             "solvers with your own script"))

    def walk(d: Any, prefix: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                f = f"{prefix}.{k}" if prefix else k
                if k in _PATH_KEYS and not _safe_relpath(v):
                    issues.append(_issue("UNSAFE_PATH", f"{f} must be a relative path without '..'", f))
                walk(v, f)

    walk(inp, "")
    return issues


def _missing_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite numbers cannot be written to TOML input")
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise ValueError(f"unsupported value type in input: {type(v).__name__}")


def to_toml(inp: dict[str, Any]) -> str:
    """Serialise an input dictionary (nested tables of scalars and lists) to TOML text."""
    lines: list[str] = []

    def emit(table: dict[str, Any], path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        if path:
            lines.append("[" + ".".join(path) + "]")
        for k, v in scalars.items():
            lines.append(f"{k} = {_toml_value(v)}")
        if path:
            lines.append("")
        for k, v in subs.items():
            emit(v, [*path, k])

    emit(inp, [])
    return "\n".join(lines).rstrip() + "\n"


def prepare(inp: dict[str, Any], workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate and write ``workdir/input.toml``. Nothing is executed."""
    issues = validate(inp)
    if issues:
        return {"status": "failed", "issues": issues}
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    path = wd / "input.toml"
    path.write_text(to_toml(inp), encoding="utf-8")
    return {"status": "ok", "files": [str(path)], "input_toml": path.read_text(encoding="utf-8")}


def run(inp: dict[str, Any], workdir: str | os.PathLike[str], *, timeout_s: float = 600.0) -> dict[str, Any]:
    """Run ODAT-SE with the built-in analytical solver in a subprocess and return :func:`parse` output."""
    issues = validate(inp, for_run=True)
    if issues:
        return {"status": "failed", "issues": issues}
    prep = prepare(inp, workdir)
    if prep["status"] != "ok":
        return prep
    wd = Path(workdir)
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL",
                                                         "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                         "MKL_NUM_THREADS")}
    # the subprocess must import the very same odatse package as the caller
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([pkg_root, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    t0 = time.monotonic()
    with open(wd / "odatse.log", "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run([sys.executable, "-m", "odatse", "input.toml"], cwd=wd, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "issues": [_issue("TIMEOUT", f"run exceeded {timeout_s} s", None,
                                                          "reduce the number of steps or increase timeout_s")],
                    "log_tail": _tail(wd / "odatse.log")}
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "issues": [_issue("RUN_FAILED", f"odatse exited with {proc.returncode}")],
                "log_tail": _tail(wd / "odatse.log")}
    out = parse(wd)
    out["elapsed_s"] = round(elapsed, 3)
    out["provenance"] = {"api_schema_version": SCHEMA_VERSION, "versions": versions(),
                         "seed": inp["algorithm"].get("seed"), "command": "python -m odatse input.toml"}
    return out


def _tail(p: Path, n: int = 20) -> list[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _read_kv(p: Path) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _best_from_table(p: Path, fx_col: int, x_cols: slice, labels: list[str]) -> dict[str, Any] | None:
    best = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = [float(c) for c in line.split()]
        if best is None or cols[fx_col] < best[fx_col]:
            best = cols
    if best is None:
        return None
    return {"fx": best[fx_col], "x": dict(zip(labels, best[x_cols]))}


def parse(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Best result and output file list of a finished run in ``workdir`` (reads ``input.toml`` there)."""
    from .util import toml

    wd = Path(workdir)
    try:
        inp = toml.load(str(wd / "input.toml"))
    except OSError as exc:
        return {"status": "failed", "issues": [_issue("NOT_FOUND", f"input.toml not readable: {exc}")]}
    name = inp["algorithm"]["name"]
    dim = int(inp["algorithm"].get("dimension", inp["base"]["dimension"]))
    labels = inp["algorithm"].get("label_list", [f"x{d + 1}" for d in range(dim)])
    out_dir = wd / inp["base"].get("root_dir", ".") / inp["base"].get("output_dir", ".")
    files = sorted(str(p.relative_to(wd)) for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() else []
    best: dict[str, Any] | None = None
    try:
        if name == "minsearch" and (out_dir / "res.txt").exists():
            kv = _read_kv(out_dir / "res.txt")
            best = {"fx": kv.get("fx"), "x": {k: kv[k] for k in labels if k in kv}}
        elif name in ("exchange", "pamc") and (out_dir / "best_result.txt").exists():
            kv = _read_kv(out_dir / "best_result.txt")
            best = {"fx": kv.get("fx"), "x": {k: kv[k] for k in labels if k in kv}}
        elif name in ("mapper", "random_search"):
            cmap = out_dir / inp["algorithm"].get("colormap", "ColorMap.txt")
            if cmap.exists():
                best = _best_from_table(cmap, dim, slice(0, dim), labels)
        elif name == "bayes" and (out_dir / "BayesData.txt").exists():
            # columns: step, best x..., best fx, action x..., action fx (fx written as in the run)
            best = _best_from_table(out_dir / "BayesData.txt", 1 + dim, slice(1, 1 + dim), labels)
    except (ValueError, IndexError, KeyError) as exc:
        return {"status": "failed", "algorithm": name, "files": files,
                "issues": [_issue("PARSE_FAILED", f"could not parse result files: {exc}")]}
    if best is None:
        return {"status": "failed", "algorithm": name, "files": files,
                "issues": [_issue("RESULT_MISSING", f"no result file for {name} in {out_dir}")]}
    return {"status": "ok", "algorithm": name, "best": best, "files": files}
