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
render(input)                  validate and return the ``input.toml`` text (nothing is written)
prepare(input, workdir)        write ``input.toml`` into ``workdir`` (nothing is executed)
run(input, workdir, timeout_s) run the built-in analytical solver or a registered solver in a subprocess
parse(workdir)                 read the best result and list output files of a finished run
versions()                     ODAT-SE / NumPy / SciPy versions

Safety
------
* ``run`` executes ``python -m odatse input.toml`` (analytical solver) or
  ``python -m odatse._api_runner input.toml --solver <target>`` (registered solver) with a fixed argument
  list, ``cwd=workdir`` and a timeout, in a separate process (ODAT-SE changes the working directory while
  running).
* Registered solvers (:mod:`odatse.solver.registry`) are Python functions the **operator** installed and
  registered (entry points ``odatse.solvers`` or the ``solvers`` argument). Callers choose one by
  ``solver.name`` and pass ``solver.params``; no code is ever accepted from the input. The functions
  ``validate``, ``render``, ``prepare``, ``run`` and ``schema`` take ``solvers={name: "module:attr"}``.
  Importing a registered module runs its code in the calling process (to read its signature).
* Every path-bearing value (``root_dir``, ``output_dir``, ``mesh_path``, ``neighborlist_path``,
  ``checkpoint_file``, ``colormap``, ``runner.log.filename``) must be a relative path without ``..``.
  :func:`prepare` and :func:`parse` also resolve them (following existing symbolic links) against the
  directory ODAT-SE resolves them against, and refuse anything that ends up outside ``workdir``.
* Keys must be TOML bare keys and values JSON scalars, lists or tables (no ``null``); the generated
  TOML is parsed back and must equal the validated dictionary.
* Functions return issues ``{code, message, field, hint}`` instead of raising on bad input.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .solver import registry as _registry

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
        "rules": ["param.mesh_path replaces min_list/max_list/num_list"],
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
                     "exchange.nreplica_per_proc", "exchange.numsteps_thermalization", "param.mesh_path",
                     "param.use_grid", "param.num_list", "param.neighborlist_path", "param.radius"],
        "rules": ["give either Tmin/Tmax or bmin/bmax",
                  "grid search: param.mesh_path (or use_grid = true with num_list) replaces step_list, and needs "
                  "param.radius unless mesh_path comes with param.neighborlist_path"],
        "result_files": ["best_result.txt", "result.txt", "trial.txt"],
        "needs": [],
    },
    "pamc": {
        "description": "Population annealing Monte Carlo (sampling, free energy / partition function).",
        "required": ["param.min_list", "param.max_list", "param.step_list", "pamc"],
        "optional": ["pamc.Tnum", "pamc.numsteps", "pamc.numsteps_annealing", "pamc.Tmin", "pamc.Tmax",
                     "pamc.bmin", "pamc.bmax", "pamc.Tlogspace", "pamc.nreplica_per_proc",
                     "pamc.resampling_interval", "param.mesh_path", "param.use_grid", "param.num_list",
                     "param.neighborlist_path", "param.radius"],
        "rules": ["exactly two of pamc.numsteps, pamc.numsteps_annealing, pamc.Tnum must be positive",
                  "give either Tmin/Tmax or bmin/bmax",
                  "grid search: as for exchange"],
        "result_files": ["best_result.txt", "fx.txt", "result.txt", "trial.txt"],
        "needs": [],
    },
    "bayes": {
        "description": "Bayesian optimisation on a grid via PHYSBO.",
        "required": ["param.min_list", "param.max_list", "param.num_list"],
        "optional": ["param.mesh_path", "bayes.random_max_num_probes", "bayes.bayes_max_num_probes",
                     "bayes.score", "bayes.interval", "bayes.num_rand_basis"],
        "rules": ["param.mesh_path replaces min_list/max_list/num_list"],
        "result_files": ["BayesData.txt"],
        "needs": ["physbo"],
    },
}


#: built-in analytical test functions (odatse.solver.analytical) and dimension restrictions;
#: keep in sync with ``Solver.__init__`` there (tests compare the two)
ANALYTICAL_FUNCTIONS: dict[str, int | None] = {
    "quadratics": None,
    "quartics": None,
    "ackley": None,
    "alpine": None,
    "exponential": None,
    "griewank": None,
    "himmelblau": 2,
    "michalewicz": None,
    "qing": None,
    "rastrigin": None,
    "rosenbrock": None,
    "schaffer": None,
    "schwefel": None,
    "linear_regression_test": 3,
}

#: path-bearing keys (at any depth) and the directories ODAT-SE resolves them against:
#: "work" = working directory, "root" = base.root_dir, "output" = output_dir, "proc" = output_dir/<rank>
_PATH_KEYS: dict[str, tuple[str, ...]] = {
    "root_dir": ("work",),
    "output_dir": ("root",),
    "mesh_path": ("root",),
    "neighborlist_path": ("work",),
    "checkpoint_file": ("proc",),
    "colormap": ("output", "proc"),
    "filename": ("proc",),
}

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
#: labels are written as ``label = value`` lines next to these keys in res.txt / best_result.txt
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_LABELS = {"fx", "nprocs", "rank", "step", "walker", "index", "iterations", "function_evaluations"}
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _issue(code: str, message: str, field: str | None = None, hint: str | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "field": field, "hint": hint}


def _registered(solvers: Mapping[str, str] | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    return _registry.registry(solvers)


def registered_solvers(solvers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Registered solvers with their description, dimension and parameters, plus registry problems."""
    reg, problems = _registered(solvers)
    out: dict[str, Any] = {}
    for name, entry in sorted(reg.items()):
        out[name] = {"target": entry["target"], "source": entry["source"], **_registry.describe(entry["target"])}
    return {"solvers": out, "problems": problems}


def schema(solvers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Supported algorithms with their required/optional keys and result files, analytical functions and
    registered solvers."""
    reg = registered_solvers(solvers)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sections": {
            "base": {"required": ["dimension"], "optional": ["output_dir", "root_dir"]},
            "algorithm": {"required": ["name"], "optional": ["dimension", "seed", "label_list", "param",
                                                             "checkpoint", "checkpoint_file", "<algorithm section>"]},
            "solver": {"required": ["name"], "optional": ["function_name (analytical)", "dimension",
                                                          "params (registered solvers)"]},
            "runner": {"optional": ["mapping", "limitation", "log", "ignore_error"]},
        },
        "algorithms": ALGORITHMS,
        "runnable_solvers": {"analytical": {"function_name": list(ANALYTICAL_FUNCTIONS)}},
        "registered_solvers": reg["solvers"],
        "registry_problems": reg["problems"],
        "path_keys": sorted(_PATH_KEYS),
        "notes": [
            "The input dictionary has the same structure as input.toml.",
            "Keys must be TOML bare keys (letters, digits, '_' and '-'); null values are not allowed.",
            "Path-bearing keys must be relative paths without '..' and must stay inside the working directory.",
            "solver.name='analytical' (with function_name) or the name of a registered solver (with params) can "
            "be run; other solvers must be run with your own script.",
            "A registered solver is called as f(x, **solver.params); params must match its parameters.",
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


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True  # Python integers are exact; math.isfinite() would overflow on very large ones
    return isinstance(v, float) and math.isfinite(v)


def _seconds(v: Any) -> float | None:
    """A finite positive number of seconds as float, or None."""
    if not _is_num(v):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) and f > 0 else None


def _num_list(v: Any, n: int) -> bool:
    return isinstance(v, list) and len(v) == n and all(_is_num(x) for x in v)


def _vectors(v: Any, n: int) -> list[list[float]] | None:
    """``initial_list`` forms accepted by ``domain/region.py``: one vector, or one vector per walker."""
    if _num_list(v, n):
        return [v]
    if isinstance(v, list) and v and all(_num_list(row, n) for row in v):
        return v
    return None


def _safe_relpath(v: Any) -> bool:
    if not isinstance(v, str) or not v or any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
        return False
    p = PurePosixPath(v.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and not v.startswith("~") and not re.match(r"^[A-Za-z]:", v)


def _value_issues(d: dict[str, Any], prefix: str, issues: list[dict[str, Any]]) -> None:
    """Keys must be TOML bare keys; values JSON scalars (no null), lists or tables, representable in TOML."""
    for k, v in d.items():
        if not isinstance(k, str) or not _BARE_KEY.fullmatch(k):
            issues.append(_issue("KEY_INVALID", f"key {k!r} in [{prefix or 'top level'}] is not a valid name",
                                 prefix or None, "use letters, digits, '_' and '-' only"))
            continue
        f = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _value_issues(v, f, issues)
        else:
            msg = _bad_value(v)
            if msg:
                issues.append(_issue("VALUE_INVALID", f"{f}: {msg}", f))


def _bad_value(v: Any) -> str | None:
    if v is None:
        return "null is not allowed (omit the key instead)"
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return None if _INT64_MIN <= v <= _INT64_MAX else "integer out of the 64-bit range"
    if isinstance(v, float):
        return None if math.isfinite(v) else "numbers must be finite"
    if isinstance(v, str):
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            return "string is not valid Unicode"
        return None
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                return "tables inside arrays are not supported"
            msg = _bad_value(x)
            if msg:
                return msg
        return None
    return f"unsupported value type {type(v).__name__}"


def _label_problem(labels: Any, dim: int) -> str | None:
    if not (isinstance(labels, list) and len(labels) == dim and all(isinstance(x, str) for x in labels)):
        return f"must be {dim} strings"
    if not all(_LABEL.fullmatch(x) for x in labels):
        return "labels must be identifiers (letters, digits and '_', not starting with a digit)"
    if len(set(labels)) != len(labels):
        return "labels must be unique"
    clash = sorted(set(labels) & _RESERVED_LABELS)
    if clash:
        return f"{clash} are reserved keys of the result files"
    return None


def _dimension(sec: dict[str, Any], name: str, issues: list[dict[str, Any]]) -> int | None:
    if "dimension" not in sec:
        return None
    v = sec["dimension"]
    if not _is_int(v) or v < 1:
        issues.append(_issue("DIMENSION_INVALID", f"{name}.dimension must be a positive integer", f"{name}.dimension"))
        return None
    return v


def _count(alg: dict[str, Any], key: str, minimum: int, issues: list[dict[str, Any]]) -> None:
    v = _get(alg, key)
    if v is not None and (not _is_int(v) or v < minimum):
        issues.append(_issue("VALUE_INVALID", f"algorithm.{key} must be an integer >= {minimum}", f"algorithm.{key}"))


def _temperature_issues(s: dict[str, Any], sec: str, issues: list[dict[str, Any]]) -> None:
    """The rules of ``util/read_ts.py``: exactly one complete pair, finite, ordered, in range."""
    f = f"algorithm.{sec}"
    if "Tinvspace" in s:
        issues.append(_issue("TEMPERATURE_INVALID", "Tinvspace is deprecated; use bmin/bmax", f"{f}.Tinvspace"))
    set_t = "Tmin" in s or "Tmax" in s
    set_b = "bmin" in s or "bmax" in s
    if set_t == set_b:
        issues.append(_issue("TEMPERATURE_INVALID", f"give either Tmin/Tmax or bmin/bmax in [{sec}]", f))
        return
    lo_key, hi_key = ("bmin", "bmax") if set_b else ("Tmin", "Tmax")
    if lo_key not in s or hi_key not in s:
        issues.append(_issue("TEMPERATURE_INVALID", f"give both {lo_key} and {hi_key} in [{sec}]", f))
        return
    lo, hi = s[lo_key], s[hi_key]
    logscale = s.get("Tlogspace", True)
    if not isinstance(logscale, bool):
        issues.append(_issue("VALUE_INVALID", f"{f}.Tlogspace must be true or false", f"{f}.Tlogspace"))
    if not _is_num(lo) or not _is_num(hi):
        issues.append(_issue("TEMPERATURE_INVALID", f"{lo_key} and {hi_key} must be finite numbers", f))
    elif set_t and (lo <= 0 or hi <= 0):
        issues.append(_issue("TEMPERATURE_INVALID", "Tmin and Tmax must be positive", f))
    elif set_b and (lo < 0 or hi < 0):
        issues.append(_issue("TEMPERATURE_INVALID", "bmin and bmax must be zero or positive", f))
    elif lo > hi:
        issues.append(_issue("TEMPERATURE_INVALID", f"{lo_key} must be <= {hi_key}", f))
    elif set_b and logscale is not False and lo == 0:
        issues.append(_issue("TEMPERATURE_INVALID", "bmin must be > 0 when Tlogspace is true", f))


def _matrix_shape(v: Any) -> tuple[int, int] | None:
    """Shape of a matrix as ``util/read_matrix.py`` reads it ((0, 0) when empty); None when it is not one."""
    if isinstance(v, str):
        try:
            rows = [[float(w) for w in line.split()] for line in v.split("\n") if line.strip()]
        except ValueError:
            return None
    elif isinstance(v, list):
        rows = v
    else:
        return None
    if not rows:
        return (0, 0)
    if not all(isinstance(r, list) and r and all(_is_num(x) for x in r) for r in rows):
        return None
    if len({len(r) for r in rows}) != 1:
        return None
    return (len(rows), len(rows[0]))


def _mapping_issues(runner: Any, alg_dim: int, solver_dim: int, issues: list[dict[str, Any]]) -> None:
    """[runner.mapping] as ``util/mapping.Affine.from_dict`` reads it: x_solver = A x_algorithm + b."""
    mapping = runner.get("mapping") if isinstance(runner, dict) else None
    if mapping is None:
        if alg_dim != solver_dim:
            issues.append(_issue("DIMENSION_MISMATCH", f"algorithm dimension {alg_dim} differs from solver "
                                 f"dimension {solver_dim} but no [runner.mapping] is given", "solver.dimension"))
        return
    if not isinstance(mapping, dict):
        issues.append(_issue("MAPPING_INVALID", "runner.mapping must be a table with A and b", "runner.mapping"))
        return
    a = _matrix_shape(mapping.get("A", []))
    b = _matrix_shape(mapping.get("b", []))
    if a is None:
        issues.append(_issue("MAPPING_INVALID", "runner.mapping.A must be a matrix (list of equal-length rows)",
                             "runner.mapping.A"))
        return
    if b is None or (b != (0, 0) and b[1] != 1):
        issues.append(_issue("MAPPING_INVALID", "runner.mapping.b must be a column vector, e.g. [[0.0], [1.0]]",
                             "runner.mapping.b"))
        return
    if a == (0, 0):
        if b != (0, 0):
            issues.append(_issue("MAPPING_INVALID", "runner.mapping.b needs runner.mapping.A", "runner.mapping.b"))
        elif alg_dim != solver_dim:
            issues.append(_issue("DIMENSION_MISMATCH", "runner.mapping without A keeps the dimension, so "
                                 "algorithm and solver dimensions must agree", "runner.mapping"))
        return
    if a != (solver_dim, alg_dim):
        issues.append(_issue("MAPPING_INVALID", f"runner.mapping.A must be a {solver_dim} x {alg_dim} matrix "
                             "(solver dimension x algorithm dimension)", "runner.mapping.A"))
    elif b != (0, 0) and b[0] != a[0]:
        issues.append(_issue("MAPPING_INVALID", f"runner.mapping.b must have {a[0]} rows", "runner.mapping.b"))


def validate(inp: dict[str, Any], *, for_run: bool = False,
             solvers: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Check an input dictionary (structure of input.toml) without running anything.

    ``for_run=True`` additionally requires the built-in analytical solver or a registered solver
    (entry points ``odatse.solvers`` and ``solvers``).
    """
    issues: list[dict[str, Any]] = []
    if not isinstance(inp, dict):
        return [_issue("INVALID_INPUT", "input must be a JSON object with base/algorithm/solver sections")]
    for sec in ("base", "algorithm", "solver"):
        if not isinstance(inp.get(sec), dict):
            issues.append(_issue("SECTION_MISSING", f"section [{sec}] is required", sec))
    if "runner" in inp and not isinstance(inp["runner"], dict):
        issues.append(_issue("SECTION_MISSING", "section [runner] must be a table", "runner"))
    if issues:
        return issues
    for k in sorted(set(inp) - {"base", "algorithm", "solver", "runner"}, key=str):
        issues.append(_issue("UNKNOWN_SECTION", f"unknown section [{k}]", str(k)))
    _value_issues(inp, "", issues)
    if issues:
        return issues  # the checks below assume well-formed keys and values

    base, alg, solver = inp["base"], inp["algorithm"], inp["solver"]
    # dimensions as resolved by AlgorithmBase (algorithm.dimension or base.dimension) and SolverBase
    # (solver.dimension if present, else base.dimension)
    a_dim, b_dim, s_dim = (_dimension(alg, "algorithm", issues), _dimension(base, "base", issues),
                           _dimension(solver, "solver", issues))
    dim = a_dim or b_dim
    if dim is None and not issues:
        issues.append(_issue("DIMENSION_INVALID", "base.dimension (or algorithm.dimension) must be a positive "
                             "integer", "base.dimension"))
    solver_dim = s_dim if "dimension" in solver else b_dim

    name = alg.get("name")
    spec = ALGORITHMS.get(name) if isinstance(name, str) else None
    if spec is None:
        issues.append(_issue("ALGORITHM_UNKNOWN", f"algorithm.name must be one of {sorted(ALGORITHMS)}",
                             "algorithm.name"))
    else:
        has_mesh = _get(alg, "param.mesh_path") is not None
        # Monte Carlo on a grid (montecarlo.py): mesh_path, or use_grid = true with min/max/num_list
        discrete = name in ("exchange", "pamc") and (has_mesh or _get(alg, "param.use_grid") is True)
        required = list(spec["required"])
        if discrete:
            required.remove("param.step_list")
            if not has_mesh:
                required.append("param.num_list")
            if not (has_mesh and _get(alg, "param.neighborlist_path") is not None):
                required.append("param.radius")  # DiscreteStateSpace builds the neighbour list from it
        for key in required:
            if has_mesh and name in ("mapper", "bayes", "exchange", "pamc") and \
                    key in ("param.min_list", "param.max_list", "param.num_list"):
                continue  # MeshGrid reads the points from mesh_path instead
            if key == "param.step_list" and _get(alg, "param.unit_list") is not None:
                continue  # obsolete alias still accepted by odatse.algorithm.state (prints a WARNING)
            if _get(alg, key) is None:
                issues.append(_issue("KEY_MISSING", f"algorithm.{key} is required for {name}", f"algorithm.{key}"))
        if discrete and _get(alg, "param.radius") is not None and not _is_num(_get(alg, "param.radius")):
            issues.append(_issue("VALUE_INVALID", "algorithm.param.radius must be a finite number",
                                 "algorithm.param.radius"))
        if dim is not None:
            for key in ("param.min_list", "param.max_list", "param.unit_list", "param.step_list",
                        "param.num_list", "minimize.initial_scale_list"):
                v = _get(alg, key)
                if v is not None and not _num_list(v, dim):
                    issues.append(_issue("LIST_INVALID", f"algorithm.{key} must be {dim} finite numbers",
                                         f"algorithm.{key}"))
            ini = _get(alg, "param.initial_list")
            vectors = _vectors(ini, dim) if ini is not None else []
            if vectors is None:
                issues.append(_issue("LIST_INVALID", f"algorithm.param.initial_list must be {dim} finite numbers "
                                     f"or a list of such vectors (one per walker)", "algorithm.param.initial_list"))
            lo, hi = _get(alg, "param.min_list"), _get(alg, "param.max_list")
            if _num_list(lo, dim) and _num_list(hi, dim):
                if not all(a < b for a, b in zip(lo, hi)):
                    issues.append(_issue("BOUNDS_INVALID", "every min_list entry must be < max_list",
                                         "algorithm.param"))
                elif name == "minsearch" and vectors and \
                        not all(a <= x <= b for vec in vectors for a, x, b in zip(lo, vec, hi)):
                    issues.append(_issue("INITIAL_OUT_OF_RANGE", "initial_list must lie inside [min_list, max_list]",
                                         "algorithm.param.initial_list"))
            nl = _get(alg, "param.num_list")
            if _num_list(nl, dim) and not all(_is_int(n) and n >= 1 for n in nl):
                issues.append(_issue("LIST_INVALID", "num_list entries must be positive integers",
                                     "algorithm.param.num_list"))
        if name == "random_search":
            _count(alg, "param.num_points", 1, issues)
        for sec in ("exchange", "pamc"):
            if name != sec:
                continue
            s = alg.get(sec)
            if not isinstance(s, dict):
                if s is not None:
                    issues.append(_issue("VALUE_INVALID", f"algorithm.{sec} must be a table", f"algorithm.{sec}"))
                continue
            _temperature_issues(s, sec, issues)
            _count(alg, f"{sec}.nreplica_per_proc", 1, issues)
        if name == "exchange":
            _count(alg, "exchange.numsteps", 1, issues)
            _count(alg, "exchange.numsteps_exchange", 1, issues)
            _count(alg, "exchange.numsteps_thermalization", 0, issues)
        if name == "pamc" and isinstance(alg.get("pamc"), dict):
            s = alg["pamc"]
            for k in ("numsteps", "numsteps_annealing", "Tnum"):
                _count(alg, f"pamc.{k}", 0, issues)
            _count(alg, "pamc.resampling_interval", 1, issues)
            positive = [k for k in ("numsteps", "numsteps_annealing", "Tnum") if _is_int(s.get(k)) and s[k] > 0]
            if len(positive) != 2:
                issues.append(_issue("SCHEDULE_INVALID", "exactly two of numsteps, numsteps_annealing and Tnum "
                                     f"must be positive in [pamc] (positive: {positive})", "algorithm.pamc"))
        if name == "bayes":
            for k in ("random_max_num_probes", "bayes_max_num_probes", "num_rand_basis"):
                _count(alg, f"bayes.{k}", 0, issues)
        labels = alg.get("label_list")
        if labels is not None and dim is not None:
            msg = _label_problem(labels, dim)
            if msg:
                issues.append(_issue("LIST_INVALID", f"algorithm.label_list: {msg}", "algorithm.label_list"))
        missing = [m for m in spec["needs"] if _missing_module(m)]
        if missing:
            issues.append(_issue("DEPENDENCY_MISSING", f"{name} needs {missing}", "algorithm.name",
                                 f"pip install {' '.join(missing)}"))

    sname = solver.get("name")
    if not isinstance(sname, str) or not sname:
        issues.append(_issue("VALUE_INVALID", "solver.name must be a non-empty string", "solver.name"))
    elif sname == "analytical":
        fn = solver.get("function_name", "quadratics")
        if not isinstance(fn, str) or fn not in ANALYTICAL_FUNCTIONS:
            issues.append(_issue("FUNCTION_UNKNOWN", f"function_name must be one of {list(ANALYTICAL_FUNCTIONS)}",
                                 "solver.function_name"))
        elif solver_dim is None and "dimension" not in solver and "dimension" not in base:
            issues.append(_issue("DIMENSION_INVALID", "the analytical solver reads solver.dimension or "
                                 "base.dimension; give one of them", "solver.dimension"))
        elif solver_dim is not None and ANALYTICAL_FUNCTIONS[fn] not in (None, solver_dim):
            issues.append(_issue("DIMENSION_INVALID", f"{fn} requires a solver dimension of "
                                 f"{ANALYTICAL_FUNCTIONS[fn]} (got {solver_dim})",
                                 "solver.dimension" if "dimension" in solver else "base.dimension"))
    registered = False
    if isinstance(sname, str) and sname and sname != "analytical":
        reg, problems = _registered(solvers)
        if sname in reg:
            registered = True
            _registered_solver_issues(solver, reg[sname]["target"], solver_dim, base, issues)
        elif problems and for_run:
            issues.append(_issue("REGISTRY_INVALID", "; ".join(problems), "solver.name"))
    if (not isinstance(sname, str) or sname != "analytical") and not registered and for_run:
        issues.append(_issue("SOLVER_NOT_RUNNABLE", f"solver {sname!r} cannot be run through this API",
                             "solver.name", "run the built-in 'analytical' solver or a registered solver "
                             "(see registered_solvers in get_schema); run other solvers with your own script"))
    if dim is not None and solver_dim is not None:
        _mapping_issues(inp.get("runner"), dim, solver_dim, issues)

    issues.extend(_path_issues(inp))
    return issues


def _registered_solver_issues(solver: dict[str, Any], target: str, solver_dim: int | None,
                              base: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """[solver] of a registered solver: only name/dimension/params, params matching the signature."""
    for k in sorted(set(solver) - {"name", "dimension", "params"}):
        issues.append(_issue("SOLVER_KEY_UNKNOWN", f"solver.{k} is not used by registered solvers "
                             "(give function arguments in solver.params)", f"solver.{k}"))
    info = _registry.describe(target)
    if "error" in info:
        issues.append(_issue("SOLVER_UNAVAILABLE", info["error"], "solver.name"))
        return
    if solver_dim is None and "dimension" not in solver and "dimension" not in base:
        issues.append(_issue("DIMENSION_INVALID", "the solver reads solver.dimension or base.dimension; give one "
                             "of them", "solver.dimension"))
    elif info["dimension"] is not None and solver_dim is not None and solver_dim != info["dimension"]:
        issues.append(_issue("DIMENSION_INVALID", f"solver {solver['name']!r} requires a solver dimension of "
                             f"{info['dimension']} (got {solver_dim})",
                             "solver.dimension" if "dimension" in solver else "base.dimension"))
    params = solver.get("params", {})
    if not isinstance(params, dict):
        issues.append(_issue("PARAMS_INVALID", "solver.params must be a table", "solver.params"))
        return
    for k, v in params.items():
        if isinstance(v, dict):
            issues.append(_issue("PARAMS_INVALID", f"solver.params.{k}: nested tables are not supported",
                                 f"solver.params.{k}"))
    known = info["params"]
    if info["vector_name"] in params:
        issues.append(_issue("PARAM_UNKNOWN", f"solver.params.{info['vector_name']} would collide with the "
                             "parameter vector x, which ODAT-SE passes as the first argument",
                             f"solver.params.{info['vector_name']}"))
    elif not info["accepts_any_params"]:
        for k in sorted(set(params) - set(known)):
            issues.append(_issue("PARAM_UNKNOWN", f"solver {solver['name']!r} has no parameter {k!r} "
                                 f"(parameters: {sorted(known)})", f"solver.params.{k}"))
    for k in sorted(n for n, p in known.items() if p["required"] and n not in params):
        issues.append(_issue("PARAM_MISSING", f"solver {solver['name']!r} needs solver.params.{k}",
                             f"solver.params.{k}"))


def _path_values(d: Any, prefix: str = "") -> list[tuple[str, str, Any]]:
    """(field, key, value) for every path-bearing key at any depth."""
    out: list[tuple[str, str, Any]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            f = f"{prefix}.{k}" if prefix else str(k)
            if f == "solver.params":
                continue  # arguments of a registered solver function, not ODAT-SE settings
            if k in _PATH_KEYS:
                out.append((f, k, v))
            out.extend(_path_values(v, f))
    return out


def _path_issues(inp: dict[str, Any]) -> list[dict[str, Any]]:
    return [_issue("UNSAFE_PATH", f"{f} must be a non-empty relative path without '..' or control characters", f)
            for f, _, v in _path_values(inp) if not _safe_relpath(v)]


def _containment_issues(workdir: Path) -> list[dict[str, Any]]:
    """Refuse a working directory that contains symbolic links.

    Every path-bearing value is already a relative path without ``..`` (see :func:`_path_issues`), so with
    no symbolic link below ``workdir`` nothing ODAT-SE reads or writes -- named in the input or implied,
    such as ``res.txt``, ``ColorMap.txt``, checkpoint temporaries or ``input.toml`` itself -- can leave it.
    """
    try:
        top = workdir.resolve()
        links = []
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            for n in dirnames + filenames:
                p = Path(dirpath) / n
                if p.is_symlink():
                    links.append(str(p.relative_to(top)))
    except (OSError, RuntimeError, ValueError) as exc:
        return [_issue("UNSAFE_PATH", f"could not inspect the working directory: {exc}")]
    if links:
        return [_issue("UNSAFE_PATH", f"the working directory contains symbolic links: {sorted(links)[:10]}",
                       None, "use a fresh working directory without symbolic links")]
    return []


def _missing_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def _toml_string(s: str) -> str:
    out = []
    for ch in s:
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_key(k: Any) -> str:
    if not isinstance(k, str):
        raise ValueError(f"keys must be strings, got {type(k).__name__}")
    return k if _BARE_KEY.fullmatch(k) else _toml_string(k)


def _toml_value(v: Any) -> str:
    msg = _bad_value(v)
    if msg:
        raise ValueError(msg)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _toml_string(v)
    return "[" + ", ".join(_toml_value(x) for x in v) + "]"


def _toml_loads(text: str) -> Any:
    try:
        import tomllib as parser  # Python >= 3.11
    except ImportError:
        try:
            import tomli as parser  # type: ignore[no-redef]
        except ImportError:
            return None
    return parser.loads(text)


def to_toml(inp: dict[str, Any]) -> str:
    """Serialise an input dictionary (nested tables of scalars and lists) to TOML text.

    Keys are quoted when they are not bare keys and strings are escaped. When a TOML parser is available
    the text is parsed back and must equal ``inp``; otherwise ``ValueError`` is raised.
    """
    lines: list[str] = []

    def emit(table: dict[str, Any], path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        if path:
            lines.append("[" + ".".join(path) + "]")
        for k, v in scalars.items():
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        if path:
            lines.append("")
        for k, v in subs.items():
            emit(v, [*path, _toml_key(k)])

    emit(inp, [])
    text = "\n".join(lines).rstrip() + "\n"
    try:
        back = _toml_loads(text)
    except ValueError as exc:
        raise ValueError(f"generated TOML does not parse: {exc}") from exc
    if back is not None and back != inp:
        raise ValueError("generated TOML does not reproduce the input")
    return text


def render(inp: dict[str, Any], *, solvers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Validate and return the ``input.toml`` text. Nothing is written."""
    issues = validate(inp, solvers=solvers)
    if issues:
        return {"status": "failed", "issues": issues}
    try:
        return {"status": "ok", "input_toml": to_toml(inp)}
    except ValueError as exc:
        return {"status": "failed", "issues": [_issue("SERIALIZATION_FAILED", str(exc))]}


def _open_new(path: Path) -> Any:
    """Open ``path`` for writing (truncating) without following a symbolic link planted at ``path``."""
    if path.is_symlink():
        raise OSError(f"{path.name} is a symbolic link")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    return os.fdopen(fd, "w", encoding="utf-8")


def prepare(inp: dict[str, Any], workdir: str | os.PathLike[str], *,
            solvers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Validate and write ``workdir/input.toml``. Nothing is executed."""
    rendered = render(inp, solvers=solvers)
    if rendered["status"] != "ok":
        return rendered
    wd = Path(workdir)
    try:
        wd.mkdir(parents=True, exist_ok=True)
        issues = _containment_issues(wd)
        if issues:
            return {"status": "failed", "issues": issues}
        path = wd / "input.toml"
        with _open_new(path) as fh:
            fh.write(rendered["input_toml"])
    except OSError as exc:
        return {"status": "failed", "issues": [_issue("WRITE_FAILED", f"could not write input.toml: {exc}")]}
    return {"status": "ok", "files": [str(path)], "input_toml": rendered["input_toml"]}


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# variables that would change which code the run imports or how the interpreter starts
_ENV_DENIED_PREFIXES = ("PYTHON", "LD_", "DYLD_")
_ENV_BASE = ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS")


def check_solver_env(names: Sequence[str]) -> list[str]:
    """The environment variable names in ``names``, each one complete name (``ValueError`` when one is malformed
    or would change how the run's interpreter starts: ``PYTHON*``, ``LD_*``, ``DYLD_*``)."""
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError("solver_env must be a list of variable names, not a single string")
    for n in names:
        if not isinstance(n, str) or not _ENV_NAME.match(n):
            raise ValueError(f"{n!r} is not an environment variable name")
        if n.upper().startswith(_ENV_DENIED_PREFIXES):
            raise ValueError(f"{n!r} cannot be passed to a solver (PYTHON*, LD_* and DYLD_* are refused)")
    return list(dict.fromkeys(names))


def parse_solver_env(text: str) -> list[str]:
    """Names in ``ODATSE_SOLVER_ENV="NAME,NAME,..."`` (``ValueError`` as in :func:`check_solver_env`)."""
    return check_solver_env([n.strip() for n in text.split(",") if n.strip()])


def run(inp: dict[str, Any], workdir: str | os.PathLike[str], *, timeout_s: float = 600.0,
        solvers: Mapping[str, str] | None = None, solver_env: Sequence[str] | None = None) -> dict[str, Any]:
    """Run ODAT-SE with the analytical or a registered solver in a subprocess and return :func:`parse` output.

    The subprocess gets only ``PATH``, ``PYTHONPATH``, ``HOME``, locale and thread-count variables. A registered
    solver that needs more (the path of an external program, a data directory) gets the variables named in
    ``solver_env`` (default: ``ODATSE_SOLVER_ENV="NAME,NAME,..."``) that are set in the caller's environment.
    Their names, not their values, are recorded in ``provenance.solver.env``.

    Exit statuses of a registered-solver run: 3 = the solver raised or returned NaN / -inf / a non-number
    (``SOLVER_FAILED``; set ``runner.ignore_error = true`` to record NaN and continue instead), 4 = the
    solver could not be imported in the subprocess.
    """
    if _seconds(timeout_s) is None:
        return {"status": "failed", "issues": [_issue("TIMEOUT_INVALID", "timeout_s must be a finite positive "
                                                      "number of seconds", "timeout_s")]}
    try:
        extra_env = parse_solver_env(os.environ.get("ODATSE_SOLVER_ENV", "")) if solver_env is None \
            else check_solver_env(solver_env)
    except ValueError as exc:
        return {"status": "failed", "issues": [_issue("SOLVER_ENV_INVALID", str(exc), None,
                                                      "name each variable separately; PYTHON*, LD_* and "
                                                      "DYLD_* cannot be added")]}
    issues = validate(inp, for_run=True, solvers=solvers)
    if issues:
        return {"status": "failed", "issues": issues}
    sname = inp["solver"]["name"]
    entry = None if sname == "analytical" else _registered(solvers)[0].get(sname)
    if sname != "analytical" and entry is None:  # the registry changed since validate()
        return {"status": "failed", "issues": [_issue("SOLVER_NOT_RUNNABLE", f"solver {sname!r} is not registered",
                                                      "solver.name")]}
    target = None if entry is None else entry["target"]
    prep = prepare(inp, workdir, solvers=solvers)
    if prep["status"] != "ok":
        return prep
    wd = Path(workdir)
    env = {k: v for k, v in os.environ.items() if k in _ENV_BASE}
    passed = [n for n in extra_env if n in os.environ and n not in _ENV_BASE] if target else []
    env.update({n: os.environ[n] for n in passed})
    # the subprocess must import the very same odatse package as the caller
    pkg_root = str(Path(__file__).resolve().parent.parent)
    # relative entries would be read against workdir in the subprocess; anchor them here
    env["PYTHONPATH"] = os.pathsep.join([pkg_root, *[os.path.abspath(p) for p in
                                                     env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    env["PYTHONSAFEPATH"] = "1"  # Python >= 3.11: do not put the working directory on sys.path
    module, args = ("odatse", ["input.toml"]) if target is None else \
        ("odatse._api_runner", ["input.toml", "--solver", target])
    # Python 3.9/3.10 ignore PYTHONSAFEPATH: drop the working directory ('' for -c) from sys.path before
    # anything is imported from it, so a file in workdir cannot shadow odatse or the registered module
    command = [sys.executable, "-c", _BOOTSTRAP, module, *args]
    t0 = time.monotonic()
    log_path = wd / "odatse.log"
    try:
        log = _open_new(log_path)
    except OSError as exc:
        return {"status": "failed", "issues": [_issue("WRITE_FAILED", f"could not open odatse.log: {exc}")]}
    with log:
        try:
            proc = subprocess.run(command, cwd=wd, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=float(timeout_s), check=False)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "issues": [_issue("TIMEOUT", f"run exceeded {timeout_s} s", None,
                                                          "reduce the number of steps or increase timeout_s")],
                    "log_tail": _tail(log_path)}
        except OSError as exc:
            return {"status": "failed", "issues": [_issue("RUN_FAILED", f"could not start odatse: {exc}")]}
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        code, msg = {3: ("SOLVER_FAILED", f"solver {sname!r} failed"),
                     4: ("SOLVER_UNAVAILABLE", f"solver {sname!r} could not be imported in the run")}.get(
            proc.returncode, ("RUN_FAILED", f"odatse exited with {proc.returncode}")) if target else \
            ("RUN_FAILED", f"odatse exited with {proc.returncode}")
        return {"status": "failed", "issues": [_issue(code, msg, "solver.name" if code != "RUN_FAILED" else None,
                                                      "see log_tail")],
                "log_tail": _tail(log_path)}
    out = parse(wd)
    out["elapsed_s"] = round(elapsed, 3)
    out["provenance"] = {"api_schema_version": SCHEMA_VERSION, "versions": versions(),
                         "seed": inp["algorithm"].get("seed"), "command": " ".join(["python -m", module, *args])}
    if entry is not None:
        dist = {"distribution": entry["distribution"], "version": entry["version"]} \
            if entry.get("version") else _registry.module_version(target)
        out["provenance"]["solver"] = {"name": sname, "target": target, "source": entry["source"],
                                       "params": inp["solver"].get("params", {}), "env": passed, **dist}
    return out


_BOOTSTRAP = ("import sys; sys.path[:] = [p for p in sys.path if p != '']; "
              "import runpy; m = sys.argv.pop(1); sys.argv[0] = m; runpy.run_module(m, run_name='__main__', "
              "alter_sys=True)")


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
        if len(cols) <= max(fx_col, x_cols.stop - 1):
            raise ValueError(f"{p.name}: incomplete row {line.strip()!r}")
        if not all(math.isfinite(c) for c in cols[x_cols]):
            raise ValueError(f"{p.name}: non-finite value in row {line.strip()!r}")
        if not math.isfinite(cols[fx_col]):
            # inf: a rejected point (for example outside the limitation); NaN: a failed evaluation recorded
            # with runner.ignore_error = true. Neither is a candidate.
            continue
        if best is None or cols[fx_col] < best[fx_col]:
            best = cols
    if best is None:
        return None
    return {"fx": best[fx_col], "x": dict(zip(labels, best[x_cols]))}


def _parse_layout(inp: Any) -> tuple[str, int, list[str]]:
    """(algorithm name, algorithm dimension, labels) of an input.toml; ValueError when malformed."""
    if not isinstance(inp, dict) or not isinstance(inp.get("algorithm"), dict):
        raise ValueError("[algorithm] section missing")
    alg = inp["algorithm"]
    base = inp.get("base") if isinstance(inp.get("base"), dict) else {}
    name = alg.get("name")
    if not isinstance(name, str):
        raise ValueError("algorithm.name missing")
    dim = alg.get("dimension") or base.get("dimension")  # as AlgorithmBase resolves it
    if not _is_int(dim) or dim < 1:
        raise ValueError("dimension is not defined (algorithm.dimension or base.dimension)")
    labels = alg.get("label_list", [f"x{d + 1}" for d in range(dim)])
    msg = _label_problem(labels, dim)
    if msg:
        raise ValueError(f"algorithm.label_list: {msg}")
    return name, dim, labels


def parse(workdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Best result and output file list of a finished run in ``workdir`` (reads ``input.toml`` there)."""
    from .util import toml

    wd = Path(workdir)
    if not wd.is_dir():
        return {"status": "failed", "issues": [_issue("NOT_FOUND", f"working directory not found: {wd}")]}
    issues = _containment_issues(wd)  # before reading anything, input.toml included
    if issues:
        return {"status": "failed", "issues": issues}
    try:
        inp = toml.load(str(wd / "input.toml"))
    except OSError as exc:
        return {"status": "failed", "issues": [_issue("NOT_FOUND", f"input.toml not readable: {exc}")]}
    except ValueError as exc:  # TOML decoding errors of tomllib / tomli / toml are ValueError subclasses
        return {"status": "failed", "issues": [_issue("INPUT_INVALID", f"input.toml does not parse: {exc}")]}
    try:
        name, dim, labels = _parse_layout(inp)
    except ValueError as exc:
        return {"status": "failed", "issues": [_issue("INPUT_INVALID", f"input.toml: {exc}")]}
    issues = _path_issues(inp)
    if issues:
        return {"status": "failed", "algorithm": name, "issues": issues}
    base = inp.get("base") if isinstance(inp.get("base"), dict) else {}
    out_dir = wd / base.get("root_dir", ".") / base.get("output_dir", ".")
    try:
        files = sorted(str(p.relative_to(wd)) for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() \
            else []
    except (OSError, ValueError) as exc:
        return {"status": "failed", "algorithm": name,
                "issues": [_issue("PARSE_FAILED", f"could not list output files: {exc}")]}
    best: dict[str, Any] | None = None
    try:
        kv_file = {"minsearch": "res.txt", "exchange": "best_result.txt", "pamc": "best_result.txt"}.get(name)
        if kv_file and (out_dir / kv_file).exists():
            kv = _read_kv(out_dir / kv_file)
            want = ["fx", *labels]
            bad = [k for k in want if not isinstance(kv.get(k), float) or not math.isfinite(kv[k])]
            if bad:
                raise ValueError(f"{kv_file} is incomplete or not numeric for {bad}")
            best = {"fx": kv["fx"], "x": {k: kv[k] for k in labels}}
        elif name in ("mapper", "random_search"):
            cmap = out_dir / inp["algorithm"].get("colormap", "ColorMap.txt")
            if cmap.exists():
                best = _best_from_table(cmap, dim, slice(0, dim), labels)
        elif name == "bayes" and (out_dir / "BayesData.txt").exists():
            # columns: step, best x..., best fx, action x..., action fx (fx written as in the run)
            best = _best_from_table(out_dir / "BayesData.txt", 1 + dim, slice(1, 1 + dim), labels)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        return {"status": "failed", "algorithm": name, "files": files,
                "issues": [_issue("PARSE_FAILED", f"could not parse result files: {exc}")]}
    if best is None:
        return {"status": "failed", "algorithm": name, "files": files,
                "issues": [_issue("RESULT_MISSING", f"no result file for {name} in {out_dir}")]}
    return {"status": "ok", "algorithm": name, "best": best, "files": files}
