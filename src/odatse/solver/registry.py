# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Solvers registered by the operator, selected by ``solver.name`` in :mod:`odatse.api` and the MCP server.

A registered solver is a Python callable ``f(x, **params) -> float`` named by a target string
``"package.module:attribute"``. Callers (and AI agents) only choose a registered **name** and give
``solver.params``; they never send code.

Sources
-------
* the entry-point group ``odatse.solvers`` of installed distributions, for example in ``pyproject.toml``::

      [project.entry-points."odatse.solvers"]
      my_loss = "mypkg.solvers:my_loss"

* an explicit mapping ``{name: target}`` given by the caller of :mod:`odatse.api`; the MCP server builds
  it from ``ODATSE_MCP_SOLVERS="my_loss=mypkg.solvers:my_loss,other=mypkg.other:f"``. An explicit entry
  replaces an entry point of the same name.

Optional attributes of the callable
-----------------------------------
``odatse_description`` (str)
    one-line description shown by ``schema()``.
``odatse_dimension`` (int or None)
    the solver dimension the function requires (None: any).
"""

from __future__ import annotations

import functools
import importlib
import inspect
import re
from typing import Any, Callable, Mapping

ENTRY_POINT_GROUP = "odatse.solvers"
RESERVED_NAMES = frozenset({"analytical", "function"})

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTTED = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_TARGET = re.compile(rf"^{_DOTTED}:{_DOTTED}$")


def name_problem(name: Any) -> str | None:
    """Why ``name`` cannot be a registered solver name (None when it can)."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return f"solver name {name!r} must be an identifier (letters, digits and '_')"
    if name in RESERVED_NAMES:
        return f"solver name {name!r} is reserved by ODAT-SE"
    return None


def target_problem(target: Any) -> str | None:
    """Why ``target`` is not a ``"package.module:attribute"`` string (None when it is)."""
    if not isinstance(target, str) or not _TARGET.fullmatch(target):
        return f"target {target!r} must have the form 'package.module:attribute'"
    return None


def parse_spec(text: str) -> dict[str, str]:
    """Parse ``"name=module:attr,name2=module2:attr2"``; ``ValueError`` describes the first problem."""
    out: dict[str, str] = {}
    for item in (s.strip() for s in text.split(",")):
        if not item:
            continue
        name, sep, target = (s.strip() for s in item.partition("="))
        if not sep:
            raise ValueError(f"{item!r}: expected name=package.module:attribute")
        msg = name_problem(name) or target_problem(target)
        if msg:
            raise ValueError(msg)
        if name in out:
            raise ValueError(f"solver name {name!r} is given twice")
        out[name] = target
    return out


def entry_point_solvers() -> dict[str, dict[str, Any]]:
    """``{name: {"target", "distribution", "version"}}`` from the ``odatse.solvers`` entry points.

    A name registered with different targets by several distributions gets ``"ambiguous": [targets]``;
    :func:`registry` reports it and leaves it out unless an explicit mapping names it.
    """
    import importlib.metadata as md

    try:
        eps = md.entry_points(group=ENTRY_POINT_GROUP)  # Python >= 3.10
    except TypeError:  # pragma: no cover - Python 3.9
        eps = md.entry_points().get(ENTRY_POINT_GROUP, [])
    out: dict[str, dict[str, Any]] = {}
    for ep in eps:
        dist = getattr(ep, "dist", None)
        entry = {
            "target": ep.value,
            "distribution": getattr(dist, "name", None) if dist is not None else None,
            "version": getattr(dist, "version", None) if dist is not None else None,
        }
        prev = out.get(ep.name)
        if prev is None:
            out[ep.name] = entry
        elif prev["target"] != entry["target"]:
            targets = prev.get("ambiguous", [prev["target"]])
            if entry["target"] not in targets:
                targets = [*targets, entry["target"]]
            out[ep.name] = {**prev, "ambiguous": targets}
    return out


def registry(solvers: Mapping[str, str] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Effective registry ``{name: {"target", "source", "distribution", "version"}}`` and problems found.

    Entry points come first; ``solvers`` (explicit, from the operator) replaces entries of the same name.
    """
    problems: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    try:
        eps = entry_point_solvers()
    except Exception as exc:  # noqa: BLE001 - broken metadata must not break the API
        eps = {}
        problems.append(f"could not read the {ENTRY_POINT_GROUP} entry points: {exc}")
    for name, ep in eps.items():
        if ep.get("ambiguous"):
            problems.append(f"entry point {name!r} is registered with different targets {ep['ambiguous']}; "
                            "name the one to use in the explicit mapping")
            continue
        msg = name_problem(name) or target_problem(ep["target"])
        if msg:
            problems.append(f"entry point {name!r}: {msg}")
            continue
        out[name] = {"source": "entry_point", **ep}
    if solvers is not None:
        if not isinstance(solvers, Mapping):
            problems.append("solvers must be a mapping of name to 'package.module:attribute'")
        else:
            for name, target in solvers.items():
                msg = name_problem(name) or target_problem(target)
                if msg:
                    problems.append(msg)
                    continue
                out[name] = {"source": "explicit", "target": target, "distribution": None, "version": None}
    return out, problems


def resolve(target: str) -> Callable[..., Any]:
    """Import ``"package.module:attribute"`` and return the callable (``ValueError`` when it is not one)."""
    msg = target_problem(target)
    if msg:
        raise ValueError(msg)
    module_name, _, attr = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"{target} is not callable")
    return obj


@functools.lru_cache(maxsize=64)
def describe(target: str) -> dict[str, Any]:
    """Import the target and describe it: description, dimension, parameters (from the signature).

    Returns ``{"error": ...}`` instead of raising when the target cannot be used.
    """
    try:
        f = resolve(target)
    except Exception as exc:  # noqa: BLE001 - an operator's module may raise anything on import
        return {"error": f"cannot import {target}: {type(exc).__name__}: {exc}"}
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError) as exc:
        return {"error": f"cannot read the signature of {target}: {exc}"}
    plist = list(sig.parameters.values())
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if not plist or plist[0].kind not in positional:
        return {"error": f"{target} must take the parameter vector x as its first positional argument"}
    params: dict[str, Any] = {}
    var_kw = False
    for p in plist[1:]:
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw = True
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif p.kind == inspect.Parameter.POSITIONAL_ONLY:
            if p.default is inspect.Parameter.empty:
                return {"error": f"{target}: positional-only parameter {p.name!r} cannot be given in "
                                 "solver.params"}
        else:
            entry: dict[str, Any] = {"required": p.default is inspect.Parameter.empty}
            if p.default is not inspect.Parameter.empty and _jsonable(p.default):
                entry["default"] = p.default
            if p.annotation is not inspect.Parameter.empty:
                entry["annotation"] = p.annotation if isinstance(p.annotation, str) else \
                    getattr(p.annotation, "__name__", str(p.annotation))
            params[p.name] = entry
    desc = getattr(f, "odatse_description", None)
    dim = getattr(f, "odatse_dimension", None)
    if desc is None:
        doc = inspect.getdoc(f) or ""
        desc = doc.strip().splitlines()[0] if doc.strip() else ""
    if not isinstance(desc, str):
        return {"error": f"{target}.odatse_description must be a string"}
    if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim < 1):
        return {"error": f"{target}.odatse_dimension must be a positive integer or None"}
    vector = plist[0].name if plist[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD else None
    return {"description": desc, "dimension": dim, "params": params, "accepts_any_params": var_kw,
            "vector_name": vector}


def _jsonable(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, str)):
        return True
    if isinstance(v, float):
        return v == v and v not in (float("inf"), float("-inf"))
    if isinstance(v, (list, tuple)):
        return all(_jsonable(x) for x in v)
    return False


def module_version(target: str) -> dict[str, Any]:
    """Distribution name and version that provide the target's top-level package (None when unknown)."""
    import importlib.metadata as md

    top = target.partition(":")[0].split(".")[0]
    try:
        dists = md.packages_distributions().get(top, [])  # Python >= 3.10
    except AttributeError:  # pragma: no cover - Python 3.9
        dists = []
    for d in dists:
        try:
            return {"distribution": d, "version": md.version(d)}
        except md.PackageNotFoundError:
            continue
    return {"distribution": None, "version": None}
