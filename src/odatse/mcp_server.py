# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""MCP (Model Context Protocol) server for ODAT-SE: a thin wrapper around :mod:`odatse.api`.

Start with ``odatse-mcp`` (stdio) after ``pip install "ODAT-SE[mcp]"``.

* Read-only tools (always on): ``get_schema``, ``versions``, ``validate_input``, ``render_input_toml``.
* ``run_calculation`` is **disabled by default**. Enable it with ``ODATSE_MCP_ALLOW_RUN=1``. It runs the
  built-in analytical solver or a solver registered by the operator, in a subprocess with a timeout,
  inside a server-created directory under ``ODATSE_MCP_WORKDIR`` (default ``./odatse-mcp-runs``).
  Clients never pass file-system paths or code; runs are addressed by the returned ``run_id``.
* Registered solvers (:mod:`odatse.solver.registry`): the ``odatse.solvers`` entry points of installed
  packages plus ``ODATSE_MCP_SOLVERS="name=package.module:attribute,..."``, read when the server starts.
  A client selects one with ``solver.name`` and passes its arguments in ``solver.params``.
* A registered solver's subprocess sees only ``PATH``, ``PYTHONPATH``, ``HOME``, locale and thread-count
  variables, plus those named in ``ODATSE_SOLVER_ENV="NAME,NAME,..."`` (read when the server starts).
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from . import api

INSTRUCTIONS = """\
ODAT-SE: inverse-problem / parameter-search framework (Nelder-Mead, grid, random, replica exchange,
population annealing, Bayesian optimisation). The input is the input.toml structure as JSON.
1. get_schema() for algorithms, required keys, the built-in analytical test functions and the solvers
   the operator registered (registered_solvers: description, dimension, params).
2. validate_input(input) before anything else; fix every issue.
3. render_input_toml(input) gives the input.toml text for the user to run with their own solver.
4. run_calculation(input) runs the 'analytical' solver (solver.function_name) or a registered solver
   (solver.name = its name, arguments in solver.params), only if the server operator enabled runs
   (ODATSE_MCP_ALLOW_RUN=1); then get_result(run_id).
"""

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _base_dir() -> Path:
    return Path(os.environ.get("ODATSE_MCP_WORKDIR", "odatse-mcp-runs")).resolve()


def _run_allowed() -> bool:
    return os.environ.get("ODATSE_MCP_ALLOW_RUN", "") == "1"


def _solvers() -> dict[str, str]:
    """Explicit registrations from ``ODATSE_MCP_SOLVERS`` (``SystemExit`` with the reason when malformed)."""
    from .solver.registry import parse_spec

    try:
        return parse_spec(os.environ.get("ODATSE_MCP_SOLVERS", ""))
    except ValueError as exc:
        raise SystemExit(f"ODATSE_MCP_SOLVERS: {exc}") from exc


def _solver_env() -> list[str]:
    """Names from ``ODATSE_SOLVER_ENV`` (``SystemExit`` with the reason when malformed)."""
    try:
        return api.parse_solver_env(os.environ.get("ODATSE_SOLVER_ENV", ""))
    except ValueError as exc:
        raise SystemExit(f"ODATSE_SOLVER_ENV: {exc}") from exc


def build_server() -> Any:
    solvers = _solvers()
    solver_env = _solver_env()
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit('the MCP server needs the optional extra: pip install "ODAT-SE[mcp]"') from exc

    server = FastMCP("odatse", instructions=INSTRUCTIONS)

    @server.tool()
    def get_schema() -> dict[str, Any]:
        """Algorithms (required/optional keys, result files), analytical functions and registered solvers."""
        return api.schema(solvers=solvers)

    @server.tool()
    def versions() -> dict[str, str]:
        """Versions of ODAT-SE, NumPy, SciPy and PHYSBO (record them for reproducibility)."""
        return api.versions()

    @server.tool()
    def validate_input(input: dict[str, Any], for_run: bool = False) -> dict[str, Any]:
        """Check an input (structure of input.toml) without running. Returns {valid, issues}."""
        issues = api.validate(input, for_run=for_run, solvers=solvers)
        return {"valid": not issues, "issues": issues}

    @server.tool()
    def render_input_toml(input: dict[str, Any]) -> dict[str, Any]:
        """Validate and return the input.toml text (no file is written)."""
        return api.render(input, solvers=solvers)

    @server.tool()
    def run_calculation(input: dict[str, Any], timeout_s: int | float = 300.0) -> dict[str, Any]:
        """Run the analytical solver or a registered solver (disabled unless ODATSE_MCP_ALLOW_RUN=1).

        Returns {status, run_id, best, files, elapsed_s, provenance} or {status: failed, issues}.
        """
        if not _run_allowed():
            return {"status": "failed", "issues": [{
                "code": "RUN_DISABLED", "field": None,
                "message": "run_calculation is disabled on this server",
                "hint": "the server operator can enable it with ODATSE_MCP_ALLOW_RUN=1"}]}
        try:
            seconds = float(timeout_s) if api._is_num(timeout_s) else None
        except OverflowError:
            seconds = None
        if seconds is None:
            return {"status": "failed", "issues": [{
                "code": "TIMEOUT_INVALID", "field": "timeout_s", "hint": None,
                "message": "timeout_s must be a finite number of seconds"}]}
        timeout_s = min(max(seconds, 1.0), 3600.0)
        run_id = uuid.uuid4().hex
        result = api.run(input, _base_dir() / run_id, timeout_s=timeout_s, solvers=solvers,
                         solver_env=solver_env)
        return {"run_id": run_id, **result}

    @server.tool()
    def get_result(run_id: str) -> dict[str, Any]:
        """Best result and output file list of a previous run_calculation (by run_id)."""
        if not _RUN_ID.match(run_id) or not (_base_dir() / run_id).is_dir():
            return {"status": "failed", "issues": [{"code": "NOT_FOUND", "message": f"unknown run_id {run_id!r}",
                                                    "field": "run_id", "hint": None}]}
        return {"run_id": run_id, **api.parse(_base_dir() / run_id)}

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
