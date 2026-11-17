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
* ``run_calculation`` is **disabled by default**. Enable it with ``ODATSE_MCP_ALLOW_RUN=1``. It only
  runs the built-in analytical solver, in a subprocess with a timeout, inside a server-created directory
  under ``ODATSE_MCP_WORKDIR`` (default ``./odatse-mcp-runs``). Clients never pass file-system paths;
  runs are addressed by the returned ``run_id``.
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
1. get_schema() for algorithms, required keys and the built-in analytical test functions.
2. validate_input(input) before anything else; fix every issue.
3. render_input_toml(input) gives the input.toml text for the user to run with their own solver.
4. run_calculation(input) runs only the built-in 'analytical' solver and only if the server operator
   enabled it (ODATSE_MCP_ALLOW_RUN=1); then get_result(run_id).
"""

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _base_dir() -> Path:
    return Path(os.environ.get("ODATSE_MCP_WORKDIR", "odatse-mcp-runs")).resolve()


def _run_allowed() -> bool:
    return os.environ.get("ODATSE_MCP_ALLOW_RUN", "") == "1"


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit('the MCP server needs the optional extra: pip install "ODAT-SE[mcp]"') from exc

    server = FastMCP("odatse", instructions=INSTRUCTIONS)

    @server.tool()
    def get_schema() -> dict[str, Any]:
        """Supported algorithms (required/optional keys, result files) and built-in analytical functions."""
        return api.schema()

    @server.tool()
    def versions() -> dict[str, str]:
        """Versions of ODAT-SE, NumPy, SciPy and PHYSBO (record them for reproducibility)."""
        return api.versions()

    @server.tool()
    def validate_input(input: dict[str, Any], for_run: bool = False) -> dict[str, Any]:
        """Check an input (structure of input.toml) without running. Returns {valid, issues}."""
        issues = api.validate(input, for_run=for_run)
        return {"valid": not issues, "issues": issues}

    @server.tool()
    def render_input_toml(input: dict[str, Any]) -> dict[str, Any]:
        """Validate and return the input.toml text (no file is written)."""
        issues = api.validate(input)
        if issues:
            return {"status": "failed", "issues": issues}
        return {"status": "ok", "input_toml": api.to_toml(input)}

    @server.tool()
    def run_calculation(input: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
        """Run the built-in analytical solver (disabled unless ODATSE_MCP_ALLOW_RUN=1).

        Returns {status, run_id, best, files, elapsed_s, provenance} or {status: failed, issues}.
        """
        if not _run_allowed():
            return {"status": "failed", "issues": [{
                "code": "RUN_DISABLED", "field": None,
                "message": "run_calculation is disabled on this server",
                "hint": "the server operator can enable it with ODATSE_MCP_ALLOW_RUN=1"}]}
        timeout_s = float(min(max(timeout_s, 1.0), 3600.0))
        run_id = uuid.uuid4().hex
        result = api.run(input, _base_dir() / run_id, timeout_s=timeout_s)
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
