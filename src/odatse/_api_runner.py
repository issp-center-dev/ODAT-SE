# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Fixed runner used by :func:`odatse.api.run` for a registered solver (see :mod:`odatse.solver.registry`).

``python -m odatse._api_runner input.toml --solver package.module:attribute``

It does what ``odatse input.toml`` does (``odatse.initialize``, ``choose_algorithm``, ``Runner``,
``Algorithm.main``) with :class:`odatse.solver.function.Solver` evaluating ``f(x, **solver.params)``.

Evaluation rules
----------------
* The return value must convert to a float. ``+inf`` is passed on (ODAT-SE treats it as a rejected point,
  like a violated ``[runner.limitation]``).
* An exception in ``f``, a value that is not a number, NaN or ``-inf`` becomes a ``RuntimeError``. With
  ``runner.ignore_error = true`` the Runner records NaN for that point and continues; otherwise the run
  stops and this process exits with status 3 (``SOLVER_FAILED`` in :func:`odatse.api.run`).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from typing import Any, Optional, Sequence

EXIT_SOLVER_FAILED = 3
EXIT_SETUP_FAILED = 4


class SolverFailure(RuntimeError):
    """A registered solver raised or returned an unusable value."""


def _safe_str(obj: Any) -> str:
    try:
        return str(obj)[:500]
    except Exception:  # noqa: BLE001
        return f"<unprintable {type(obj).__name__}>"


def _wrap(f: Any, name: str, params: dict[str, Any]) -> Any:
    def evaluate(x: Any) -> float:
        try:
            value = f(x.copy(), **copy.deepcopy(params))
        except Exception as exc:  # noqa: BLE001 - reported with the point that failed
            raise SolverFailure(f"solver {name!r} raised {type(exc).__name__} at x={list(x)}: "
                                f"{_safe_str(exc)}") from exc
        try:
            fx = float(value)
        except Exception as exc:  # noqa: BLE001 - __float__ of a returned object may raise anything
            raise SolverFailure(f"solver {name!r} returned a {type(value).__name__} at x={list(x)} that is not a "
                                f"number ({type(exc).__name__}: {_safe_str(exc)})") from exc
        if math.isnan(fx) or fx == -math.inf:
            raise SolverFailure(f"solver {name!r} returned {fx} at x={list(x)}")
        return fx

    return evaluate


def _solver_class() -> Any:
    import os

    from odatse.solver.function import Solver

    class RegisteredSolver(Solver):
        """``function.Solver`` that returns to the previous directory even when the function fails."""

        def evaluate(self, x: Any, args: tuple = ()) -> float:
            cwd = os.getcwd()
            try:
                return super().evaluate(x, args)
            finally:
                os.chdir(cwd)

    return RegisteredSolver


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m odatse._api_runner")
    parser.add_argument("inputfile")
    parser.add_argument("--solver", required=True, help="package.module:attribute")
    args = parser.parse_args(argv)

    import odatse
    from odatse import exception
    from odatse.solver.registry import resolve

    try:
        f = resolve(args.solver)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot import solver {args.solver}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_SETUP_FAILED
    try:
        info, run_mode = odatse.initialize([args.inputfile])
        name = info.solver.get("name", "?")
        params = info.solver.get("params", {}) or {}
        if not isinstance(params, dict):
            raise exception.InputError("solver.params must be a table")
        alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])
        solver = _solver_class()(info)
        solver.set_function(_wrap(f, name, dict(params)))
        runner = odatse.Runner(solver, info)
        alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
        alg.main()
    except SolverFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SOLVER_FAILED
    except exception.Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
