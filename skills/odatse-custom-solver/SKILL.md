---
name: odatse-custom-solver
description: Use when connecting your own forward problem (a Python loss function, a wrapper around an external simulation such as a DFT, Hartree-Fock or tight-binding code) to ODAT-SE so that odatse.api.run or the odatse-mcp tool run_calculation can use it — writing f(x, **params), registering it with the odatse.solvers entry point or ODATSE_MCP_SOLVERS, passing solver.params, or debugging SOLVER_NOT_RUNNABLE, SOLVER_UNAVAILABLE, SOLVER_FAILED, PARAM_UNKNOWN, PARAM_MISSING, PARAMS_INVALID, SOLVER_KEY_UNKNOWN or REGISTRY_INVALID.
---

# Registering your own solver

`odatse input.toml` only knows the built-in `analytical` solver. To make your own forward problem selectable by name, from `odatse.api` or from an AI agent through `odatse-mcp`, **register** a Python function. The caller then picks it with `solver.name` and gives its arguments in `solver.params`. No code is ever sent in the input.

## 1. Write the function

```python
# mypkg/solvers.py
import numpy as np

def hubbard_fit(x, data="ms.csv", L=16):
    """Squared misfit between measured and computed m_s(T) for U = x[0]."""
    ...                                   # run or call your simulation here
    return float(loss)                    # a finite float; float("inf") rejects the point

hubbard_fit.odatse_description = "fit U to m_s(T) data"   # optional, shown by get_schema
hubbard_fit.odatse_dimension = 1                            # optional: required solver dimension
```

- The first positional parameter receives `x`, a NumPy vector of length `solver.dimension` (or `base.dimension`), after `[runner.mapping]` and `[runner.limitation]` are applied. It is a copy, so changing it has no effect on the search.
- Every other parameter can be set from `solver.params`. Parameters without a default are required (`PARAM_MISSING`). Names that are not parameters are refused (`PARAM_UNKNOWN`) unless the function takes `**kwargs`; the name of the first parameter (the vector) is always refused unless it is positional-only (`def f(x, /, **kw)`). Values are TOML/JSON scalars or lists; nested tables and `null` are refused. Each call gets a fresh copy of the params. Params are not ODAT-SE settings: a param called `output_dir` or `colormap` is passed to your function as given and is not checked as a path.
- Return value: any value `float()` accepts. `+inf` marks a rejected point, as a violated `[runner.limitation]` does. An exception, NaN, `-inf` or a value that is not a number stops the run (`SOLVER_FAILED`, with the message and `x` in `log_tail`). With `[runner] ignore_error = true` the point is recorded as NaN instead and the search continues; NaN rows are not candidates for the best result.
- ODAT-SE changes into `output/<rank>` before every call. Write scratch files there or in a temporary directory, and use absolute paths for input data. The directory is restored after each call even if the function changes it or fails.
- Keep importing the module cheap and free of side effects: it is imported in the calling process to read the signature (`get_schema`, `validate`) and again in the run's subprocess.

## 2. Register it

Either an entry point in the package that provides the function (`pyproject.toml`):

```toml
[project.entry-points."odatse.solvers"]
hubbard_fit = "mypkg.solvers:hubbard_fit"
```

or an explicit mapping from whoever runs the API or the server:

```python
odatse.api.run(inp, workdir, solvers={"hubbard_fit": "mypkg.solvers:hubbard_fit"})
```

```bash
ODATSE_MCP_SOLVERS="hubbard_fit=mypkg.solvers:hubbard_fit" ODATSE_MCP_ALLOW_RUN=1 odatse-mcp
```

- Names are identifiers; `analytical` and `function` are reserved. Targets have the form `package.module:attribute`.
- An explicit entry replaces an entry point of the same name. If two installed distributions register the same name with different targets, the name is left out (listed in `registry_problems`) until an explicit entry chooses one. A malformed `ODATSE_MCP_SOLVERS` stops the server at start-up.
- The module must be importable by the same Python in the run's subprocess (installed, or on `PYTHONPATH`; relative entries are made absolute). The run's working directory is removed from the import path, so files there cannot replace your module or ODAT-SE.

## 3. Use it

```json
{"base": {"dimension": 1, "output_dir": "output"},
 "solver": {"name": "hubbard_fit", "params": {"data": "/data/ms.csv", "L": 16}},
 "algorithm": {"name": "minsearch", "seed": 1,
               "param": {"min_list": [0.5], "max_list": [6.0], "initial_list": [2.0]}}}
```

1. `get_schema()` → `registered_solvers` lists each name with its target, description, dimension and parameters (`required`, `default`). An entry with `error` cannot be used (for example the module does not import); `registry_problems` lists malformed registrations.
2. `validate_input(input, for_run=True)` checks the params against the signature and the dimension.
3. `run_calculation(input)` (MCP, opt-in) or `odatse.api.run(...)` runs `python -m odatse._api_runner input.toml --solver <target>` in a subprocess.
4. `provenance.solver` records the name, target, source (`entry_point` or `explicit`), params and the distribution and version that provide the module (`null` when unknown, for example a module on `PYTHONPATH`).

## Security

Registering a function is a decision by the operator: anything that can call the API or the MCP server can run it with any params that pass the checks above. The params reach your code as given, so validate paths, sizes and ranges inside the function, and do not register functions that run shell commands built from params. Leave `ODATSE_MCP_ALLOW_RUN` unset when clients should only validate and render inputs.

## Errors

| Code | Meaning |
|---|---|
| `SOLVER_NOT_RUNNABLE` | `solver.name` is neither `analytical` nor registered. |
| `REGISTRY_INVALID` | A registration is malformed (also listed in `registry_problems`). |
| `SOLVER_UNAVAILABLE` | The target does not import, is not callable, has a bad `odatse_dimension`, or does not take `x` first; or it imports here but not in the run's subprocess. |
| `SOLVER_KEY_UNKNOWN` | `[solver]` of a registered solver has keys other than `name`, `dimension`, `params`. |
| `PARAM_UNKNOWN` / `PARAM_MISSING` / `PARAMS_INVALID` | `solver.params` does not match the function. |
| `DIMENSION_INVALID` | The solver dimension differs from `odatse_dimension`, or none is given. |
| `SOLVER_FAILED` | The function raised or returned NaN, `-inf` or a non-number during the run. |
