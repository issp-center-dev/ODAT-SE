---
name: odatse-practice
description: Use when setting up an ODAT-SE (odatse) analysis — choosing an algorithm (minsearch, mapper, random_search, exchange, pamc, bayes), writing input.toml, connecting your own forward solver as a Python function (script or registered solver), running `odatse input.toml`, reading res.txt / ColorMap.txt / best_result.txt / BayesData.txt, or using the odatse.api functions and the odatse-mcp server from a tool or an AI agent.
---

# ODAT-SE in practice

ODAT-SE searches the parameters `x` of a forward problem, minimising a scalar `f(x)` that the solver returns (for example an R-factor or a loss).
The input is a TOML file with `[base]`, `[solver]`, `[algorithm]` (with `[algorithm.param]` and an algorithm section) and an optional `[runner]`.

## 1. Choose an algorithm

| Goal | `algorithm.name` | Key settings | Main result |
|---|---|---|---|
| Local minimum from a good start | `minsearch` (Nelder-Mead) | `param.min_list/max_list/initial_list`, `[algorithm.minimize]` | `output/res.txt` |
| Map f on a grid / landscape | `mapper` | `param.num_list` (or `mesh_path`) | `output/ColorMap.txt` |
| Random points in the box | `random_search` | `param.num_points` | `output/ColorMap.txt` |
| Global search / sampling at temperatures | `exchange` (replica exchange MC) | `param.step_list`, `[algorithm.exchange]` numsteps, numsteps_exchange, Tmin/Tmax | `output/best_result.txt` |
| Sampling, free energy | `pamc` (population annealing) | `param.step_list`, two of numsteps / numsteps_annealing / Tnum | `output/best_result.txt`, `fx.txt` |
| Expensive f, few evaluations, on a grid | `bayes` (needs PHYSBO) | `param.num_list`, `[algorithm.bayes]` random_max_num_probes, bayes_max_num_probes | `output/BayesData.txt` |

`odatse.api.schema()` (or the MCP tool `get_schema`) lists the required and optional keys for each algorithm.

## 2. Minimal input (built-in test function)

```toml
[base]
dimension = 2
output_dir = "output"

[solver]
name = "analytical"
function_name = "himmelblau"     # see odatse.api.schema(); himmelblau needs dim 2, linear_regression_test dim 3

[algorithm]
name = "minsearch"
seed = 12345

[algorithm.param]
min_list = [-6.0, -6.0]
max_list = [6.0, 6.0]
initial_list = [0.0, 0.0]
```

Run it with `odatse input.toml`. The official versions are in `sample/analytical/<algorithm>/`.

## 3. Your own forward solver

The `odatse` command only knows `solver.name = "analytical"`. For your own problem, write a short script:

```python
import numpy as np, odatse
from odatse.solver.function import Solver

info = odatse.Info.from_file("input.toml")
solver = Solver(info)
solver.set_function(lambda x: my_loss(x))      # x: np.ndarray of length dimension -> float
runner = odatse.Runner(solver, info)
alg = odatse.algorithm.choose_algorithm(info.algorithm["name"]).Algorithm(info, runner)
result = alg.main()
```

`function.Solver` changes into its work directory for every evaluation, and every algorithm changes into `output/` while running. Run ODAT-SE in its own process; do not run it inside a long-lived server process that other code shares.

To run your function through `odatse.api.run` or the MCP tool `run_calculation`, register it as `f(x, **params)` and select it with `solver.name` (skill `odatse-custom-solver`).

## 4. JSON API and MCP server (tools, agents)

- `odatse.api`: `schema()`, `validate(input)`, `render(input)` (returns the `input.toml` text), `prepare(input, workdir)` (writes `input.toml`), `run(input, workdir, timeout_s=...)` (built-in analytical solver or a registered solver, in a subprocess), `parse(workdir)` and `versions()`. The input is the `input.toml` structure as a dict. All but `parse` and `versions` take `solvers={name: "package.module:attribute"}`; entry points in the group `odatse.solvers` are always included.
- `odatse-mcp` (`pip install "ODAT-SE[mcp]"`) provides the tools `get_schema`, `versions`, `validate_input`, `render_input_toml`, `run_calculation` and `get_result`.
  - `run_calculation` is off unless the server is started with `ODATSE_MCP_ALLOW_RUN=1`.
  - Runs are written under `ODATSE_MCP_WORKDIR` and addressed by `run_id`.
  - Registered solvers: the `odatse.solvers` entry points plus `ODATSE_MCP_SOLVERS="name=package.module:attribute,..."`. `get_schema` lists them under `registered_solvers` with their parameters.
- For a solver that is not registered, use `validate_input` and `render_input_toml`, then run your own script (section 3).

## 5. Record for reproducibility

ODAT-SE version (`odatse --version`), the full `input.toml` including `algorithm.seed`, the solver and its version, and the result files.
With MPI, results also depend on the number of processes (replicas per process).
