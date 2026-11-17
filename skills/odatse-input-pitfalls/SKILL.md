---
name: odatse-input-pitfalls
description: Use when an ODAT-SE (odatse) run stops at input reading or behaves unexpectedly — "ERROR: section ... does not appear in input", "ERROR: dimension is not defined", "algorithm.param.min_list is not defined", "num_list is not defined", "num_points is not defined", "step_list not specified", "WARNING: unit_list is obsolete", "both Tmin/Tmax and bmin/bmax are defined", "neither Tmin/Tmax nor bmin/bmax are defined", "Two of 'numsteps', 'numsteps_annealing', and 'Tnum' should be positive", "ERROR: Unknown solver", "himmelblau works only with dimension=2", "Warning: out of range", or a minsearch that never moves.
---

# ODAT-SE input pitfalls

Each entry gives the message or symptom, the cause, and the fix. Source locations refer to `src/odatse/`.

## Structure

| Message | Cause → fix |
|---|---|
| `ERROR: section base/algorithm/solver does not appear in input` | All three tables are required (`_info.py`). `[runner]` is optional. |
| `ERROR: dimension is not defined` | Set `base.dimension` (or `algorithm.dimension`). List lengths must equal it. |
| `ERROR: Unknown solver (...)` | The `odatse` command only runs `solver.name = "analytical"` (`_main.py`). For your own solver, write a script (see `odatse-practice`). |
| `himmelblau works only with dimension=2` | `himmelblau` needs dimension 2 and `linear_regression_test` needs dimension 3 (`solver/analytical.py`). |

## Search domain (`[algorithm.param]`)

| Message / symptom | Cause → fix |
|---|---|
| `min_list ... is not defined` / `lengths of min_list and max_list do not match` | Region domain (`domain/region.py`): both lists are required, and both must have length `dimension`. |
| `num_list is not defined` | `mapper` and `bayes` use a grid (`domain/meshgrid.py`). Give `num_list`, or give `mesh_path` (relative to `root_dir`). |
| `num_points is not defined` | `random_search` needs `num_points`. |
| `step_list not specified` / `WARNING: unit_list is obsolete` | `exchange` and `pamc` need the Monte Carlo step widths `step_list`. `unit_list` is still read as an alias, but it is obsolete (`algorithm/state.py`). |
| minsearch: `Warning: out of range: [...]` and it does not move | Points outside the open box `(min, max)` return `inf` without calling the solver (`algorithm/min_search.py`). Keep `initial_list` strictly inside, and make the initial simplex (`minimize.initial_scale_list`, default 0.25 per axis) small enough to stay inside. |
| minsearch stops early with a poor value | Nelder-Mead is local. Tighten `minimize.xatol` / `fatol` (default 1e-4), raise `maxiter` / `maxfev`, or start several runs, or use `exchange` / `pamc` first. |

## Temperatures (`exchange`, `pamc`)

| Message | Cause → fix |
|---|---|
| `both Tmin/Tmax and bmin/bmax are defined` / `neither ... are defined` | Give exactly one pair (`util/read_ts.py`). `Tlogspace` (default true) spaces them logarithmically. With `bmin`, it must be > 0 when `Tlogspace` is true. |
| `Two of 'numsteps', 'numsteps_annealing', and 'Tnum' should be positive` | `pamc` schedule: set exactly two of them (`algorithm/pamc.py`). |
| `Tinvspace is deprecated` | Use `bmin` / `bmax` instead. |

## Process and files

- Algorithms change into `output/<rank>` (`AlgorithmBase.run`) and `function.Solver` into its work directory. Run each analysis in its own process and its own directory.
- `odatse.api.validate` checks all of the above before running. It also rejects absolute paths or paths with `..` in `root_dir`, `output_dir`, `mesh_path` and similar keys (`UNSAFE_PATH`).
