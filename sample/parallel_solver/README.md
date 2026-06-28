# parallel_solver — MPI Parallel Solver Sample

This sample demonstrates how to write a custom solver that exploits
**two levels of MPI parallelism** simultaneously within ODAT-SE:

| Level | Controlled by | Typical use |
|-------|--------------|-------------|
| Algorithm-layer MPI (`nalg` processes) | `odatse.mpi.algcomm` | Distribute the search space across independent evaluations |
| Solver-layer MPI (`nsolve` processes per group) | `odatse.mpi.solcomm` | Distribute heavy work (e.g., matrix operations) within one evaluation |

Total parallelism = `nalg × nsolve`.

## Problem

Find the integer seed `s ∈ {1, …, 20}` that minimises the **average largest
singular value** of `nmats` random matrices of size `matsize × matsize`:

```
minimise  (1/nmats) Σ_{i=1}^{nmats} σ_max( A_i(s) )
```

where each `A_i(s)` is drawn from `numpy.random.default_rng(seed=s)`.

## Prerequisites

| Package | Version | Notes |
|---------|---------|-------|
| `odatse` | current | install from repo root: `pip install -e .` |
| `mpi4py` | ≥ 3.1.0 | `pip install mpi4py` |
| `numpy` | ≥ 1.14 | installed with odatse |
| MPI runtime | any | OpenMPI, MPICH, … |

## Files

```
parallel_solver/
├── parallel_solver.py   # Main script; defines ParallelSolver and the run loop
├── input.toml           # ODAT-SE configuration file
└── README.md
```

## Running

The total number of MPI processes must equal `nalg × nsolve`:

```bash
mpirun -np 6 python3 parallel_solver.py -m 3 -n 2
```

This launches **6 MPI processes** partitioned as:

```
┌──────────────────── COMM_WORLD (6 processes) ─────────────────────┐
│  algrank=0          algrank=1          algrank=2                   │
│  rank 0 | rank 1    rank 2 | rank 3    rank 4 | rank 5             │
│ (sol 0) | (sol 1)  (sol 0) | (sol 1)  (sol 0) | (sol 1)           │
│ ←── solcomm ──→    ←── solcomm ──→    ←── solcomm ──→             │
│ ←────────────────── algcomm ──────────────────→                    │
│                  (ranks 0, 2, 4 only)                              │
└────────────────────────────────────────────────────────────────────┘
```

Each `solcomm` group controller (`solrank==0`) is also a member of `algcomm`
and receives a subset of seeds from the mapper algorithm. Workers (`solrank!=0`)
wait for work from `_algorithm.py`'s worker loop and participate in the
matrix computation via `solcomm` collectives.

### Single-process run (no MPI)

```bash
ODATSE_NOMPI=1 python3 parallel_solver.py
```

All parallelism is disabled; the script runs serially for testing purposes.

## Input file

### `[base]`

| Key | Description |
|-----|-------------|
| `output_dir` | Directory for output files |
| `dimension` | Number of search parameters |

### `[solver]`

`name = "custom"` selects `ParallelSolver` defined in this script.

### `[algorithm.param]`

| Key | Description |
|-----|-------------|
| `min_list` | Lower bound of the search range |
| `max_list` | Upper bound of the search range |
| `num_list` | Number of grid points (seeds to evaluate) |

`nalg` and `nsolve` are passed via command-line arguments (`-m`/`-n`), not
through `input.toml`.

## Execution flow

```
python3 parallel_solver.py -m nalg -n nsolve
    │
    ├─ odatse.initialize() — parses args, calls odatse.mpi.setup(nalg, nsolve)
    │
    └─ alg.main()
          │
          ├─ [algorithm processes, solrank==0]
          │     mapper sweeps seeds 1..20, splits across algcomm
          │     runner.submit(x) → Bcast MSG_EVALUATE + x to solcomm
          │                      → calls solver.evaluate(x)
          │
          ├─ [solver workers, solrank>0]
          │     worker loop: receive MSG_EVALUATE + x → call solver.evaluate(x)
          │
          └─ ParallelSolver.evaluate(seeds)   [called on all solcomm ranks]
                │
                └─ _compute(seeds)
                      for each seed:
                        solrank==0: generate nmats matrices
                        solcomm.bcast(matrices)
                        each rank: SVD on its slice
                        solcomm.allreduce(partial sums) / nmats
                        → average largest singular value

algcomm.allgather(local_best)  # collect best result from each solver group
```

## Adapting to your own solver

`ParallelSolver` is a subclass of `odatse.solver.SolverBase`. To adapt this
pattern to a real solver:

1. **Replace `_testfunc`** with your actual computation.
2. **Replace the matrix broadcast** in `_compute` with whatever data your
   solver workers need.
3. **Read solver parameters** from `info.solver` (populated from the
   `[solver]` section of `input.toml`) instead of hardcoding them.
4. **Adjust `nalg` and `nsolve`** to match your hardware and workload:
   - Increase `nsolve` when a single evaluation is expensive enough to benefit
     from intra-group MPI parallelism.
   - Increase `nalg` to explore more of the parameter space concurrently.

## Notes

- `odatse.mpi` is the canonical module for accessing MPI communicators and
  parallelism parameters.
- Worker processes (`solrank != 0`) are driven by `_algorithm.py`'s worker
  loop; they call `evaluate()` directly after receiving a broadcast from the
  controller. The return value of `evaluate()` on workers is discarded.
- The global optimum is collected via `algcomm.allgather` after `alg.main()`
  returns, so only algorithm processes (`run_on_algorithm() == True`) participate.
