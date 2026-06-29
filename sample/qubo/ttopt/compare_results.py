#!/usr/bin/env python3

import os
import pandas as pd

results_path = "./qubo_results.csv"
sol_dir = "./output_qubo_mats_sol"

# Best (minimum) objective value found by ODAT-SE for each instance.
# Aggregate from the flat per-trial results file (which has an explicit
# header: f, dim, instance, min_params, min_f, time) instead of parsing
# the MultiIndex "qubo_results_agg.csv" by fixed row/column positions.
df = pd.read_csv(results_path)
best = df.groupby(["f", "dim", "instance"])["min_f"].min().reset_index()


def read_solution_value(filepath):
    """Read the objective value (the last numeric line) from a result file."""
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
    return float(lines[-1])


results = []

for _, row in best.iterrows():
    f = row['f']
    dim = int(row['dim'])
    instance = int(row['instance'])
    min_value = row['min_f']

    # get reference solution
    filename = f"{f}_{dim}_{instance}_result.txt"
    filepath = os.path.join(sol_dir, filename)
    sol_value = read_solution_value(filepath)

    # calculate relative error
    if sol_value != 0:
        rel_error = abs((sol_value - min_value) / sol_value)
    else:
        rel_error = float('inf') if min_value != 0 else 0.0

    results.append({
        'f': f,
        'dim': dim,
        'instance': instance,
        'min_value': min_value,
        'sol_value': sol_value,
        'difference': min_value - sol_value,
        'rel_error': rel_error
    })

    clipped_error = rel_error if abs(rel_error) >= 1e-5 else 0.0

    print(f"{f}_d{dim}_i{instance}: best={min_value:10.4e}, sol={sol_value:10.4e}, rel_err={clipped_error:10.4e}")

print(f"\nTotal comparisons: {len(results)}")

print("\nRelative error range by problem type and dimension:")
grouped = {}
for r in results:
    key = (r['f'], r['dim'])
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(r['rel_error'])
for (f, dim) in sorted(grouped.keys()):
    errors = grouped[(f, dim)]
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)

    max_err_display = max_err if max_err >= 1e-5 else 0.0
    avg_err_display = avg_err if avg_err >= 1e-5 else 0.0

    print(f"{f}_d{dim}: max={max_err_display:10.4e}, avg={avg_err_display:10.4e}")
