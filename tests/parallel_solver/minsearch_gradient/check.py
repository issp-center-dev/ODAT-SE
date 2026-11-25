# Checks the output of the successful run of do.sh (case "converged").
#
#   python check.py OUTPUT_DIR ref.txt
#
# For every walker (one per solver group, --nalg 2):
#  * res.txt: fx and x within the tolerances of ref.txt (the known minimum
#    of the objective in gradient_solver.py), and likewise the overall
#    OUTPUT_DIR/res.txt;
#  * History_Gradient.txt exists (use_gradient = true was honoured) with more
#    than the initial row, and its first row is the initial point with the
#    exact objective value and gradient there: both components, i.e. the
#    share computed by the worker at the same x as well as the controller's;
#  * History_FunctionCall.txt has no finite-difference probe (an evaluation
#    that differs from another in exactly one coordinate by less than 1e-6),
#    so the gradient came from gradient() and not from scipy's differences.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_solver import objective, objective_gradient  # noqa: E402

# the rows of initial_list in input.toml, one per walker
INITIAL = [[3.0, 1.5], [-2.0, -3.0]]


def read_res(path):
    d = {}
    with open(path) as fp:
        for line in fp:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    d[k.strip()] = float(v)
                except ValueError:
                    pass
    return d


def read_rows(path):
    with open(path) as fp:
        return [np.array([float(v) for v in L.split()])
                for L in fp if L.strip() and not L.startswith("#")]


failures = []


def check(cond, msg):
    print(("ok: " if cond else "FAILED: ") + msg)
    if not cond:
        failures.append(msg)


def check_res(path, ref):
    if not os.path.isfile(path):
        check(False, f"{path} exists")
        return
    res = read_res(path)
    for key in ("fx", "x1", "x2"):
        if key not in res:
            check(False, f"{key} in {path}")
            continue
        tol = ref["tol_fx"] if key == "fx" else ref["tol_x"]
        diff = abs(res[key] - ref[key])
        check(diff <= tol, f"{path}: {key} = {res[key]} (ref {ref[key]}, "
                           f"|diff| = {diff:.3e}, tol {tol:.1e})")


def main(outdir, reffile):
    ref = read_res(reffile)
    check_res(os.path.join(outdir, "res.txt"), ref)
    for k, x0 in enumerate(INITIAL):
        wdir = os.path.join(outdir, str(k))
        check_res(os.path.join(wdir, "res.txt"), ref)

        gpath = os.path.join(wdir, "History_Gradient.txt")
        if not os.path.isfile(gpath):
            check(False, f"{gpath} exists (use_gradient = true)")
        else:
            rows = read_rows(gpath)
            check(len(rows) >= 2, f"{gpath}: {len(rows)} rows, more than the initial one")
            if rows:
                # columns: No, x1, x2, fx, grad_x1, grad_x2
                r = rows[0]
                x0 = np.array(x0)
                check(np.allclose(r[1:3], x0, rtol=0, atol=1e-12),
                      f"{gpath}: first row at the initial point {x0.tolist()} (got {r[1:3].tolist()})")
                check(np.isclose(r[3], objective(x0), rtol=1e-10, atol=1e-12),
                      f"{gpath}: f(x0) = {r[3]} (expected {objective(x0)})")
                check(np.allclose(r[4:6], objective_gradient(x0), rtol=1e-10, atol=1e-12),
                      f"{gpath}: gradient at x0 = {r[4:6].tolist()} "
                      f"(expected {objective_gradient(x0).tolist()}, both solver ranks' shares)")

        fpath = os.path.join(wdir, "History_FunctionCall.txt")
        if not os.path.isfile(fpath):
            check(False, f"{fpath} exists")
            continue
        xs = [r[1:3] for r in read_rows(fpath)]
        probes = 0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                d = np.abs(xs[i] - xs[j])
                if np.count_nonzero(d) == 1 and np.max(d) < 1e-6:
                    probes += 1
        check(probes == 0, f"{fpath}: {probes} finite-difference probe(s) among {len(xs)} evaluations")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
