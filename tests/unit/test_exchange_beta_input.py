# sys.path and odatse.mpi.setup() are handled by conftest.py
from pathlib import Path

import numpy as np
import pytest

import odatse
import odatse.mpi as mpi
import odatse.solver.function
from odatse.algorithm.exchange import Algorithm
from odatse.util.separateT import (
    _read_result_T_entries,
    _compute_temperature_statistics,
)


def _run_exchange(workdir, temperature_range):
    inp = {
        "base": {"dimension": 2, "output_dir": str(workdir / "output")},
        "algorithm": {
            "name": "exchange",
            "seed": 12345,
            "param": {
                "min_list": [-5.0, -5.0],
                "max_list": [5.0, 5.0],
                "step_list": [0.5, 0.5],
            },
            "exchange": {
                "numsteps": 400,
                "numsteps_exchange": 10,
                "nreplica_per_proc": 3,
                "Tlogspace": True,
                **temperature_range,
            },
        },
        "solver": {"name": "function"},
        "runner": {},
    }
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(lambda x: float(np.sum(np.asarray(x) ** 2)))
    runner = odatse.Runner(solver, info)
    alg = Algorithm(info, runner)
    alg.main()
    return alg


def _beta_of_result_T(path):
    """Read the ``# beta = v`` / ``# T = v`` header of a result_T file as beta."""
    with open(path) as f:
        label, _, value = f.readline().lstrip("#").split()
    value = float(value)
    return value if label == "beta" else 1.0 / value


@pytest.mark.parametrize(
    "temperature_range",
    [{"Tmin": 0.5, "Tmax": 2.0}, {"bmin": 0.5, "bmax": 2.0}],
    ids=["T", "beta"],
)
def test_fx_txt_is_the_same_for_T_and_beta_input(tmp_path, temperature_range):
    """fx.txt must list beta in its first column and integrate log(Z/Z0)
    towards higher beta whether the temperatures were given as Tmin/Tmax or
    as bmin/bmax.  With bmin/bmax the first column used to hold 1/beta and
    log(Z/Z0) was integrated in the wrong direction (issue #133)."""
    # Under mpirun every rank has its own tmp_path; the ranks must share one
    # output directory for separateT to find every rank's result.txt.
    if mpi.enabled() and mpi.algsize() > 1:
        workdir = Path(mpi.algcomm().bcast(str(tmp_path), root=0))
    else:
        workdir = tmp_path

    alg = _run_exchange(workdir, temperature_range)
    if mpi.algrank() != 0:
        return  # fx.txt is written by the root rank only

    output_dir = workdir / "output"
    rows = np.loadtxt(output_dir / "fx.txt", ndmin=2)
    betas = np.sort(alg.betas)
    assert rows.shape[0] == len(betas)
    np.testing.assert_allclose(rows[:, 0], betas)

    # Each row must describe the result_T file written for that beta: the
    # mean of f(x) over the production samples, and log(Z/Z0) accumulated
    # from the samples at the previous (lower) beta.
    files = {}
    for index in range(len(betas)):
        path = output_dir / f"result_T{index}.txt"
        files[int(np.argmin(np.abs(betas - _beta_of_result_T(path))))] = path
    assert sorted(files) == list(range(len(betas)))

    logZ = 0.0
    for i, beta in enumerate(betas):
        samples = _read_result_T_entries(files[i])
        mean, _, _, _ = _compute_temperature_statistics(
            samples, alg.numsteps_thermalization, dbeta=0.0)
        assert rows[i, 1] == pytest.approx(mean)
        assert rows[i, 4] == pytest.approx(logZ)
        if i + 1 < len(betas):
            _, _, dlogZ, _ = _compute_temperature_statistics(
                samples, alg.numsteps_thermalization, dbeta=betas[i + 1] - beta)
            logZ += dlogZ

    # For f(x) = |x|^2, <f> = 1/beta and log Z(beta) = log(pi/beta) + const,
    # so both must fall from beta = 0.5 to beta = 2.
    assert rows[0, 1] > rows[-1, 1]
    assert rows[-1, 4] < rows[0, 4]
