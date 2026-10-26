"""A failure inside PAMC _run() on one rank only must reach the phase
consensus in AlgorithmBase.run() on every rank, instead of leaving the other
ranks waiting in a collective of their own.

_run() used to end with an algcomm barrier.  The failing rank skipped it and
went straight to the Allreduce of the consensus, the others waited in the
barrier, and the job hung.
"""
# sys.path and odatse.mpi.setup() are handled by conftest.py
import numpy as np
import pytest

import odatse
import odatse.solver.function
import odatse.algorithm.pamc as pamc


def _make_pamc(workdir):
    inp = {
        "base": {"dimension": 1, "output_dir": str(workdir / "output")},
        "algorithm": {
            "name": "pamc",
            "seed": 1,
            "param": {"min_list": [-5.0], "max_list": [5.0], "step_list": [0.5]},
            "pamc": {
                "numsteps_annealing": 2,
                "Tnum": 3,
                "bmin": 0.0,
                "bmax": 1.0,
                "Tlogspace": False,
                "nreplica_per_proc": 2,
                "fix_num_walkers": True,
                "resampling_interval": 1,
            },
        },
        "solver": {"name": "function"},
        "runner": {},
    }
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(lambda x: float(np.sum(x * x)))
    runner = odatse.Runner(solver, info)
    return pamc.Algorithm(info, runner)


def test_failure_on_one_rank_does_not_hang(tmp_path, monkeypatch):
    """A failure on one rank after the last collective of _run() (here: while
    closing the output files) must make that rank re-raise its own error and
    the others raise OtherAlgorithmProcessError (this test would time out
    under mpirun while _run() ended with an algcomm barrier the failing rank
    never reached).

    The failure is injected after the annealing loop on purpose: PAMC gathers
    across the ranks at every temperature, so a rank failing inside the loop
    (e.g. in the objective) leaves the others in one of those gathers, which
    no change at the end of _run() can fix."""
    if not (odatse.mpi.enabled() and odatse.mpi.algsize() > 1):
        pytest.skip("needs more than one algorithm rank (run under mpirun)")
    monkeypatch.chdir(tmp_path)
    alg = _make_pamc(tmp_path)

    close_writer = alg._close_writer

    def fails_on_rank0(writers):
        close_writer(writers)
        if odatse.mpi.algrank() == 0:
            raise ValueError("closing the output failed on rank 0 only")

    monkeypatch.setattr(alg, "_close_writer", fails_on_rank0)

    # call the phases directly: main() turns OtherAlgorithmProcessError into
    # sys.exit(0), which would leave this rank out of the allgather below
    alg.prepare()
    try:
        alg.run()
        outcome = "returned"
    except ValueError:
        outcome = "own"
    except odatse.mpi.OtherAlgorithmProcessError:
        outcome = "other"

    # gather before asserting, so that a wrong outcome on one rank cannot
    # leave the others waiting in this collective
    outcomes = odatse.mpi.algcomm().allgather(outcome)
    assert outcomes == ["own"] + ["other"] * (odatse.mpi.algsize() - 1)
