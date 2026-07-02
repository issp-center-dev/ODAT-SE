# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os, time, argparse
import numpy as np
from mpi4py import MPI
import odatse
from odatse.algorithm import choose_algorithm

# Parallel solver sample demonstrating two-level MPI parallelism:
#   nalg  algorithm processes each evaluate a disjoint subset of seeds (algcomm)
#   nsolve solver processes per group share the SVD computation within one evaluation (solcomm)
#
# Problem: find the integer seed minimising the average largest singular value
# of nmats random matrices of size matsize x matsize.


def _require_mpi() -> None:
    """Raise if MPI is disabled. This sample uses comm/solcomm collectives."""
    if not odatse.mpi.enabled():
        raise RuntimeError(
            "This sample requires MPI (do not set ODATSE_NOMPI=1); "
            "run with mpirun/mpiexec."
        )


class ParallelSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)
        _require_mpi()

        self.opt_x = None
        self.opt_fx = np.inf

        self.nmats=kwargs["nmats"]
        self.matsize=kwargs["matsize"]

        if odatse.mpi.rank()==0:
            print(f"nalg: {odatse.mpi.algsize()}")
            print(f"nsolve: {odatse.mpi.solsize()}")
        odatse.mpi.comm().barrier()

    def _testfunc(self, mats):
        return np.sum([np.max(np.linalg.svd(mat, compute_uv=False)) for mat in mats])

    def _compute(self, seeds): # called by all solcomm ranks
        results = []
        for seed in seeds:
            if odatse.mpi.solrank() == 0:
                prng = np.random.default_rng(seed=int(seed))
                mats = [prng.random(size=(self.matsize, self.matsize)) for _ in range(self.nmats)]
            else:
                mats = None
            mats = odatse.mpi.solcomm().bcast(mats, root=0)
            mats = np.array_split(mats, odatse.mpi.solsize())[odatse.mpi.solrank()]
            results.append(self._testfunc(mats))
        odatse.mpi.solcomm().barrier()
        results = odatse.mpi.solcomm().allreduce(np.asarray(results), op=MPI.SUM)
        results /= self.nmats
        return results

    def evaluate(self, xs, args):
        seeds = xs.astype(int)

        if odatse.mpi.solrank() == 0:
            print(f"algrank: {odatse.mpi.algrank()}, seeds: {list(seeds)}")

        results = self._compute(seeds)

        if odatse.mpi.solrank() == 0:
            print(f"algrank: {odatse.mpi.algrank()}, results: {list(results)}")

            best_x = np.argmin(results)
            best_fx = results[best_x]
            if best_fx < self.opt_fx:
                self.opt_x = xs[best_x]
                self.opt_fx = best_fx
        return results

def main():
    _require_mpi()

    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--nalg', help='# of processes for search algorithm', type=int, default=1)
    parser.add_argument('-n','--nsolve', help='# of processes for solver', type=int, default=1)
    args=parser.parse_args()

    assert args.nalg*args.nsolve == odatse.mpi.comm().size

    argv = ["input.toml", "--init", f"--nalg={args.nalg}", f"--nsolve={args.nsolve}"]
    info, run_mode = odatse.initialize(argv)

    if odatse.mpi.rank() == 0:
        print(f"total mpi: {odatse.mpi.size()}")

    nmats = info.solver["param"].get("nmats", 50)
    matsize = info.solver["param"].get("matsize", 1000)

    output_dir = info.base.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    solver = ParallelSolver(info, nmats=nmats, matsize=matsize)
    runner = odatse.Runner(solver, info)
    alg_module = choose_algorithm(info.algorithm["name"])
    alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
    time0 = time.time()
    result = alg.main()
    odatse.mpi.comm().barrier()
    time1 = time.time()
    elapsed_time = time1 - time0

    if odatse.mpi.run_on_algorithm():
        opt_fx, opt_x = min(
            odatse.mpi.algcomm().allgather( (solver.opt_fx, solver.opt_x) ),
            key=lambda x: x[0])

    odatse.mpi.comm().barrier()

    if odatse.mpi.rank() == 0:
        print(f"\nopt_x={opt_x}")
        print(f"opt_fx={opt_fx}")
        print(f"time: {elapsed_time:.6f}s")

if __name__ == "__main__":
    main()
