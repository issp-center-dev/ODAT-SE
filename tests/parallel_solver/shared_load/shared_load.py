# A solver that reads files through odatse.util.io, in its constructor
# (scope "job") and inside evaluate() (scope "solver"). FAIL_RANK makes the
# process with that global rank look for a file that does not exist inside
# evaluate(): only the root of the read (the controller of the group) reads,
# so the job completes when FAIL_RANK is a worker and ends with an error on
# every process of the group, without a hang, when it is a controller.

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

import numpy as np
import odatse
import odatse.util.io as oio
from odatse.algorithm import choose_algorithm

FAIL_RANK = int(os.environ.get("FAIL_RANK", "-1"))


class Solver(odatse.solver.SolverBase):
    def __init__(self, info):
        super().__init__(info)
        # read once on rank 0, received by every process of the job
        self.reference = oio.loadtxt("reference.txt")
        assert self.reference.shape == (2,)
        # the algorithm changes into its output directory during the run:
        # resolve the paths read inside evaluate() now
        self.ref_path = os.path.abspath("reference.txt")
        self.missing_path = os.path.abspath("missing.txt")

    def evaluate(self, xs, args):
        # re-read inside the evaluation: the controller reads, its workers
        # receive the same data
        path = self.missing_path if odatse.mpi.rank() == FAIL_RANK else self.ref_path
        scale = oio.loadtxt(path, scope="solver")
        return float(np.sum(scale * (xs - self.reference) ** 2))


def main():
    info, run_mode = odatse.initialize()
    with odatse.mpi.fail_together():
        os.makedirs(info.base.get("output_dir", "./output"), exist_ok=True)
        solver = Solver(info)
        runner = odatse.Runner(solver, info)
    alg = choose_algorithm(info.algorithm["name"]).Algorithm(info, runner, run_mode=run_mode)
    alg.main()


if __name__ == "__main__":
    try:
        main()
    except odatse.mpi.OtherAlgorithmProcessError:
        sys.exit(0)   # another process failed and reports its error
