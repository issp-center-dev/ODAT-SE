import sys, os, time, argparse
import numpy as np
from mpi4py import MPI
import odatse
from odatse.algorithm import choose_algorithm
import threadpoolctl

#initialize nalg*nsolve mpi processes with ranks 0...nalg*nsolve-1
#split the global communicator into nalg subcommunicators with nsolve processes each
#each process has its own thread pool (via threadpoolctl)
#process 0 in each communicator is the controller
#each of the nalg subcommunicators obtains its share of tasks

#solve the following problem in parallel:
#find random seed with minimal average largest singular value
#nmats matrices of size matsize*matsize are generated for each seed
#each algcomm rank receives a seed which generates a set of matrices
#each solcomm rank receives a subset of matrices

#for each matrix, svd is computed using thread parallelization
#each solcomm rank returns a sum of the largest singular values for its subset of matrices
#each algcomm rank reduces the sum of largest singular values from its solcomm ranks and returns the minimum
#the global minimum is returned as the optimal solution

class ParallelSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.opt_x = None
        self.opt_fx = np.inf

        self.nmats=kwargs["nmats"]
        self.matsize=kwargs["matsize"]

        if odatse.mpi.rank()==0:
            print(f"nalg: {odatse.mpi.algsize()}")
            print(f"nsolve: {odatse.mpi.solsize()}")
            print(f"nthreads: {odatse.mpi.solthreads()}")
        odatse.mpi.comm().barrier()

    def _testfunc(self, mats):
        with threadpoolctl.threadpool_limits(limits=odatse.mpi.solthreads()):
            return np.sum([np.max(np.linalg.svd(mat, compute_uv=False)) for mat in mats])

    def _compute(self, seeds): # called by all solcomm ranks after broadcast
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

    def evaluate(self, xs, args): # only called by algcomm ranks (solrank==0)
        seeds = xs.astype(int)

        if odatse.mpi.solrank() == 0:
            odatse.mpi.solcomm().bcast((seeds, args), root=0) # wake up workers in this solcomm
            print(f"color {odatse.mpi.color()}, seeds: {list(seeds)}")

        results = self._compute(seeds) # algcomm rank also calls compute

        if odatse.mpi.solrank() == 0:
            print(f"color {odatse.mpi.color()}, results: {list(results)}")
        
        best_x = np.argmin(results)
        best_fx = results[best_x]
        if best_fx < self.opt_fx:
            self.opt_x = xs[best_x]
            self.opt_fx = best_fx
        return results

parser=argparse.ArgumentParser()
parser.add_argument('-m','--nalg', help='# of processes for search algorithm', type=int, default=1)
parser.add_argument('-n','--nsolve', help='# of processes for solver', type=int, default=1)
parser.add_argument('-t','--nthreads', help='# of threads', type=int, default=1)
args=parser.parse_args()

assert args.nalg*args.nsolve == odatse.mpi.comm().size

nmats=50
matsize=1000

sys.argv = ["script.py", "input.toml", "--init", f"--nalg={args.nalg}", f"--nsolve={args.nsolve}", f"--nthreads={args.nthreads}"]
info, run_mode = odatse.initialize()
output_dir = info.base.get("output_dir", "./output")
os.makedirs(output_dir, exist_ok=True)
print(odatse.mpi.size(), odatse.mpi.algsize(), odatse.mpi.solsize(), odatse.mpi.solthreads())
solver = ParallelSolver(info, nmats=nmats, matsize=matsize)
runner = odatse.Runner(solver, info)
alg_module = choose_algorithm(info.algorithm["name"])
alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
time0 = time.time()
result = alg.main()
odatse.mpi.comm().barrier()
time1 = time.time()
elapsed_time = time1 - time0

if odatse.mpi.algrank() is not None:
    solver.opt_fx, solver.opt_x = min(odatse.mpi.algcomm().allgather((solver.opt_fx, solver.opt_x)), key=lambda x: x[0])

odatse.mpi.comm().barrier()

if odatse.mpi.rank() == 0:
    print(f"\nopt_x={solver.opt_x}")
    print(f"opt_fx={solver.opt_fx}")
    print(f"time: {elapsed_time}s")