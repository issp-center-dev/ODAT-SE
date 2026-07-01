Two-level MPI parallelization of the solver
===========================================

Introduction
------------

This tutorial shows how to write a custom solver that exploits two levels of
MPI parallelism within the ODAT-SE framework. The sample files are located in
``sample/parallel_solver`` relative to the root of the repository; the main
script is named ``parallel_solver.py``.

Two levels of parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow of ODAT-SE has two stages: a search *algorithm* proposes candidate
points in the parameter space, and a *solver* evaluates the objective function
at those points. ODAT-SE can parallelize both stages with MPI at the same time:

- **Algorithm-layer parallelism** (``nalg`` processes): the search space is
  distributed across independent evaluations.
- **Solver-layer parallelism** (``nsolve`` processes per group): the work of a
  single evaluation is distributed within a group of processes.

The total number of MPI processes is ``nalg × nsolve``.

The number of processes for each layer is given on the command line with
``--nalg`` and ``--nsolve``; ``odatse.initialize()`` forwards them to
``odatse.mpi.setup()``. For example,

.. code:: bash

    mpirun -np 6 python3 parallel_solver.py --nalg 3 --nsolve 2

runs with 3 algorithm processes and 2 solver processes per group, 6 MPI ranks
in total.

Thread-level parallelism inside each solver process (for example the BLAS
routines called by NumPy) is controlled separately through the
``OMP_NUM_THREADS`` environment variable; ODAT-SE itself does not manage
threads.

.. note::

   Earlier versions of ODAT-SE controlled the number of threads per solver
   process from within the framework (through a ``--nthreads`` command-line
   option and an ``odatse.mpi`` thread accessor). That mechanism has been
   removed. Set the thread count with the ``OMP_NUM_THREADS`` environment
   variable instead, for example ``export OMP_NUM_THREADS=2`` before launching
   the job.

How the two layers are set up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ODAT-SE runs under MPI, ``odatse.mpi.setup(nalg=..., nsolve=...)`` splits
the global communicator (``MPI_COMM_WORLD``) into ``nalg`` solver
subcommunicators (``solcomm``), each with ``nsolve`` processes. The process
with rank 0 in each ``solcomm`` acts as the group *controller* and participates
in the search algorithm; the controllers together form the algorithm
subcommunicator (``algcomm``). Because the controller is the lowest rank of each
group, global rank 0 is always a controller.

The ``odatse.mpi`` module provides the following accessors. The global-layer
ones and ``enabled()`` are available immediately; the solver- and
algorithm-layer ones require ``setup()`` to have been called.

- ``odatse.mpi.comm()`` / ``size()`` / ``rank()``: the global communicator, its
  size, and this process's rank in it.
- ``odatse.mpi.solcomm()`` / ``solsize()`` / ``solrank()``: the solver
  subcommunicator, its size (``nsolve``), and this process's rank in it.
- ``odatse.mpi.algcomm()`` / ``algsize()`` / ``algrank()``: the algorithm
  subcommunicator, its size (``nalg``), and this process's rank in it.
  ``algcomm()`` returns ``None`` on solver-worker processes; ``algsize()`` and
  ``algrank()`` return the values of the group controller (broadcast to the
  workers), so they can be used to identify the group on any process.
- ``odatse.mpi.run_on_algorithm()``: ``True`` on the group controllers
  (``solrank() == 0``), ``False`` on the solver workers.
- ``odatse.mpi.enabled()``: whether MPI is available (``False`` when the
  environment variable ``ODATSE_NOMPI=1`` is set).

Master-worker execution
~~~~~~~~~~~~~~~~~~~~~~~~~

Within each solver group a master-worker scheme keeps the controller and the
workers synchronized. This is handled entirely by the framework, so the solver
author does not have to write a worker loop.

- The controller (``solrank() == 0``) runs the algorithm's ``prepare()``,
  ``run()``, and ``post()`` as usual. Whenever the algorithm evaluates a
  candidate ``x``, ``Runner.submit()`` broadcasts ``x`` and the extra ``args``
  to the whole solver group and then calls the solver's ``evaluate(x, args)``.
- The workers (``solrank() > 0``) sit in a loop inside the framework, receive
  the broadcast ``x`` / ``args``, and call the same ``evaluate(x, args)``. Their
  return value is discarded.

In other words, ``evaluate(x, args)`` is invoked on **every** process of a
solver group with the **same** ``x`` and ``args``. The solver body is then free
to distribute the work of that single evaluation across the group using
``solcomm`` collectives. When the algorithm finishes, the framework signals the
workers to leave their loop.

Custom solver example
~~~~~~~~~~~~~~~~~~~~~~~

As a toy problem we look for the integer seed in ``{1, …, 20}`` that minimises
the average largest singular value of ``nmats`` random matrices of size
``matsize × matsize``. The solver is a subclass of ``odatse.solver.SolverBase``
(the full script is ``sample/parallel_solver/parallel_solver.py``):

.. code:: python

    import os, time, argparse
    import numpy as np
    from mpi4py import MPI
    import odatse
    from odatse.algorithm import choose_algorithm

    class ParallelSolver(odatse.solver.SolverBase):
        def __init__(self, info, **kwargs):
            super().__init__(info)

            self.opt_x = None
            self.opt_fx = np.inf

            self.nmats = kwargs["nmats"]
            self.matsize = kwargs["matsize"]

            if odatse.mpi.rank() == 0:
                print(f"nalg: {odatse.mpi.algsize()}")
                print(f"nsolve: {odatse.mpi.solsize()}")
            odatse.mpi.comm().barrier()

        def _testfunc(self, mats):
            return np.sum([np.max(np.linalg.svd(mat, compute_uv=False)) for mat in mats])

        def _compute(self, seeds):  # called by all solcomm ranks
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

The objective value is computed in ``evaluate``. Note that ``evaluate`` runs on
every rank of the solver group: the controller (``solrank() == 0``) generates
the ``nmats`` random matrices for each seed, broadcasts them over ``solcomm``,
each rank computes the largest singular value of its own slice with
``_testfunc``, and the partial sums are reduced across the group before being
averaged. Only the controller records the running best solution in
``self.opt_x`` / ``self.opt_fx``.

Driver and input file
~~~~~~~~~~~~~~~~~~~~~~~

The script builds the ODAT-SE pipeline in its ``main()``. It parses ``--nalg``
/ ``--nsolve``, hands them to ``odatse.initialize()`` (which calls
``odatse.mpi.setup()``), constructs the solver and runner, chooses the
algorithm, and runs it. After ``alg.main()`` returns, the best result of each
solver group is collected over ``algcomm``:

.. code:: python

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--nalg', help='# of processes for search algorithm', type=int, default=1)
        parser.add_argument('-n', '--nsolve', help='# of processes for solver', type=int, default=1)
        args = parser.parse_args()

        assert args.nalg * args.nsolve == odatse.mpi.comm().size

        argv = ["input.toml", "--init", f"--nalg={args.nalg}", f"--nsolve={args.nsolve}"]
        info, run_mode = odatse.initialize(argv)

        nmats = info.solver["param"].get("nmats", 50)
        matsize = info.solver["param"].get("matsize", 1000)

        output_dir = info.base.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        solver = ParallelSolver(info, nmats=nmats, matsize=matsize)
        runner = odatse.Runner(solver, info)
        alg_module = choose_algorithm(info.algorithm["name"])
        alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
        result = alg.main()

        if odatse.mpi.run_on_algorithm():
            opt_fx, opt_x = min(
                odatse.mpi.algcomm().allgather((solver.opt_fx, solver.opt_x)),
                key=lambda x: x[0])

        if odatse.mpi.rank() == 0:
            print(f"\nopt_x={opt_x}")
            print(f"opt_fx={opt_fx}")

The input file ``input.toml`` selects the ``mapper`` algorithm and sets the
search range and the solver parameters:

.. code:: toml

    [base]
    dimension = 1
    output_dir = "output"

    [solver]
    name = "custom"

    [solver.param]
    nmats = 50
    matsize = 1000

    [algorithm]
    name = "mapper"

    [algorithm.param]
    min_list = [1]
    max_list = [20]
    num_list = [20]

Here ``[solver.param]`` holds the parameters read in ``main()`` (``nmats`` and
``matsize``, the number and size of the random matrices). The ``[solver] name``
is nominal: ``parallel_solver.py`` instantiates ``ParallelSolver`` directly, so
this string only labels the run. Because the algorithm is ``mapper``, ODAT-SE
sweeps the ``num_list = [20]`` grid points (the integer seeds ``1`` to ``20``)
and distributes them across the ``nalg`` solver-group controllers.

The values of ``nalg`` and ``nsolve`` are passed on the command line, not
through ``input.toml``.

Running
~~~~~~~

The number of MPI processes must equal ``nalg × nsolve``. For 3 algorithm
processes and 2 solver processes per group (6 ranks in total), with 2 BLAS
threads per process:

.. code:: bash

    export OMP_NUM_THREADS=2
    mpirun -np 6 python3 parallel_solver.py --nalg 3 --nsolve 2

The sample's ``do.sh`` runs a smaller configuration:

.. code:: bash

    export OMP_NUM_THREADS=2
    mpirun -np 4 python3 parallel_solver.py -m 2 -n 2

(``-m`` / ``-n`` are the short forms of ``--nalg`` / ``--nsolve``.) When viewing
the process activity (with ``top`` or a similar tool), you should see the MPI
``python`` processes each using up to ``OMP_NUM_THREADS × 100 %`` CPU during the
solver step.

To run serially without MPI, set ``ODATSE_NOMPI=1``:

.. code:: bash

    ODATSE_NOMPI=1 python3 parallel_solver.py

All parallelism is disabled and the script runs in a single process, which is
convenient for testing.
