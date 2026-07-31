Usage
================================

The following flow solves the optimization problem.
The numbers of the steps correspond to the comments in the program example below.

1. Define your ``Algorithm`` and/or ``Solver``.

   - The classes provided by ODAT-SE can also be used.

2. Prepare the input parameter, ``info: odatse.Info``.

   - ``Info`` class has a class method to read input files in TOML format.
     It is also possible to prepare a set of parameters as a dict and to pass it to the constructor of ``Info`` class.

3. Call ``odatse.mpi.setup()`` (it partitions the MPI communicator and is required before constructing the solver/algorithm; ``odatse.initialize()`` does this automatically), then instantiate ``solver: Solver``, ``runner: odatse.Runner``, and ``algorithm: Algorithm``.

4. Invoke ``algorithm.main()``.


Example:

.. code-block:: python

    import sys
    import odatse

    # (1)
    class Solver(odatse.solver.SolverBase):
        # Define your solver
        ...

    class Algorithm(odatse.algorithm.AlgorithmBase):
        # Define your algorithm
        ...


    # (2)
    input_file = sys.argv[1]
    info = odatse.Info.from_file(input_file)

    # (3)
    odatse.mpi.setup()
    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner)

    # (4)
    result = algorithm.main()


Handling command-line arguments
--------------------------------------------

To use the same argument scheme as the ``odatse`` command in your own scripts (restarting with ``--resume``, MPI splitting with ``--nalg`` / ``--nsolve``), it is convenient to perform the initialization of steps (2) and (3) with ``odatse.initialize()`` (see :doc:`common`).

.. code-block:: python

    import odatse

    # (1) user-defined classes (omitted)

    # (2)(3) parse command-line arguments and initialize
    #        (odatse.mpi.setup() is called internally)
    info, run_mode = odatse.initialize()

    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner, run_mode=run_mode)

    # (4)
    result = algorithm.main()

Passing ``run_mode`` to the constructor of ``Algorithm`` enables restarting from a checkpoint (``--resume``) and continuation (``--cont``) also in your own scripts.
If you do not want to depend on ``sys.argv``, pass an explicit argument list as ``odatse.initialize(["input.toml"])``.
