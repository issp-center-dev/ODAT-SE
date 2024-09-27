Usage
================================

The following flow solves the optimization problem.
The number of flow corresponds the comment in the program example.

1. Define your ``Algorithm`` and/or ``Solver``.

   - Classes that ODAT-SE provides are available, of course.

2. Prepare the input parameter, ``info: odatse.Info``.

   - ``Info`` class has a class method to read input files in TOML format.
     It is also possible to prepare a set of parameters as a dict and to pass it to the constructor of ``Info`` class.

3. Instantiate ``solver: Solver``, ``runner: odatse.Runner``, and ``algorithm: Algorithm``.

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
    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner)

    # (4)
    result = algorithm.main()
