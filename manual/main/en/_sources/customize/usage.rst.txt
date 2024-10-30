Usage
===========

The following flow solves the optimization problem.
The number of flow corresponds the comment in the program example.

1. Define your ``Algorithm`` and/or ``Solver``

   - Of course, classes that ``py2dmat`` defines are available

2. Prepare the input parameter, ``info: py2dmat.Info``

   - Make a dictionary as your favorite way

      - The below example uses a TOML formatted input file for generating a dictionary

3. Instantiate ``solver: Solver``, ``runner: py2dmat.Runner``, and ``algorithm: Algorithm``

4. Invoke ``algorithm.main()``


Example

.. code-block:: python

    import sys
    import tomli
    import py2dmat

    # (1)
    class Solver(py2dmat.solver.SolverBase):
        # Define your solver
        ...

    class Algorithm(py2dmat.algorithm.AlgorithmBase):
        # Define your algorithm
        ...
    

    # (2)
    with open(sys.argv[1]) as f:
        inp = tomli.load(f)
    info = py2dmat.Info(inp)

    # (3)
    solver = Solver(info)
    runner = py2dmat.Runner(solver, info)
    algorithm = Algorithm(info, runner)

    # (4)
    algorithm.main()
