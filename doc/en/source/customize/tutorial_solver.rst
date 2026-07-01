Tutorial: Adding a Custom Solver
==========================================

This tutorial explains step by step how to define your own objective function and minimize it
using ODAT-SE's search algorithms.

As an example, we find the minimum of the following 2-variable function using the Nelder-Mead method.

.. math::

   f(x, y) = (x - 3)^2 + (y - 2)^2

The minimum is :math:`f(3, 2) = 0`.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~

- ODAT-SE is installed (see :doc:`/start`)
- Basic knowledge of Python syntax (function definitions, class basics)


Overview
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Define a solver in a Python script
2. Create a TOML configuration file
3. Run and check the results


Step 1: Define a Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file called ``my_solver.py`` with the following content.

.. code-block:: python

    import sys
    import numpy as np
    import odatse

    # --- Solver definition ---
    class MySolver(odatse.solver.SolverBase):
        """A solver that computes a custom objective function"""

        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_solver"

            # You can read parameters from the [solver] section of the TOML file
            # Example: self.param = info.solver.get("my_param", 1.0)

        def evaluate(self, x, args=()):
            """
            Compute and return the objective function value.

            Parameters
            ----------
            x : np.ndarray
                Search parameters (here a 2D vector [x, y])
            args : tuple
                (step number, set number) tuple. Can be used for logging.
            """
            # Write your objective function here
            fx = (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2
            return fx

    # --- Main execution code ---
    # Get the TOML file path from command line arguments
    input_file = sys.argv[1]
    info = odatse.Info.from_file(input_file)

    # Assemble: Solver -> Runner -> Algorithm
    solver = MySolver(info)
    runner = odatse.Runner(solver, info)
    algorithm = odatse.algorithm.choose_algorithm(info, runner)

    # Run
    result = algorithm.main()

**Key points:**

- ``MySolver`` inherits from ``odatse.solver.SolverBase``
- ``__init__`` must call ``super().__init__(info)``. This automatically sets up output directories
- Write the objective function computation in the ``evaluate`` method. The argument ``x`` is a numpy array and the return value is a float
- ``odatse.algorithm.choose_algorithm`` automatically selects the algorithm specified in the ``[algorithm]`` section of the TOML file


Step 2: Create a TOML Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file called ``input.toml`` with the following content.

.. code-block:: toml

    [base]
    dimension = 2
    output_dir = "output"

    [solver]
    name = "my_solver"
    # You can add solver-specific parameters here
    # my_param = 1.0

    [algorithm]
    name = "minsearch"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    initial_list = [0.0, 0.0]

**What each section means:**

- ``[base]``: ``dimension = 2`` specifies that there are 2 search parameters (x, y)
- ``[solver]``: Solver settings. ``name`` is for log output and can be any string
- ``[algorithm]``: Search algorithm settings. ``minsearch`` is the Nelder-Mead method
- ``[algorithm.param]``: Max/min search range and initial values


Step 3: Run and Check the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place ``my_solver.py`` and ``input.toml`` in the same directory and run:

.. code-block:: bash

    $ python3 my_solver.py input.toml

When complete, output similar to the following is displayed:

.. code-block:: text

    Iterations: 43
    Function evaluations: 82
    Solution:
    x1 = 2.9999999...
    x2 = 1.9999999...

The parameters have converged to :math:`(x, y) = (3, 2)`, confirming that the minimum was found.

Execution logs are also output under the ``output/`` directory.


Advanced: Searching with Other Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By changing only the ``[algorithm]`` section in the TOML file, you can search with a different algorithm without modifying the Python code.

**Bayesian optimization example:**

.. code-block:: toml

    [algorithm]
    name = "bayes"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]

    [algorithm.bayes]
    random_max_num_probes = 10
    score = "TS"
    num_search_each_probe = 1

**Grid search example:**

.. code-block:: toml

    [algorithm]
    name = "mapper"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    num_list = [31, 31]


Advanced: Reading Solver Parameters from TOML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add parameters to the objective function, add values to the ``[solver]`` section
and read them in ``__init__``.

.. code-block:: toml

    [solver]
    name = "my_solver"
    center_x = 5.0
    center_y = 3.0

.. code-block:: python

    class MySolver(odatse.solver.SolverBase):
        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_solver"
            # Read values from the [solver] section of the TOML file
            self.cx = info.solver.get("center_x", 0.0)
            self.cy = info.solver.get("center_y", 0.0)

        def evaluate(self, x, args=()):
            return (x[0] - self.cx) ** 2 + (x[1] - self.cy) ** 2

This way, you can change parameters without modifying the code, simply by editing the TOML file.


Advanced: Using Solver Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To organize your solver as an installable package, use the `ODAT-SE solver templates <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery/-/tree/main/data/tutorial/solver-template>`_.
Four templates are provided for different use cases.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Template
     - Use Case
     - Features
   * - ``user_function``
     - Quickly optimize a Python function
     - Minimal script. Similar to Steps 1-3 above
   * - ``function_module``
     - Package an analytical function solver
     - ``pip install``-able module. Just add your function
   * - ``solver_module``
     - Build a custom solver that reads data files
     - Template for solvers with reference data comparison (likelihood, etc.)
   * - ``external_solver_module``
     - Use an external program (C/Fortran) as a solver
     - Includes I/O file management, subprocess execution, working directory management

**Example usage (function_module):**

1. Copy the template

   .. code-block:: bash

       $ cp -r odat-se-gallery/data/tutorial/solver-template/function_module my_solver_pkg
       $ cd my_solver_pkg

2. Change the package name in ``pyproject.toml`` and add your solver under ``src/Solver/``

3. Install and run

   .. code-block:: bash

       $ python3 -m pip install .

Each template includes sample configuration files (``sample/``) and test scaffolding (``tests/``),
making it a good starting point for developing production-ready solver packages.
For details, see the ``docs/`` directory within each template.


Next Steps
~~~~~~~~~~~~~~~~~~~~~~~~~

- You can call external programs inside the ``evaluate`` method to integrate with more complex forward problem solvers
- For detailed Solver API, see :doc:`solver`
- To define custom search algorithms, see :doc:`algorithm`
