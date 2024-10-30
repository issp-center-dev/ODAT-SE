Addition of a direct problem solver
=====================================

Solver for benchmarking, ``analytical``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``py2dmat`` provides an ``analytical`` solver as a direct problem solver that can be used to test search algorithms.

To use the ``analytical`` solver, set ``name`` to ``"analytical"`` in the ``[solver]`` section of the input file.
You can also use the ``function_name`` parameter to select the benchmark function :math:`f(x)`.

For example, to use Himmelblau function, make an input file including the following::

    [solver]
    name = "analytical"
    function_name = "himmelblau"

For details of ``analytical`` solver, please check the :doc:`the reference of the analytical solver <../solver/analytical>`.


Addition of a direct problem solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to define and analyze a user-defined direct problem solver is to add it to the ``analytical`` solver.
As an example, we will explain the case of adding `the Booth function <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_ (the minimum point is :math:`f(1,3) = 0`):

.. math::

    f(x,y) = (x+2y-7)^2 + (2x+y-5)^2.


To do so, we need to download the source code for ``py2dmat`` and edit the file for ``analytical`` solver.
For instructions on how to download the source code and run ``py2dmat`` from the source code, see :doc:`how to install <../start>`.
``analytical`` solver is defined in the ``src/py2dmat/solver/analytical.py``, so edit it.

First, define the booth function as follows:

.. code-block:: python

    def booth(xs: np.ndarray) -> float:
        """Booth function

        it has one global minimum f(xs) = 0 at xs = [1,3].
        """

        if xs.shape[0] != 2:
            raise RuntimeError(
                f"ERROR: booth expects d=2 input, but receives d={xs.shape[0]} one"
            )
        return (xs[0] + 2 * xs[1] - 7.0) ** 2 + (2 * xs[0] + xs[1] - 5.0) ** 2

Next, insert the following code in the if branch of the ``Solver`` class's constructor (``__init__``)
to allow users to choose the ``booth`` function by the ``solver.function_name`` parameter of the input file.

.. code-block:: python

    elif function_name == "booth":
            self.set_function(booth)

With this modified ``analytical`` solver, you can optimize the Booth function.
For example, to optimize it by the Nelder-Mead method, pass the following input file (``input.toml``) ::

    [base]
    dimension = 2
    output_dir = "output"

    [algorithm]
    name = "minsearch"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    initial_list = [0, 0]

    [solver]
    name = "analytical"
    function_name = "booth"

to ``src/py2dmat_main.py`` script as follows::

    $ python3 src/py2dmat_main.py input.toml

    ... skipped ...

    Iterations: 38
    Function evaluations: 75
    Solution:
    x1 = 1.0000128043523089
    x2 = 2.9999832920260863
