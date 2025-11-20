Random search ``random_search``
**********************************

``random_search`` is an algorithm to search for the minimum value by computing :math:`f(x)` on random points in the parameter space.
This algorithm is effective when it is difficult to use other methods such as grid search for high-dimensional problems.
The random search is compatible with MPI. The sampling points are evaluated in an trivially parallel way over MPI processes.
In addition to pseudo-random sequence, quasi-random (low-discrepancy) sequences such as Sobol sequence are available.

Preparation
~~~~~~~~~~~~

For MPI parallelism, you need to install `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.

.. code-block::

   $ python3 -m pip install mpi4py

For quasi-random sequence, you need to install `scipy <https://scipy.org>`_.

.. code-block::

   $ python3 -m pip install scipy

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _random_search_input_algorithm:

[``algorithm``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``name``

  Format: String

  Description: To use random search, specify ``random_search``.

.. _random_search_input_mode:

[``mode``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, the search mode is defined. If this section is omitted, the quasi-random sequence (``random``) is chosen.

- ``mode``

  Format: String

  Description: Specify ``random`` for quasi-random sequence, or ``quasi-random`` for quasi-random sequence.

- ``sequence``

  Format: String

  Description: Specify the type of quasi-random sequence. Available types are:

  - ``sobol``: Sobol sequence

  - ``halton``: Halton sequence

  - ``latin``: Latin Hypercube

  For details, refer to the descriptions in scipy.stats.qmc manual.

.. _random_search_input_param:

[``param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, the search parameter space is defined.

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: The minimum value the parameter can take.

- ``max_list``

  Format: List of float.The length should match the value of dimension.

  Description: The maximum value the parameter can take.

- ``num_points``

  Format: Integer.

  Description: The number of points to be randomly sampled.


Output file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ColorMap.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the candidate parameters for each sampling point and the function value at that time.
The mesh data is listed in the order of the variables defined in ``string_list`` in the ``[solver]`` - ``[param]`` sections of the input file, and the value of the function value is listed last.

Below, output example is shown.

.. code-block::

    -0.829780 2.203245 232.782783
    -4.998856 -1.976674 72748.392856
    -3.532441 -4.076614 27426.531418
    -3.137398 -1.544393 12984.994049
    -1.032325 0.388167 50.034778
    -0.808055 1.852195 147.087285
    -2.955478 3.781174 2469.533327
    -4.726124 1.704675 42598.971437
    -0.826952 0.586898 4.277709
    -3.596131 -3.018985 25465.012749
    3.007446 4.682616 1906.833520
    ...


Restart
~~~~~~~~~~~~~~~~~~~~~~
The execution mode is specified by the ``run_mode`` parameter to the constructor.
The operation of each mode is described as follows.
The parameter values correspond to ``--init``, ``--resume``, and ``--cont`` options of ``odatse`` command, respectively.

- ``"initial"`` (default)

  The program is started from the initial state.
  If the checkpointing is enabled, the intermediate states will be stored at the folloing occasions:

  #. the specified number of grid points has been evaluated, or the specified period of time has passed.
  #. at the end of the execution.

- ``"resume"``

  The program execution is resumed from the latest checkpoint.
  The conditions such as the number of MPI processes should be kept the same.

- ``"continue"``

  The continue mode is not supported. For the quasi-random sequence, the calculation can be continued by starting with a different seed number.
