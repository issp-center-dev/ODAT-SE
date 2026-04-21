======================================
Random search ``random_search``
======================================

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

In this section, the search mode is defined. If this section is omitted, the pseudo-random sequence (``random``) is chosen.

- ``mode``

  Format: String

  Description: Specify ``random`` for pseudo-random sequence, or ``quasi-random`` for quasi-random sequence.

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

  Format: List of float. The length should match the value of dimension.

  Description: The maximum value the parameter can take.

- ``num_points``

  Format: Integer.

  Description: The number of points to be randomly sampled.

- ``unit_list``

  Format: List of float. The length should match the value of dimension.

  Description:
  Units for each parameter.
  In the search algorithm, each parameter is divided by each of these values to perform a simple dimensionless and normalization.
  If not defined, the value is 1.0 for all dimensions.

The following parameters can be set in the ``[algorithm]`` section.

- ``seed``

  Format: Integer.

  Description: The seed for the pseudo-random number generator used to generate the parameters.


Output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ColorMap.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the candidate parameters for each sample point and the objective function value at that point.
The data is listed in the order of the variables defined in ``string_list`` in the ``[solver]`` - ``[param]`` sections of the input file, and the value of the objective function is listed last.

Below, an output example is shown.

.. code-block::

    5.155393 -2.203493 187.944291
    -3.792974 -3.545277 3.179381
    0.812700 1.146536 108.254643
    5.574174 1.838125 483.841834
    2.986880 1.842838 0.436331
    ...


Restart
~~~~~~~~~~~~~~~~~~~~~~
The execution mode is specified by the ``run_mode`` parameter to the constructor.
The operation of each mode is described as follows.
The parameter values correspond to ``--init``, ``--resume``, and ``--cont`` options of ``odatse`` command, respectively.

- ``"initial"`` (default)

  The program is started from the initial state.
  If the checkpointing is enabled, the intermediate states will be stored at the following occasions:

  #. the specified number of grid points has been evaluated, or the specified period of time has passed.
  #. at the end of the execution.

- ``"resume"``

  The program execution is resumed from the latest checkpoint.
  The conditions such as the number of MPI processes should be kept the same.

- ``"continue"``

  The continue mode is not supported. For the pseudo-random sequence, the calculation can be continued by starting with a different seed number.


Algorithm description
~~~~~~~~~~~~~~~~~~~~~~

The random search algorithm generates ``num_points`` parameter vectors :math:`x` by uniform random sampling from the range defined by ``min_list`` and ``max_list`` for each dimension. For each generated point, the solver is called to evaluate the objective function :math:`f(x)`.

When running with MPI parallelism, the generated candidate points are divided equally among the processes, and each process evaluates its assigned points in parallel.

Unlike the grid-based search (``mapper``), which discretizes the parameter space regularly, random search samples uniformly at random. This avoids the curse of dimensionality, where the number of grid points increases exponentially with the number of dimensions. On the other hand, random search is not well-suited for precisely locating optimal solutions, so it is often used as a preliminary exploration before applying other optimization methods.

