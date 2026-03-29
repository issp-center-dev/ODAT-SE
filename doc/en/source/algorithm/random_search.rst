======================================
Random search ``random_search``
======================================

``random_search`` is an algorithm that randomly samples parameters from a uniform distribution within a specified search range and evaluates the objective function :math:`f(x)` at each point.
It is effective for high-dimensional problems where the computational cost of grid search becomes prohibitive, or when you want to obtain a global overview of the parameter space.
In the case of MPI execution, the set of candidate points is divided into equal parts and automatically assigned to each process to perform trivial parallel computation.


Preparation
~~~~~~~~~~~~

For MPI parallelism, you need to install `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.

.. code-block::

   $ python3 -m pip install mpi4py

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Algorithm description
~~~~~~~~~~~~~~~~~~~~~~

The random search algorithm generates ``num_points`` parameter vectors :math:`x` by uniform random sampling from the range defined by ``min_list`` and ``max_list`` for each dimension. For each generated point, the solver is called to evaluate the objective function :math:`f(x)`.

When running with MPI parallelism, the generated candidate points are divided equally among the processes, and each process evaluates its assigned points in parallel.

Unlike the grid-based search (``mapper``), which discretizes the parameter space regularly, random search samples uniformly at random. This avoids the curse of dimensionality, where the number of grid points increases exponentially with the number of dimensions. On the other hand, random search is not well-suited for precisely locating optimal solutions, so it is often used as a preliminary exploration before applying other optimization methods.


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

  The continue mode is not supported.
