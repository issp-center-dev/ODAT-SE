Direct parallel search ``mapper``
**********************************

``mapper_mpi`` is an algorithm to search for the minimum value by computing :math:`f(x)` on all the candidate points in the parameter space prepared in advance.
In the case of MPI execution, the set of candidate points is divided into equal parts and automatically assigned to each process to perform trivial parallel computation.

Preparation
~~~~~~~~~~~~

For MPI parallelism, you need to install `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.

.. code-block::

   $ python3 -m pip install mpi4py

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _mapper_input_param:

[``param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, the search parameter space is defined.

If ``mesh_path`` is defined, it is read from a mesh file.
In the mesh file, one line defines one point in the parameter space, the first column is the data number, and the second and subsequent columns are the coordinates of each dimension.

If ``mesh_path`` is not defined, ``min_list``, ``max_list``, and ``num_list`` are used to create an evenly spaced grid for each parameter.

- ``mesh_path``

  Format: String

  Description: Path to the mesh definition file.

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: The minimum value the parameter can take.

- ``max_list``

  Format: List of float.The length should match the value of dimension.

  Description: The maximum value the parameter can take.

- ``num_list``

  Format: List of integer. The length should match the value of dimension.

  Description:  The number of grids the parametar can take at each dimension.


Refernce file
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh definition file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the grid space to be explored in this file.
1 + ``dimension`` columns are required.
The first column is the index of the mesh, and the second and subsequent columns are the values of parameter.
The lines starting from ``#`` are ignored as comments.

A sample file for two dimensions is shown below.

.. code-block::

    1 6.000000 6.000000
    2 6.000000 5.750000
    3 6.000000 5.500000
    4 6.000000 5.250000
    5 6.000000 5.000000
    6 6.000000 4.750000
    7 6.000000 4.500000
    8 6.000000 4.250000
    9 6.000000 4.000000
    ...

Output file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ColorMap.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the candidate parameters for each mesh and the function value at that time.
The mesh data is listed in the order of the variables defined in ``string_list`` in the ``[solver]`` - ``[param]`` sections of the input file, and the value of the function value is listed last.

Below, output example is shown.

.. code-block::

    6.000000 6.000000 0.047852
    6.000000 5.750000 0.055011
    6.000000 5.500000 0.053190
    6.000000 5.250000 0.038905
    6.000000 5.000000 0.047674
    6.000000 4.750000 0.065919
    6.000000 4.500000 0.053675
    6.000000 4.250000 0.061261
    6.000000 4.000000 0.069351
    6.000000 3.750000 0.071868
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

  The continue mode is not supported.
