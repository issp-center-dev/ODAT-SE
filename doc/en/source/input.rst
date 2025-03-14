Input file
================================

As the input file format, `TOML <https://toml.io/ja/>`_ format is used.
The input file consists of the following four sections.

- ``base``

  - Specify the basic parameters about ODAT-SE. 

- ``solver``

  - Specify the parameters about ``Solver`` .

- ``algorithm``

  - Specify the parameters about ``Algorithm`` .

- ``runner``

  - Specify the parameters about ``Runner`` .


[``base``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``dimension``

  Format: Integer

  Description: Dimension of the search space (number of parameters to search)

- ``root_dir``

  Format: string (default: The directory where the program was executed)

  Description: Name of the root directory. The origin of the relative paths to input files.

- ``output_dir``

  Format: string (default: The directory where the program was executed)

  Description: Name of the directory to output the results.

[``solver``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``name`` determines the type of solver. Each parameter is defined for each solver.

- ``name``

  Format: String

  Description: Name of the solver. The following solvers are available.

    - ``analytical`` : Solver to provide analytical solutions (mainly used for testing).

    The following are solvers for 2D material structure analysis distributed as separate modules:

    - ``sim-trhepd-rheed`` :
      Solver to calculate Total-reflection high energy positron diffraction (TRHEPD) or Reflection High Energy Electron Diffraction (RHEED) intensities.

    - ``sxrd`` : Solver for Surface X-ray Diffraction (SXRD)

    - ``leed`` : Solver for Low-energy Electron Diffraction (LEED)

- ``dimension``

  Format: Integer (default: ``base.dimension``)

  Description:
  Number of input parameters for Solvers

See :doc:`solver/index` for details of the various solvers and their input/output files.

.. _input_parameter_algorithm:

[``algorithm``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``name`` determines the type of algorithm. Each parameter is defined for each algorithm.

- ``name``

  Format: String

  Description: Algorithm name. The following algorithms are available.

    - ``minsearch`` : Minimum value search using Nelder-Mead method

    - ``mapper`` : Grid search

    - ``exchange`` :  Replica Exchange Monte Carlo method

    - ``pamc`` :  Population Annealing Monte Carlo method

    - ``bayes`` :  Bayesian optimization

- ``seed``

  Format: Integer

  Description:
  A parameter to specify seeds of the pseudo-random number generator used for random generation of initial values, Monte Carlo updates, etc.
  For each MPI process, the value of ``seed + mpi_rank * seed_delta`` is given as seeds.
  If omitted, the initialization is done by  `the Numpy's prescribed method <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng>`_.

- ``seed_delta``

  Format: Integer (default: 314159)

  Description:
  A parameter to calculate the seed of the pseudo-random number generator for each MPI process.
  For details, see the description of ``seed``.

- ``checkpoint``

  Format: Boolean (default: false)

  Description:
  A parameter to specify whether the intermediate states are periodically stored to files. The final state is also saved. In case when the execution is terminated, it will be resumed from the latest checkpoint.

- ``checkpoint_steps``

  Format: Integer (default: 16,777,216)

  Description:
  A parameter to specify the iteration steps between the previous and next checkpoints. One iteration step corresponds to one evaluation of grid point in the mapper algorithm, one evaluation of Bayesian search in the bayes algorithm, and one local update in  the Monte Carlo (exchange and PAMC) algorithms.
  The default value is a sufficiently large number of steps. To enable checkpointing, at least either of ``checkpoint_steps`` or ``checkpoint_interval`` should be specified.

- ``checkpoint_interval``

  Format: Floating point number (default: 31,104,000)

  Description:
  A parameter to specify the execution time between the previous and next checkpoints in unit of seconds.
  The default value is a sufficiently long period (360 days). To enable checkpointing, at least either of ``checkpoint_steps`` or ``checkpoint_interval`` should be specified.

- ``checkpoint_file``

  Format: String (default: ``"status.pickle"``)

  Description:
  A parameter to specify the name of output file to which the intermediate state is written.
  The files are generated in the output directory of each process.
  The past three generations are kept with the suffixes .1, .2, and .3 .


See :doc:`algorithm/index` for details of the various algorithms and their input/output files.

[``runner``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section sets the configuration of ``Runner``, which bridges ``Algorithm`` and ``Solver``.
It has three subsections, ``mapping``, ``limitation``, and ``log`` .

- ``ignore_error``

  Format: Boolean (default: false)

  Description:
  A parameter to specify whether a RuntimeError occuured within the direct problem solver is ignored and the calculation is continued with NaN as the result. Note that only the RuntimeError exceptions are captured.


[``runner.mapping``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section defines the mapping from an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, to an :math:`M` dimensional parameter used in ``Solver``, :math:`y` .
In the case of :math:`N \ne M`, the parameter ``dimension`` in ``[solver]`` section should be specified.

In the current version, the affine mapping (linear mapping + translation) :math:`y = Ax+b` is available.

- ``A``

  Format: List of list of float, or a string (default: ``[]``)

  Description:
  :math:`N \times M` matrix :math:`A`. An empty list ``[]`` is a shorthand of an identity matrix.
  If you want to set it by a string, arrange the elements of the matrix separated with spaces and newlines (see the example).


- ``b``

  Format: List of float, or a string (default: ``[]``)

  Description:
  :math:`M` dimensional vector :math:`b`. An empty list ``[]`` is a shorthand of a zero vector.
  If you want to set it by a string, arrange the elements of the vector separated with spaces.

For example, both ::

  A = [[1,1], [0,1]]

and ::

  A = """
  1 1
  0 1
  """

mean

.. math::

  A = \left(
  \begin{matrix}
  1 & 1 \\
  0 & 1
  \end{matrix}
  \right).


[``limitation``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section defines the limitation (constraint) in an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, in addition of ``min_list`` and ``max_list``.

In the current version, a linear inequation with the form :math:`Ax+b>0` is available. Specifically, you can apply constraints as follows:

.. math::

  A_{1,1} x_{1} + A_{1,2} x_{2} + &... + A_{1,N} x_{N} + b_{1} > 0\\
  A_{2,1} x_{1} + A_{2,2} x_{2} + &... + A_{2,N} x_{N} + b_{2} > 0\\
  &...\\
  A_{M,1} x_{1} + A_{M,2} x_{2} + &... + A_{M,N} x_{N} + b_{M} > 0 

where :math:`M` is the number of constraint equations (arbitrary).

- ``co_a``

  Format: List of list of float, or a string (default: ``[]``)

  Description:
  :math:`M \times N` matrix :math:`A` for the constraint equations. 
  The number of rows should be the number of constraints :math:`M`, and the number of columns should be the number of search variables :math:`N`.
  You must define ``co_b`` together with this parameter.

- ``co_b``

  Format: List of float, or a string (default: ``[]``)

  Description:
  :math:`M` dimensional vector :math:`b` for the constraint equations. 
  You need to set a column vector with the dimension equal to the number of constraints :math:`M`.
  You must define ``co_a`` together with this parameter.

For example, both ::

  A = [[1,1], [0,1]]

and ::

  A = """
  1 1
  0 1
  """

mean

.. math::

  A = \left(
  \begin{matrix}
  1 & 1 \\
  0 & 1
  \end{matrix}
  \right).

Also, the following examples:

.. code-block:: toml
  
  co_b = [[0], [-1]]

and

.. code-block:: toml

  co_b = """0 -1"""
  
and

.. code-block:: toml

  co_b = """
  0 
  -1
  """

all represent:

.. math::

  b = \left(
  \begin{matrix}
  0 \\
  -1 
  \end{matrix}
  \right)

If neither ``co_a`` nor ``co_b`` is defined, no constraint equation will be applied to the search.


[``log``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting parametrs related to logging of solver calls.

- ``filename``

  Format: String (default: "runner.log")

  Description: Name of log file.

- ``interval``

  Format: Integer (default: 0)

  Description:
  The log will be written out every time solver is called ``interval`` times.
  If the value is less than or equal to 0, no log will be written.

- ``write_result``

  Format: Boolean (default: false)

  Description: Whether to record the output from solver.

- ``write_input``

  Format: Boolean (default: false)

  Description: Whether to record the input to solver.

MPI Parallel Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODAT-SE supports parallel computation using MPI. Using MPI, you can speed up calculations by utilizing multiple processes.

- Algorithms such as ``exchange``, ``pamc``, and ``mapper`` can benefit from MPI parallelization
- During parallel execution, each process has its own random number sequence (see ``seed`` and ``seed_delta`` parameters)
- Checkpoint files are created for each process

Execution example:

.. code-block:: bash

    $ mpirun -np 4 odatse input.toml

The ``-np 4`` part specifies the number of processes to use. Adjust according to the number of cores available.

Depending on your environment, you may need to use ``mpiexec`` or other commands, or execute MPI programs through a job scheduler. Large-scale computing centers in particular may have system-specific execution methods. Please refer to the manual for your environment for details.

.. note::
   Parallelization efficiency varies by algorithm. For example, with ``exchange``, it is efficient to use the same number of processes as replicas or fewer.
