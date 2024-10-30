Input file
===========================

As the input file format, `TOML <https://toml.io/ja/>`_ format is used.
The input file consists of the following six sections.

- ``base``

  - Specify the basic parameters about ``py2dmat`` . 

- ``solver``

  - Specify the parameters about ``Solver`` .

- ``algorithm``

  - Specify the parameters about ``Algorithm`` .

- ``runner``

  - Specify the parameters about ``Runner`` .

- ``mapping``

  - Define the mapping from a parameter searched by ``Algorithm`` .

- ``limitation``

  - Define the limitation (constration) of parameter searched by ``Algorithm`` .

- ``log``

  - Specify parameters related to logging of solver calls.
  
[``base``] section
************************

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
************************

The ``name`` determines the type of solver. Each parameter is defined for each solver.


- ``name``

  Format: String

  Description: Name of the solver. The following solvers are available.

    - ``sim-trhepd-rheed`` : Solver to calculate Total-reflection high energy positron diffraction (TRHEPD) or Reflection High Energy Electron Diffraction (RHEED) intensities.

    - ``analytical`` : Solver to provide analytical solutions (mainly used for testing).

See :doc:`solver/index` for details of the various solvers and their input/output files.

.. _input_parameter_algorithm:

[``algorithm``] section
*******************************

The ``name`` determines the type of algorithm. Each parameter is defined for each algorithm.

- ``name``

  Format: String

  Description: Algorithm name. The following algorithms are available.

    - ``minsearch`` : Minimum value search using Nelder-Mead method

    - ``mapper`` : Grid search

    - ``exchange`` :  Replica Exchange Monte Carlo

    - ``bayes`` :  Bayesian optimization

- ``seed``

  Format: Integer

  Description: A parameter to specify seeds of the pseudo-random number generator used for random generation of initial values, Monte Carlo updates, etc.
        For each MPI process, the value of ``seed + mpi_rank * seed_delta`` is given as seeds.
        If omitted, the initialization is done by  `the Numpy's prescribed method <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng>`_.


- ``seed_delta``

  Format: Integer (default: 314159)

  Description: A parameter to calculate the seed of the pseudo-random number generator for each MPI process.
        For details, see the description of ``seed``.

See :doc:`algorithm/index` for details of the various algorithms and their input/output files.

[``runner``] section
************************

This section sets the configuration of ``Runner``, which bridges ``Algorithm`` and ``Solver``.
It has two subsections, ``mapping`` and ``log`` .

[``mapping``] section
************************

This section defines the mapping from an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, to an :math:`M` dimensional parameter used in ``Solver``, :math:`y` .
In the case of :math:`N \ne M`, the parameter ``dimension`` in ``[solver]`` section should be specified.

In the current version, the affine mapping (linear mapping + translation) :math:`y = Ax+b` is available.

- ``A``

  Format: List of list of float, or a string (default: ``[]``)

  Description: :math:`N \times M` matrix :math:`A`. An empty list ``[]`` is a shorthand of an identity matrix.
               If you want to set it by a string, arrange the elements of the matrix separated with spaces and newlines (see the example).


- ``b``

  Format: List of float, or a string (default: ``[]``)

  Description: :math:`M` dimensional vector :math:`b`. An empty list ``[]`` is a shorthand of a zero vector.
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
*************************

This section defines the limitation (constraint) in an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, in addition of ``min_list`` and ``max_list``.

In the current version, a linear inequation with the form :math:`Ax+b>0` is available.

- ``co_a``

  Format: List of list of float, or a string (default: ``[]``)

  Description: :math:`N \times M` matrix :math:`A`. An empty list ``[]`` is a shorthand of an identity matrix.
               If you want to set it by a string, arrange the elements of the matrix separated with spaces and newlines (see the example).


- ``co_b``

  Format: List of float, or a string (default: ``[]``)

  Description: :math:`M` dimensional vector :math:`b`. An empty list ``[]`` is a shorthand of a zero vector.
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


[``log``] section
************************

Setting parametrs related to logging of solver calls.

- ``filename``

  Format: String (default: "runner.log")

  Description: Name of log file.

- ``interval``

  Format: Integer (default: 0)

  Description: The log will be written out every time solver is called ``interval`` times.
        If the value is less than or equal to 0, no log will be written.

- ``write_result``

  Format: Boolean (default: false)

  Description: Whether to record the output from solver.

- ``write_input``

  Format: Boolean (default: false)

  Description: Whether to record the input to solver.
