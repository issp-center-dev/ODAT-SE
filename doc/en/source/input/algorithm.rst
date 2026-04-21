.. _input_parameter_algorithm:

==========================================
``[algorithm]`` section
==========================================

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


See :doc:`/algorithm/index` for details of the various algorithms and their input/output files.
