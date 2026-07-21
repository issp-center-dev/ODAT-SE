Output files
=====================

See :doc:`solver/index` and :doc:`algorithm/index` for the output files of each ``Solver`` and ``Algorithm``.

Quick reference of output files by algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main files produced by each algorithm are listed below.
``RANK/`` denotes the per-MPI-rank subfolder (such as ``output_dir/0/``), and ``#`` is the index of a temperature point.
See the "Output files" section of each algorithm page for the details such as the meaning of the columns.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Algorithm
     - Main output files
     - Contents
   * - :doc:`minsearch <algorithm/minsearch>`
     - ``res.txt``, ``RANK/SimplexData.txt``, ``RANK/History_FunctionCall.txt``
     - Optimization result, simplex search path, history of function evaluations
   * - :doc:`mapper <algorithm/mapper_mpi>`
     - ``ColorMap.txt``
     - Coordinates and objective function values of the grid points (the file name can be changed by ``colormap``)
   * - :doc:`random_search <algorithm/random_search>`
     - ``ColorMap.txt``
     - Coordinates and objective function values of the sampled points
   * - :doc:`bayes <algorithm/bayes>`
     - ``BayesData.txt``
     - History of the estimated optimum and the evaluated points at each step
   * - :doc:`ttopt <algorithm/ttopt>`
     - ``res.txt``, ``ttopt_hyperparameters.txt``, ``ttopt_history.txt``, ``ttopt_eval_history.txt``
     - Optimization result, hyperparameters, history of the best value, evaluation history (optional)
   * - :doc:`exchange <algorithm/exchange>`
     - ``RANK/trial.txt``, ``RANK/result.txt``, ``result_T#.txt``, ``best_result.txt``, ``fx.txt``
     - Proposed and accepted samples, per-temperature logs, best solution, per-temperature statistics
   * - :doc:`pamc <algorithm/pamc>`
     - ``RANK/trial_T#.txt``, ``RANK/trial.txt``, ``RANK/result_T#.txt``, ``RANK/result.txt``, ``RANK/weight.txt``, ``best_result.txt``, ``fx.txt``, ``pr.txt``
     - Proposed and accepted samples (per temperature / all), replica weights, best solution, per-temperature statistics, partition function ratios

In addition, ``time.log`` is written as a common file regardless of the algorithm.
Depending on the settings, ``runner.log`` (when ``runner.log.interval`` is a positive integer) and ``status.pickle`` (when the checkpointing feature is enabled) may also be written, as described below.

Common file
~~~~~~~~~~~~~~~~~~

``time.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The total time taken for the calculation for each MPI rank is outputted.
These files will be output under the subfolders of each rank respectively.
The time taken to pre-process the calculation, the time taken to compute, and the time taken to post-process the calculation are listed in the ``prepare`` , ``run`` , and ``post`` sections.

The following is an example of the output.

.. code-block::

    #prepare
     total = 0.007259890999989693
    #run
     total = 1.3493346729999303
     - file_CM = 0.0009563499997966574  # Time spent on file I/O
     - submit = 1.3224223930001244      # Time spent on calculation processing
    #post
     total = 0.000595873999941432

The ``prepare`` section shows the time spent on initialization, ``run`` shows the main calculation processing time, and ``post`` shows the post-processing time.
The items within the ``run`` section may vary depending on the execution environment and settings.


``runner.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The log information about solver calls for each MPI rank is outputted.
These files will be output under the subfolder of each rank.
The output is only available when the ``runner.log.interval`` parameter is a positive integer in the input. This value specifies how frequently the log entries are recorded. For example, if ``runner.log.interval = 10``, logs will be recorded every 10 calls.

Each column in the log represents the following information:

- The first column is the serial number of the solver call.
- The second column is the time elapsed since the last solver call.
- The third column is the time elapsed since the start of the calculation.

The following is an example of the output.

.. code-block::

    # $1: num_calls
    # $2: elapsed_time_from_last_call
    # $3: elapsed_time_from_start

    1 0.0010826379999999691 0.0010826379999999691
    2 6.96760000000185e-05 0.0011523139999999876
    3 9.67080000000009e-05 0.0012490219999999885
    4 0.00011765699999999324 0.0013666789999999818
    5 4.965899999997969e-05 0.0014163379999999615
    6 8.666900000003919e-05 0.0015030070000000006
       ...

``status.pickle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If ``algorithm.checkpoint`` is set to true, the intermediate states are stored to ``status.pickle`` (or the filename specified by the ``algorithm.checkpoint_file`` parameter) for each MPI process in its subfolder.
They are read when the execution is resumed.
The content of the file depends on the algorithm.

The checkpoint feature allows you to resume calculations from the last saved state if a long calculation is interrupted.
To resume, run the program with the same input file using ``odatse --resume input.toml``.
To continue from a previous run while extending the calculation, use ``odatse --cont input.toml``.
If you want to use a new random number sequence when resuming or continuing, add the ``--reset_rand`` option.
