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
     - ``res.txt``, ``RANK/SimplexData.txt``, ``RANK/History_FunctionCall.txt``, ``RANK/BasinHoppingData.txt``
     - Optimization result, simplex search path, history of function evaluations, result of each hop (only when ``basinhopping`` is enabled)
   * - :doc:`global_search <algorithm/global_search>`
     - ``res.txt``, ``RANK/History_FunctionCall.txt``, ``0/GenerationData.txt`` or ``0/IterationData.txt``, ``0/LocalMinimaData.txt``
     - Optimization result, history of function evaluations (per rank), best point at each iteration (``GenerationData.txt`` for differential evolution, ``IterationData.txt`` for shgo and direct; rank 0 only), list of the local minima found (only for shgo; rank 0 only)
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
Depending on the settings, ``runner.log`` (when ``runner.log.interval`` is a positive integer) and ``status.pickle`` (when the checkpointing feature is enabled for an algorithm that supports it) may also be written, as described below.

Common file
~~~~~~~~~~~~~~~~~~

``time.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The total time taken for the calculation is outputted.
Only the rank-0 process of the algorithm layer writes this file, so it is placed at ``output_dir/0/time.log``.
The time taken to initialize the calculation, to pre-process it, to compute, and to post-process it is listed in the ``init``, ``prepare``, ``run``, and ``post`` sections.

The following is an example of the output.

.. code-block::

    #in units of seconds
    #init
     total = 0.4090206250548363
    #prepare
     total = 0.0002522082068026066
    #run
     total = 0.017200791044160724
     - min_search = 0.016241166973486543
    #post
     total = 0.0016664580907672644

The ``init`` section shows the time spent before the algorithm starts (parsing the input file and constructing the solver and the algorithm), ``prepare`` shows the time spent on the preparation step of the algorithm, ``run`` shows the main calculation processing time, and ``post`` shows the post-processing time.
The items within the ``run`` section depend on the algorithm, the execution environment, and the settings.


``runner.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The log information about solver calls is outputted.
It is written for each rank of the algorithm layer, under its subfolder
(when solver-level parallelism is used, the solver worker processes do not write it).
The output is only available when the ``runner.log.interval`` parameter is a positive integer in the input.
**Every** solver call is recorded; this value specifies how many entries are buffered before they are flushed to the file.
For example, ``runner.log.interval = 10`` writes the entries in batches of 10 (it does not record only every tenth call).

Each column in the log represents the following information:

- The first column is the serial number of the solver call.
- The second column is the time elapsed since the last solver call.
- The third column is the time elapsed since the start of the calculation.

The following is an example of the output.

.. code-block::

    # $1: num_calls
    # $2: elapsed time from last call
    # $3: elapsed time from start

    1      0.000844 0.000844
    2      0.000237 0.001082
    3      0.000096 0.001177
    4      0.000106 0.001283
    5      0.000119 0.001402
    6      0.000107 0.001509
       ...

When ``runner.log.write_result`` and ``runner.log.write_input`` are enabled,
the objective function value and the input parameters are appended from the fourth column onwards.

``status.pickle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If ``algorithm.checkpoint`` is set to true, the intermediate states are stored to ``status.pickle`` (or the filename specified by the ``algorithm.checkpoint_file`` parameter) for each rank of the algorithm layer in its subfolder
(when solver-level parallelism is used, the solver worker processes do not write it).
They are read when the execution is resumed.
The content of the file depends on the algorithm.

Checkpointing is supported by ``exchange``, ``pamc``, ``mapper``, ``random_search``, ``bayes``, and ``ttopt``.
``minsearch`` and ``global_search`` do not write a checkpoint even when ``algorithm.checkpoint`` is set to true.

The checkpoint feature allows you to resume calculations from the last saved state if a long calculation is interrupted.
To resume, run the program with the same input file and the same MPI process layout using ``odatse --resume input.toml``.
To continue from a previous run while extending the calculation, use ``odatse --cont input.toml``.
If you want to use a new random number sequence when resuming or continuing, add the ``--reset_rand`` option.
