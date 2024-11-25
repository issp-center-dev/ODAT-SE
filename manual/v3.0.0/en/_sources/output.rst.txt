Output files
=====================

See :doc:`solver/index` and :doc:`algorithm/index` for the output files of each ``Solver`` and ``Algorithm``.

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
     - file_CM = 0.0009563499997966574
     - submit = 1.3224223930001244
    #post
     total = 0.000595873999941432


``runner.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The log information about solver calls for each MPI rank is outputted.
These files will be output under the subfolder of each rank.
The output is only available when the ``runner.log.interval`` parameter is a positive integer in the input.

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
