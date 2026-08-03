================================
Troubleshooting
================================

This page collects typical errors and their remedies, organized by symptom.
For installation problems, see :doc:`install`. For tuning when the search does not progress well, see :doc:`montecarlo`.

The headings below show the body of the error message.
Errors that ODAT-SE raises for invalid input and the like (the ``odatse.exception.Error`` family)
are printed to standard error by the ``odatse`` command with an ``ERROR:`` prefix
(and further prefixed with ``[rank N]`` for an error raised on one specific rank under MPI).
Other exceptions (``ModuleNotFoundError``, ``ValueError``, ``RuntimeError``, ...) propagate
as ordinary Python tracebacks.

Errors at startup
================================

``ModuleNotFoundError: No module named 'scipy'`` etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms depend on optional packages, and a ``ModuleNotFoundError`` is raised at run time if they are not installed.

- ``scipy`` : required by ``minsearch`` (local optimization such as the Nelder-Mead method) and ``global_search`` (global optimization)
- ``physbo`` : required by ``bayes`` (Bayesian optimization)
- ``mpi4py`` : required for MPI parallel execution via ``mpiexec``

Install the package shown in the error message individually, or install all the optional packages at once:

.. code-block:: bash

    $ python3 -m pip install 'ODAT-SE[all]'

See the prerequisites section of :doc:`../start` for details.


``failed to load 'input.toml' on rank 0: ...``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input file is not found, or it contains a TOML syntax error.
The cause (missing file, line number of the syntax error, etc.) is shown in the latter half of the error message; fix the input accordingly.
The path of the input file is interpreted relative to the directory in which the ``odatse`` command is executed.


``section [...] does not appear in input``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A required section is missing from the input file.
Check that the ``[base]``, ``[solver]``, and ``[algorithm]`` sections are all defined.
See :doc:`../input/index` for the specification of the input file.


``Unknown solver`` / ``unknown algorithm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``name`` in the ``[solver]`` or ``[algorithm]`` section specifies an undefined name.
Check the spelling. See the ``name`` entry of :doc:`../input/algorithm` for the available algorithm names.
Note that the name of the grid search algorithm is ``mapper`` (not ``mapper_mpi``).


``mesh_path not found: ...``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In grid-search type algorithms, the mesh definition file (``mesh_path`` in ``[algorithm.param]``) is not found.
``mesh_path`` is resolved relative to the directory in which the ``odatse`` command is executed (the root directory).
Check the location of the file relative to the execution directory.


``ValueError`` concerning ``Tmin`` / ``Tmax`` / ``bmin`` / ``bmax``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a problem in the temperature specification of ``exchange`` or ``pamc``. Typical messages and causes:

- ``both Tmin/Tmax and bmin/bmax are defined`` : Both the temperatures (``Tmin``/``Tmax``) and the inverse temperatures (``bmin``/``bmax``) are specified. Use only one of them.
- ``neither Tmin/Tmax nor bmin/bmax are defined`` : No temperature range is specified.
- ``bmin must be greater than 0.0 when Tlogspace is True`` : ``bmin = 0`` cannot be used with the logarithmic scale (``Tlogspace = true``). Use a positive value or set ``Tlogspace = false``.

See :doc:`../algorithm/exchange` and :doc:`../algorithm/pamc` for the meaning of the parameters.


Errors during execution
================================

``RuntimeError`` from the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the solver raises errors in certain parameter regions, set ``ignore_error = true`` in the ``[runner]`` section. This returns NaN for error-producing parameters and continues the calculation.

.. code-block:: toml

    [runner]
    ignore_error = true

However, this is a workaround. It is preferable to investigate the cause of the error and exclude problematic regions using search range or constraint settings (``[runner.limitation]``).


``mpiexec`` fails with "not enough slots"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the number of MPI processes exceeds the number of CPU cores, Open MPI fails with
"There are not enough slots available in the system".
To launch more processes than cores, add the ``--oversubscribe`` option:

.. code-block:: bash

    $ mpiexec -np 10 --oversubscribe odatse input.toml


Errors on restart
================================

How to restart from a checkpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a long calculation is interrupted, it can be restarted from the middle using the checkpointing feature.
Checkpointing is supported by ``exchange``, ``pamc``, ``mapper``, ``random_search``, ``bayes``, and ``ttopt``
(``minsearch`` and ``global_search`` write no checkpoint even with ``checkpoint = true``).

First, run with checkpointing enabled:

.. code-block:: toml

    [algorithm]
    checkpoint = true
    checkpoint_steps = 1000
    checkpoint_interval = 3600  # every hour

If the calculation is interrupted, run again with the ``--resume`` option and the same input file. It restarts from the last checkpoint.

.. code-block:: bash

    $ odatse --resume input.toml

Without the option, the calculation starts from the beginning (same as the default ``--init``). To extend a finished calculation, use ``--cont``.
``--cont`` is supported by ``exchange``, ``pamc``, ``bayes``, ``ttopt``, and ``random_search``.
``mapper`` supports ``--resume`` only, because its search points are fixed by the grid, and raises an error when ``--cont`` is given.
See :doc:`../manual/command` for the details of the command-line options.


``checkpoint file ... does not exist``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You tried to restart with ``--resume`` but the checkpoint file is not found. Check the following:

- The original calculation was run with ``checkpoint = true``, and the algorithm supports checkpointing.
- You are running in the same directory with the same input file (the same ``output_dir``) as the original calculation.
- You are running with the same MPI process layout (including ``--nalg`` / ``--nsolve``) as the original calculation.
- The calculation was not interrupted before the first checkpoint was saved (i.e., before the first ``checkpoint_steps`` steps or ``checkpoint_interval`` seconds).
