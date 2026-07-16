================================
Error Handling
================================

``ModuleNotFoundError: No module named 'scipy'`` etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms depend on optional packages, and a ``ModuleNotFoundError`` is raised at run time if they are not installed.

- ``scipy`` : required by ``minsearch`` (Nelder-Mead method)
- ``physbo`` : required by ``bayes`` (Bayesian optimization)
- ``mpi4py`` : required for MPI parallel execution via ``mpiexec``

Install the package shown in the error message individually, or install all the optional packages at once:

.. code-block:: bash

    $ python3 -m pip install 'ODAT-SE[all]'

See the prerequisites section of :doc:`../start` for details.


``RuntimeError`` from the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the solver raises errors in certain parameter regions, set ``ignore_error = true`` in the ``[runner]`` section. This returns NaN for error-producing parameters and continues the calculation.

.. code-block:: toml

    [runner]
    ignore_error = true

However, this is a workaround. It is preferable to investigate the cause of the error and exclude problematic regions using search range or constraint settings (``[runner.limitation]``).


How to resume from a checkpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For long calculations that are interrupted, the checkpoint feature allows resuming from where it left off.

First, enable checkpointing:

.. code-block:: toml

    [algorithm]
    checkpoint = true
    checkpoint_steps = 1000
    checkpoint_interval = 3600  # every 1 hour

If the calculation is interrupted, run the program again with the same input file and the ``--resume`` option to restart from the last checkpoint:

.. code-block:: bash

    $ odatse --resume input.toml

Without an option the program starts from the beginning (equivalent to ``--init``, the default). Use ``--cont`` to continue a finished run for additional steps.
