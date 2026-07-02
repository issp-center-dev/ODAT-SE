================================
Error Handling
================================

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
