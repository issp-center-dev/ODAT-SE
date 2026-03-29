================================
Installation and Execution
================================

Does it work with Python 3.8 or earlier?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.9 or later is required. Check your version with ``python3 --version``.


The ``odatse`` command is not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The install location may not be in your PATH. Check the following:

.. code-block:: bash

    # Check installation status
    $ python3 -m pip show ODAT-SE

    # Check install location
    $ python3 -m pip show ODAT-SE | grep Location

If installed with the ``--user`` option, make sure ``~/.local/bin`` is in your PATH.
You can also run directly:

.. code-block:: bash

    $ python3 -m odatse input.toml


Can I run without MPI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, most algorithms can run in serial without MPI.
If mpi4py is not installed, ODAT-SE automatically falls back to serial mode.

However, the following algorithms require or strongly benefit from MPI parallelization:

- ``exchange``: Requires at least as many processes as replicas
- ``mapper``: Distributes grid points across processes (runs with 1 process but is very slow for large grids)
