Installation of ODAT-SE
================================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 (>=3.9)

    - The following Python packages are required:
        - tomli >= 1.2 : For reading configuration files in TOML format
        - numpy >= 1.14 : For numerical calculations

    - Optional packages (required for specific optimization methods):

        - mpi4py : For MPI parallelization in algorithms such as ``mapper``, ``random_search``, ``exchange``, and ``pamc``
        - scipy : For optimization using the Nelder-Mead method
        - physbo (>=2.0) : For Bayesian optimization


How to download and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the ODAT-SE python package and the ``odatse`` command following the instructions shown below.

- Installation using PyPI (recommended)

    - ``python3 -m pip install ODAT-SE``

        - ``--user``  option to install locally (``$HOME/.local``)

        - If you use ``ODAT-SE[all]``, optional packages will be installed at the same time.
	  
- Installation from source code

    #. ``git clone https://github.com/issp-center-dev/ODAT-SE``
    #. ``python3 -m pip install ./ODAT-SE``

        - The ``pip`` version must be 19 or higher (can be updated with ``python3 -m pip install -U pip``).

- Download the sample files

    -  Sample files are included in the source code.
    - ``git clone https://github.com/issp-center-dev/ODAT-SE``

Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that the installation was completed successfully, run the following commands:

.. code-block:: bash

    $ odatse --version
    $ odatse --help


How to uninstall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To uninstall the ODAT-SE module, please run the following command:

.. code-block:: bash

    $ python3 -m pip uninstall ODAT-SE

If you need to uninstall related optional packages individually, you can run similar commands for each package.

How to run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ODAT-SE, the analysis is carried out by using a predefined optimization algorithm ``Algorithm`` and a direct problem solver ``Solver``.

.. code-block:: bash
    
    $ odatse input.toml

See :doc:`algorithm/index` for the predefined ``Algorithm`` and :doc:`solver/index` for the ``Solver``.

Command-line options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``odatse`` command provides options to control the execution mode.

- ``--init``

  Starts a fresh calculation. This is the default behavior.

- ``--resume``

  Restores the interrupted state from checkpoint files and resumes the run.

- ``--cont``

  Continues a previous calculation and extends it from the saved state.

- ``--reset_rand``

  Used together with ``--resume`` or ``--cont`` to start with a new random number sequence.

Examples:

.. code-block:: bash

    $ odatse --resume input.toml
    $ odatse --cont --reset_rand input.toml

Wrapper packages for using direct problem solvers for two-dimensional material structure analysis from ODAT-SE are provided as separate modules.
To perform these analyses, you need to install the wrapper package and the direct problem solver itself.
At present, the following wrapper packages are available:

- `odatse-STR <https://github.com/2DMAT/odatse-STR>`_ -- Total Reflection High-energy Positron Diffraction (TRHEPD)
  A high-precision method for surface structure analysis.

- `odatse-SXRD <https://github.com/2DMAT/odatse-SXRD>`_ -- Surface X-ray Diffraction (SXRD)
  An X-ray diffraction method for investigating atomic arrangements at surfaces and interfaces.

- `odatse-LEED <https://github.com/2DMAT/odatse-LEED>`_ -- Low-energy Electron Diffraction (LEED)
  An electron diffraction method for studying crystal structures of solid surfaces.
  
If you want to prepare the ``Algorithm`` or ``Solver`` by yourself, use the ODAT-SE package.
See :doc:`customize/index` for details.

The program can be executed without installing ``odatse`` command; instead, run ``src/odatse_main.py`` script directly as follows. It would be convenient when you are rewriting programs.

.. code-block:: bash

   $ python3 src/odatse_main.py input.toml

MPI Parallel Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ODAT-SE supports parallel computation using MPI. Using MPI, you can speed up calculations by utilizing multiple processes.

- ``mapper``, ``random_search``, ``exchange``, and ``pamc`` can benefit from MPI parallelization
- ``bayes`` can also use MPI parallel execution when ``mpi4py`` is available
- During parallel execution, each process has its own random number sequence (see ``seed`` and ``seed_delta`` parameters)
- Checkpoint files are created for each process

Execution example:

.. code-block:: bash

    $ mpirun -np 4 odatse input.toml

The ``-np 4`` part specifies the number of processes to use. Adjust according to the number of cores available.

Depending on your environment, you may need to use ``mpiexec`` or other commands, or execute MPI programs through a job scheduler. Large-scale computing centers in particular may have system-specific execution methods. Please refer to the manual for your environment for details.

.. note::
   Parallelization efficiency varies by algorithm. For example, with ``exchange``, it is efficient to use the same number of processes as replicas or fewer.
