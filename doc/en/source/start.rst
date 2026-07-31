Installation of ODAT-SE
================================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 (>=3.9)

  - The following Python packages are required:
    - tomli >= 1.2 : For reading configuration files in TOML format
    - numpy >= 1.14 : For numerical calculations
    - matplotlib >= 3 : For visualizing calculation results and plotting in the post-processing tools

  - Optional packages (required for specific optimization methods):

    - mpi4py : For MPI parallelization in algorithms such as ``mapper``, ``random_search``, ``exchange``, ``pamc``, and ``global_search``
    - scipy : For ``minsearch`` (local optimization such as the Nelder-Mead method) and ``global_search`` (global optimization)
    - physbo (>=2.0) : For Bayesian optimization
    - tqdm : For showing progress bars in the post-processing tools


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

Quick start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a check that the installation works, let us minimize an analytical function.
Create ``input.toml`` with the following contents (no sample files need to be downloaded):

.. code-block:: toml

    [base]
    dimension = 2
    output_dir = "output"

    [solver]
    name = "analytical"
    function_name = "himmelblau"

    [algorithm]
    name = "minsearch"
    seed = 12345

    [algorithm.param]
    min_list = [-6.0, -6.0]
    max_list = [ 6.0,  6.0]
    initial_list = [0, 0]

.. note::
   ``minsearch`` requires scipy.
   ``python3 -m pip install ODAT-SE`` alone does not install scipy, so use
   ``python3 -m pip install 'ODAT-SE[min_search]'`` (or ``'ODAT-SE[all]'``) instead.
   If you get a ``ModuleNotFoundError``, see :doc:`faq/error`.

Run the following command in the same directory. The calculation finishes in a few seconds.

.. code-block:: bash

    $ odatse input.toml

The optimization result is written to ``output/res.txt``:

.. code-block::

    fx = 4.2278370361994904e-08
    x1 = 2.9999669562950175
    x2 = 1.9999973389336225

One of the minima of the Himmelblau function, :math:`(3, 2)` (with the function value :math:`0`), is obtained correctly.
See :doc:`tutorial/index` for detailed explanations and the usage of the other algorithms.

Command-line options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``odatse`` command provides options to control the execution mode (such as ``--resume`` to restart from a checkpoint and ``--cont`` to extend a finished calculation) and options to control the assignment of MPI processes (``--nalg``, ``--nsolve``).

Example:

.. code-block:: bash

  $ odatse --resume input.toml

See :doc:`manual/command` for the complete list and the details of the options.

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
- ``global_search`` can evaluate candidate points in parallel with MPI for differential evolution and shgo (but not for direct)
- ``bayes`` can also use MPI parallel execution when ``mpi4py`` is available
- During parallel execution, each process has its own random number sequence (see ``seed`` and ``seed_delta`` parameters)
- For algorithms that support checkpointing, a checkpoint file is created for each rank of the algorithm layer

Execution example:

.. code-block:: bash

  $ mpirun -np 4 odatse input.toml

The ``-np 4`` part specifies the number of processes to use. Adjust according to the number of cores available.

Depending on your environment, you may need to use ``mpiexec`` or other commands, or execute MPI programs through a job scheduler. Large-scale computing centers in particular may have system-specific execution methods. Please refer to the manual for your environment for details.

.. note::
  Parallelization efficiency varies by algorithm. For example, with ``exchange``, it is efficient to use the same number of processes as replicas or fewer.
