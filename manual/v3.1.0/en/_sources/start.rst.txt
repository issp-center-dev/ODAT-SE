Installation of ODAT-SE
================================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 (>=3.9)

    - The following Python packages are required:
        - tomli >= 1.2 : For reading configuration files in TOML format
        - numpy >= 1.14 : For numerical calculations

    - Optional packages (required for specific optimization methods):

        - mpi4py : For parallelization and speedup when using grid search
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


How to run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ODAT-SE, the analysis is carried out by using a predefined optimization algorithm ``Algorithm`` and a direct problem solver ``Solver``.

.. code-block:: bash
    
    $ odatse input.toml

See :doc:`algorithm/index` for the predefined ``Algorithm`` and :doc:`solver/index` for the ``Solver``.

The direct problem solvers for analyses of experimental data of two-dimensional material structure are provided as separate modules.
To perform these analyses, you need to install the modules and the required software packages.
At present, the following modules are provided:

- Total Reflection High-energy Positron Diffraction (TRHEPD) -- odatse-STR package
  A high-precision method for surface structure analysis.

- Surface X-ray Diffraction (SXRD) -- odatse-SXRD package
  An X-ray diffraction method for investigating atomic arrangements at surfaces and interfaces.

- Low-energy Electron Diffraction (LEED) -- odatse-LEED package
  An electron diffraction method for studying crystal structures of solid surfaces.
  
If you want to prepare the ``Algorithm`` or ``Solver`` by yourself, use the ODAT-SE package.
See :doc:`customize/index` for details.

The program can be executed without installing ``odatse`` command; instead, run ``src/odatse_main.py`` script directly as follows. It would be convenient when you are rewriting programs.

.. code-block:: bash

   $ python3 src/odatse_main.py input.toml


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
