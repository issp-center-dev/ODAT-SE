Installation of ODAT-SE
================================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 (>=3.6.8)

    - The following Python packages are required.
        - tomli >= 1.2
        - numpy >= 1.14

    - Optional packages

        - mpi4py (required for grid search)
        - scipy (required for Nelder-Mead method)
        - physbo (>=0.3, required for Baysian optimization)


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

- odatse-STR module for Total Refrection High-energy Positron Diffraction (TRHEPD)

- odatse-SXRD module for Surface X-ray Diffraction (SXRD)

- odatse-LEED module for Low-energy Electron Diffraction (LEED)
  
If you want to prepare the ``Algorithm`` or ``Solver`` by yourself, use the ODAT-SE package.
See :doc:`customize/index` for details.

The program can be executed without installing ``odatse`` command; instead, run ``src/odatse_main.py`` script directly as follows. It would be convenient when you are rewriting programs.

.. code-block:: bash

   $ python3 src/odatse_main.py input


How to uninstall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please type the following command:

.. code-block:: bash

    $ python3 -m pip uninstall ODAT-SE
