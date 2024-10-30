Install of py2dmat
=============================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 (>=3.6.8)

    - The following Python packages are required.
        - tomli >= 1.2
        - numpy >= 1.14

    - Optional packages

        - mpi4py (required for grid search)
        - scipy (required for Nelder-Mead method)
        - physbo (>=0.3, required for Baysian optimization)

How to download and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can install the ``py2dmat`` python package and the ``py2dmat`` command using the method shown below.

- Installation using PyPI (recommended)

    - ``python3 -m pip install py2dmat``

        - ``--user``  option to install locally (``$HOME/.local``)

        - If you use ``py2dmat[all]``, optional packages will be installed at the same time.
	  
- Installation from source code

    #. ``git clone https://github.com/issp-center-dev/2DMAT``
    #. ``python3 -m pip install ./2DMAT``

        - The ``pip`` version must be 19 or higher (can be updated with ``python3 -m pip install -U pip``).

- Download the sample files

    -  Sample files are included in the source code.
    - ``git clone https://github.com/issp-center-dev/2DMAT``

Note that among the direct problem solvers used in ``py2dmat``,
the following solver must be installed separately:

- TRHEPD forward problem solver (``sim-trhepd-rheed``)
- SXRD forward problem solver (``sxrdcalc``)
- LEED forward problem solver (``satleed``)

Please refer to the tutorials of each solver for installation details.

How to run
~~~~~~~~~~~~~
In ``py2dmat`` , the analysis is done by using a predefined optimization algorithm ``Algorithm`` and a direct problem solver ``Solver`` ::
    
    $ py2dmat input.toml

See :doc:`algorithm/index` for the predefined ``Algorithm`` and :doc:`solver/index` for the ``Solver``.

If you want to prepare the ``Algorithm`` or ``Solver`` by yourself, use the ``py2dmat`` package.
See :doc:`customize/index` for details.

How to uninstall
~~~~~~~~~~~~~~~~~~~~~~~~
Please type the following command:

.. code-block:: bash

    $ python3 -m pip uninstall py2dmat
