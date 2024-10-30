Related Tools
================================

``odatse_neighborlist``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tool generates a neighborhood-list file from the mesh file.

When you install ODAT-SE via ``pip`` command, ``odatse_neighborlist`` is also installed under the ``bin``.
A python script ``src/odatse_neighborlist.py`` is also available.

Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a path to the mesh file as an argument.
The filename of the generated neighborhood-list file is specified by ``-o`` option.

.. code-block:: bash

  $ odatse_neighborlist -o neighborlist.txt MeshData.txt

  Or

  $ python3 src/odatse_neighborlist.py -o MeshData.txt


The following command-line options are available.

- ``-o output`` or ``--output output``

  - The filename of output (default: ``neighborlist.txt``)

- ``-u "unit1 unit2..."`` or ``--unit "unit1 unit2..."``

  - Length scale for each dimension of coordinate (default: 1.0 for all dims)

    - Put values splitted by whitespaces and quote the whole

      - e.g.) ``-u "1.0 0.5"``

  - Each dimension of coordinate is divided by the corresponding ``unit``.

- ``-r radius`` or ``--radius radius``

  - A pair of nodes where the Euclidean distance is less than ``radius`` is considered a neighborhood (default: 1.0)
  - Distances are calculated in the space after coordinates are divided by ``-u``

- ``-q`` or ``--quiet``

  - Do not show a progress bar
  - Showing a progress bar requires ``tqdm`` python package

- ``--allow-selfloop``

  - Include :math:`i` in the neighborhoods of :math:`i` itself

- ``--check-allpairs``

  - Calculate distances of all pairs
  - This is for debug


MPI parallelization is available.

