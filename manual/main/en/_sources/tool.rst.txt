Related Tools
=====================

``py2dmat_neighborlist``
*****************************

This tool generates a neighborhood-list file from the mesh file.

When you install py2dmat via ``pip`` command, ``py2dmat_neighborlist`` is also installed under the ``bin``.
A python script ``src/py2dmat_neighborlist.py`` is also available.

Usage
~~~~~~~~~~~~~~~~

Pass a path to the mesh file as an argument.
The filename of the generated neighborhood-list file is specified by ``-o`` option.

.. code-block:: bash

  $ py2dmat_neighborlist -o neighborlist.txt MeshData.txt

  Or

  $ python3 src/py2dmat_neighborlist.py -o MeshData.txt


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

``tool/to_dft/to_dft.py``
******************************

This tool generates input data for `Quantum Espresso (QE) <https://www.quantum-espresso.org/>`_ , a first-principles electronic structure calculation software, from the atomic structures of (001) and (111) surface models of systems with Si isotetrahedral bond networks. This is used to validate the obtained structure and to obtain microscopic information such as the electronic state. In order to eliminate the influence of dangling bond-derived electrons from the opposite surface of interest, we use a technique called hydrogen termination, in which a hydrogen atom is placed at the position of the lowest dangling bond.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python3 >= 3.6

The following packages are required:

- `Atomic Simulation Environment(ASE) <https://wiki.fysik.dtu.dk/ase>`_ (>= 3.21.1)
- Numpy
- Scipy
- Matplotlib
  
Overview of this tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input file including the information such as the name of the structure file (XYZ format) and the lattice vector information to represent the two-dimensional periodic structure is read in, and the coordinates of the lowest layer and the next layer of atoms are extracted from the obtained coordinate data.
The bottom layer atoms are removed, and H atoms are placed at the corresponding positions to create a model with the distance to the next layer atoms adjusted to a tetrahedral structure (for example, the distance to a silane molecule in the case of Si).
The hydrogen-terminated model is saved in XYZ format, and a cif file and an input file for Quantum Espresso (QE) are also created.
If you have QE installed, you can also run the calculation as is.

Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Prepare an XYZ file for reference.

In the following, we will use the file ``surf_bulk_new111.xyz`` in the folder ``tool/todft/sample/111``. The contents of the file are as follows.

.. code-block::

   12
    surf.txt             / bulk.txt
    Si    1.219476    0.000000    4.264930
    Si    6.459844    0.000000    4.987850
    Si    1.800417    1.919830    3.404650
    Si    5.878903    1.919830    3.404650
    Si    3.839660    1.919830    2.155740
    Si    0.000000    1.919830    1.900440
    Si    3.839660    0.000000    0.743910
    Si    0.000000    0.000000    0.597210
    Si    1.919830    0.000000   -0.678750
    Si    5.759490    0.000000   -0.678750
    Si    1.919830    1.919830   -2.036250
    Si    5.759490    1.919830   -2.036250

2. Next, create an input file for setting the various parameters.
   
The file format of the input file is ``toml``. The following section describes the contents of the input file using ``input.toml`` in the ``tool/todft/sample/111`` folder. The contents of the file are as follows.

.. code-block::

   [Main]
   input_xyz_file = "surf_bulk_new111.xyz"
   output_file_head = "surf_bulk_new111_ext"
   [Main.param]
   z_margin = 0.001
   slab_margin = 10.0
   r_SiH = 1.48 #angstrom
   theta = 109.5 #H-Si-H angle in degree
   [Main.lattice]
   unit_vec = [[7.67932, 0.00000, 0.00000], [0.00000, 3.83966, 0.00000]]
   [ASE]
   solver_name = "qe"
   kpts = [3,3,1]        # sampling k points (Monkhorst-Pack grid)
   command = "mpirun -np 4 ./pw.x -in espresso.pwi > espresso.pwo"
   [Solver]
   [Solver.control]
   calculation='bands' # 'scf','realx','bands',...
   pseudo_dir='./'     # Pseudopotential directory
   [Solver.system]
   ecutwfc = 20.0        # Cut-off energy in Ry
   nbands=33           # # of bands (only used in band structure calc
   [Solver.pseudo]
   Si = 'Si.pbe-mt_fhi.UPF'
   H = 'H.pbe-mt_fhi.UPF'

The input file consists of three sections: ``Main``, ``ASE``, and ``Solver``.
Below is a brief description of the variables for each section.

``Main`` section
------------------------
This section contains settings related to the parameters required for hydrogen termination.

- ``input_xyz_file``
  
  Format: string

  Description: Name of the xyz file to input

- ``output_file_head``

  Format: string

  Description: Header for output files (xyz and cif files)

``Main.Param`` section
-----------------------------

- ``z_margin``

  Format: float

  Description: Margin used to extract the lowest and second-to-last atoms. For example, if the z-coordinate of the atom in the bottom layer is ``z_min``, the atoms in ``z_min - z_margin <= z <= z_min + z_margin`` will be extracted.

- ``slab_margin``

  Format: float

  Description: Margin for tuning the size of the slab. If the z-coordinates of the atoms in the bottom and top layers are ``z_min`` , ``z_max``, then the slab size is given by ``z_max-z_min+slab_margin``.
  
- ``r_SiH``

  Format: float

  Description: The distance (in :math:`\mathrm{\mathring{A}}`) between a vertex (e.g. Si) and H of a tetrahedral structure.
  
- ``theta``

  Format: float

  Description: The angle between the vertex and H of the tetrahedral structure (e.g. Si-H-Si).

``Main.lattice`` section
-----------------------------

- ``unit_vec``

  Format: list

  Description: Specify a unit vector that forms a 2D plane (ex. ``unit_vec = [[7.67932, 0.00000, 0.00000], [0.00000, 3.83966, 0.00000]]``).

  
``ASE`` section
------------------------
This section specifies parameters related to ``ASE``.

- ``solver_name``

  Format: string

  Description: The name of the solver. Currently, only ``qe`` is given.

- ``kpts``

  Format: list

  Description: Specify the k-points to be sampled (Monkhorst-Pack grid).

- ``command``

  Format: string

  Description: Set the command used to run the solver.

``Solver`` section
------------------------
In this section, parameters related to ``Solver`` are specified.
You will need to specify this if you want to perform first-principles calculations directly using ASE.
Basically, the configuration is the same as the one specified in the input file of each solver.
For example, in the case of QE, ``Solver.control`` contains the parameters to be set in the ``control`` section of QE.
  
3. Execute the following command.

.. code-block::

    python3 to_dft.py input.toml

After finishing calculations, the following files are generated:

- ``surf_bulk_new111_ext.xyz``
- ``surf_bulk_new111_ext.cif``
- ``espresso.pwi``
  
If the path to the QE and pseudopotential is set in the input file, the first-principle calculation will be performed as is. If not, the ab initio calculation will not be performed and you will get the message ``Calculation of get_potential_energy is not normally finished.`` at the end, but the above file will still be output.

The following is a description of the output file.

- ``surf_bulk_new111_ext.xyz``

The output is the result of the replacement of the lowest level atom with H and the addition of H to form a tetrahedral structure.
The actual output is as follows.

.. code-block::

    14
    Lattice="7.67932 0.0 0.0  0.0 3.83966 0.0  0.0 0.0 17.0241" Properties=species:S:1:pos:R:3 pbc="T T T"
    Si  1.219476  0.000000  4.264930
    Si  6.459844  0.000000  4.987850
    Si  1.800417  1.919830  3.404650
    Si  5.878903  1.919830  3.404650
    Si  3.839660  1.919830  2.155740
    Si  0.000000  1.919830  1.900440
    Si  3.839660  0.000000  0.743910
    Si  0.000000  0.000000  0.597210
    Si  1.919830  0.000000  -0.678750
    Si  5.759490  0.000000  -0.678750
    H  1.919830  -1.208630  -1.532925
    H  1.919830  1.208630  -1.532925
    H  5.759490  -1.208630  -1.532925
    H  5.759490  1.208630  -1.532925

This file can be read by appropriate visualization software as ordinary XYZFormat coordinate data, but the lattice vector information of the periodic structure is written in the place where comments are usually written. You can also copy the data of "element name + 3D coordinate" from the third line of the output file to the input file of QE.

``espresso.pwi`` is the input file for QE's scf calculation, and structural optimization and band calculation can be done by modifying this file accordingly. For details, please refer to the `QE online manual <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`_ .
