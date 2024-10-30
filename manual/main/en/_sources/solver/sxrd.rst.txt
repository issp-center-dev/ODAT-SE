``sxrd`` solver
***********************************************

``sxrd`` is a ``Solver`` that uses ``sxrdcalc`` to calculate the Rocking curve by giving atomic positions :math:`x` , atomic occupancies, and Debye-Waller factor and finally returnes the error :math:`f(x)`  from the experimental Rocking curve.

Preparation
~~~~~~~~~~~~

The ``sxrdcalc`` is called from ``py2dmat``.
You will need to install ``sxrdcalc`` beforehand.
``sxrdcalc`` is available on GitHub at the following URL:

https://github.com/sxrdcalc/sxrdcalc

Access the site and download the source code from "Code" - "Download zip".
After unzipping the zip file, edit the Makefile to fit your computing environment, and then type ``make`` to create the ``sxrdcalc`` executable.

It is noted that the bulk data must be prepared in advance before running ``py2dmat`` (see the auxiliary file for solvers below for the format).

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input parameters are specified in subsections in the ``solver`` section (``config``, ``post``, ``param``, and ``reference``).

[``config``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``sxrd_exec_file``

  Format: string

  Description: Path to the solver ``sxrdcalc``.

- ``bulk_struc_in_file``

  Format: string

  Description: Input file name of the bulk structure.

 An example of the input input is given as follows:

.. code-block::

   [config]
   sxrd_exec_file = "../../sxrdcalc"
   bulk_struc_in_file = "sic111-r3xr3.blk"

[``param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``scale_factor``

  Format: float (default: 1.0)

  Description: The value of the target Rocking Curve and the scale of the Rocking Curve obtained from the simulation.

- ``opt_scale_factor``

  Format: bool (default: false)

  Description: Flag whether ``scale_factor`` should be optimized or not.
  
- ``type_vector``

  Format: list

  Description: A list of positive numbers which specifies the type of variables to be optimized.
  This list corresponds to the types specified in the [``param.atom``] subsection.
  If the type is the same, they are treated as the same variable.

[``param.domain``] subsection
-----------------------------------
In this section, parameters to create domains are specified.
You will need to define the domains you want to create.
In the [``param.domain.atom``] sub-subsection, paramters of the information in the domain are specified.

- ``domain_occupancy``

  Format: float

  Description: Occupancy of the whole domain.

[``param.domain.atom``] subsection
---------------------------------------------
This section needs to be defined as many times as the number of atoms you want to optimize belonging to the domain.
Note that the type, which represents the type of variable, must be a positive number.

- ``name``

  Format: string (can be duplicated)

  Description: The name of the atom to be optimized.

- ``pos_center``

  Format: list

  Description: Center coordinates of the atom. Describe in format of [:math:`x_0, y_0, z_0`](:math:`x_0, y_0, z_0` はfloat).

- ``DWfactor``

  Format: float

  Description: Debye-Waller factor (in the unit of :math:`\text{Å}^{2}` ).

- ``occupancy``

  Format: float (default: 1.0)

  Description: Atom occupancy.

- ``displace_vector`` (can be omitted)

  Format: list of lists

  Description: A vector that defines the direction in which the atoms are moved. A maximum of three directions can be specified.
  Define displacement vectors and initial values in each list as [type, :math:`D_{i1}, D_{i2}, D_{i3}` ](type is int, :math:`D_{i1}, D_{i2}, D_{i3}` is float type).
  Follwing the specified information, :math:`l_{type}` is varied as 
  :math:`dr_i = (D_{i1} \vec{a} + D_{i2} \vec{b} + D_{i3} \vec{c}) * l_{type}` 
  ( :math:`\vec{a}, \vec{b}, \vec{c}` is a unit lattice vector defined in ``bulk_struc_in_file`` or ``struc_in_file``).
       
- ``opt_DW`` (can be omitted)

  Format: list

  Description: Sets the scale at which the Debye-Waller coefficient is varied.
  It is defined as [type, scale].
 
- ``opt_occupancy``

  Format: int

  Description: If defined, the occupancy changes. The specified variable represents the type.


An example of an input file is given as follows:

.. code-block::

   [param]
   scale_factor = 1.0
   type_vector = [1, 2]

   [[param.domain]]
   domain_occupancy = 1.0
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.00000000, 0.00000000, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.33333333, 0.66666667, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.66666667, 0.33333333, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.33333333, 0.33333333, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[2, 0.0, 0.0, 1.0]]
  

[``reference``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``f_in_file``

  Format: string

  Description: Path to the input file for the target locking curve.

  
Reference file for Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Target reference file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file containing the data to be targeted to fit. The path is specified by ``f_in_file`` in the [``reference``] section.
For each line, ``h k l F sigma`` is written. Here, ``h, k, l`` are the wavenumbers, ``F`` is the intensity, and ``sigma`` is the uncertainty of ``F``.
An example file is shown below.

.. code-block::
   
   0.000000 0.000000 0.050000 572.805262 0.1 
   0.000000 0.000000 0.150000 190.712559 0.1 
   0.000000 0.000000 0.250000 114.163340 0.1 
   0.000000 0.000000 0.350000 81.267319 0.1 
   0.000000 0.000000 0.450000 62.927325 0.1 
   ...

Bulk structure file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file containing the bulk structure data. The path is specified by ``bulk_struc_in_file`` in the [``config``] section.
The first line is a comment, the second line is ``a b c alpha beta gamma``.
Here, ``a`` , ``b``, and ``c`` are the lattice constants of the unit cells, and ``alpha``, ``beta`` , and ``gamma`` are their angles.
The third and subsequent lines specify the ``atomsymbol r1 r2 r3 DWfactor occupancy``.
Here, ``atomsymbol`` is the atom species, ``r1``, ``r2``, and ``r3`` are the position coordinates of the atom, ``DWfactor`` is the Debye-Waller factor, and ``occupancy`` is the occupancy rate.
An example file is given below.

.. code-block::

   # SiC(111) bulk
   5.33940 5.33940  7.5510487  90.000000 90.000000 120.000000
   Si 0.00000000   0.00000000   0.00000000 0.0 1.0
   Si 0.33333333   0.66666667   0.00000000 0.0 1.0
   Si 0.66666667   0.33333333   0.00000000 0.0 1.0
   C  0.00000000   0.00000000   0.25000000 0.0 1.0
   ...
   
Output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``sxrd``, the output files are stored in the folder with the rank number.
Here is a description of the files that are output by ``py2dmat``.

``stdout``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard output by ``sxrd`` is described.
For sxrd's Least square fitting, we give variables as initial parameters and calculate the Rfactor for a 1-shot calculation (number of iterations = 0).
The Rfactor is written in R under Fit results.
Here is an example of the output.

.. code-block::

    ---------------------------------------
    Program py2dmat/mapper_sxrd/sxrdcalc for surface x-ray diffraction calculations.
    Version 3.3.3 - August 2019


     Inputfile: lsfit.in
    Least-squares fit of model to experimental structure factors.

    ...

    Fit results:
    Fit not converged after 0 iterations.
    Consider increasing the maximum number of iterations or find better starting values.
    chi^2 = 10493110.323318, chi^2 / (degree of freedom) = 223257.666454 (Intensities)
    chi^2 = 3707027.897897, chi^2 / (degree of freedom) = 78872.933998 (Structure factors)
    R = 0.378801

    Scale factor:   1.00000000000000 +/- 0.000196
    Parameter Nr. 1:   3.500000 +/- 299467640982.406067
    Parameter Nr. 2:   3.500000 +/- 898402922947.218384

    Covariance matrix:
              0            1            2
     0  0.0000000383 20107160.3315223120 -60321480.9945669472
     1  20107160.3315223120 89680867995567253356544.0000000000 -269042603986701827178496.0000000000
     2  -60321480.9945669472 -269042603986701827178496.0000000000 807127811960105615753216.0000000000

