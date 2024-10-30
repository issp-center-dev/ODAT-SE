``sim-trhepd-rheed`` solver
***********************************************

.. _sim-trhepd-rheed: https://github.com/sim-trhepd-rheed/sim-trhepd-rheed

``sim-trhepd-rheed`` is a ``Solver`` that uses sim-trhepd-rheed_ to calculate the diffraction rocking curve from the atomic position :math:`x` and returns the deviation from the experimental rocking curve as :math:`f(x)`. 

Preparation
~~~~~~~~~~~~

You will need to install sim-trhepd-rheed_ beforehand.

1. Download the source code from the official ``sim-trhepd-rheed`` website. 
2. Move to ``sim-trhepd-rheed/src`` folder and make ``bulk.exe`` and ``surf.exe`` by using ``make``.

Before running ``py2dmat``, run ``bulk.exe`` to create the bulk data.
The ``surf.exe`` is called from ``py2dmat``.

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input parameters can be specified in subsections ``config``, ``post``, ``param``, ``reference`` in ``solver`` section.

[``solver``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``generate_rocking_curve``

  Format: boolean (default: false)

  Description: Whether to generate ``RockingCurve_calculated.txt``.
  If ``true``, ``RockingCurve_calculated.txt`` will be generated in the working directory ``Log%%%_###``.
  Note that if ``remove_work_dir`` (in [``post``] subsection) is ``true``, ``Log%%%_###`` will be removed.

[``solver.config``] subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``surface_exec_file``

  Format: string (default: "surf.exe")

  Description: Path to ``sim-trhepd-rheed`` surface reflection solver ``surf.exe``.

- ``surface_input_file``

  Format: string (default: "surf.txt")

  Description: Input file for surface structure.

- ``bulk_output_file``

  Format: string (default: "bulkP.b")

  Description: Output file for bulk structure.

- ``surface_output_file``

  Format: string (default: "surf-bulkP.s")

  Description: Output file for surface structure.

- ``calculated_first_line``

  Format: integer (default: 5)

  Description: In the output file, the first line to be read as D(x).
  The last line is automatically calculated from the number of the reference data.

- ``calculated_info_line``

  Format: integer (default: 2)

  Description: In the output file, the line contains the information of the calculated data -- the number of glancing angles (second column) and the number of beams (third column).
  
- ``cal_number``

  Format: Integer or List of integers

  Description: In the output file, the columns to be read as D(x). Multiple columns can be specified (many-beam condition).


[``solver.post``] subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This subsection is used to the postprocess -- to specify how to calculate the objective function, that is, the deviation between the experimental and computational data, and to draw the rocking curve.

- ``Rfactor_type``

  Format: string ("A" or "B", default: "A")

  Description: This parameter specifies how to calculate the R-factor to be minimized.
  Let :math:`n` be the number of dataset, :math:`m` be the number of glancing angles, and :math:`v^{(n)} = (v^{(n)}_{1}, v^{(n)}_{2}, \dots ,v^{(n)}_{m})` be the calculated data.
  With the weights of the beams, :math:`w^{(j)}`, R-factors is defined as follows:

    - "A" type:

      .. math::

        R = \sqrt{ \sum_{j}^{n} w^{(j)} \sum_{i}^{m} \left(u^{(j)}_{i}-v^{(j)}_{i}\right)^{2} }

    - "A2" type:

      .. math::

        R^{2} = \sum_{j}^{n} w^{(j)} \sum_{i}^{m} \left(u^{(j)}_{i}-v^{(j)}_{i}\right)^{2}

    - "B" type:

      .. math::

        R = \frac{\sum_{i}^{m} \left(u^{(1)}_{i}-v^{(1)}_{i}\right)^{2}}{\sum_{i}^{m} \left(u^{(1)}_{i}\right)^{2} + \sum_{i}^{m} (v^{(1)}_{i})^2}

      - "B" type is available only for the single dataset (:math:`n=1`).


- ``normalization``

  Format: string ("TOTAL" or "MANY_BEAM")

  Description: This parameter specifies how to normalize the experimental and computational data vectors.

  - "TOTAL"

    - To normalize the data as the summation is 1.
    - The number of dataset should be one (the number of ``cal_number`` should be one).

  - "MANY_BEAM"

    - To normalize with weights as specified by ``weight_type``.

  **NOTE: "MAX" is no longer available**

- ``weight_type``

  Format: string or ``None``. "calc" or "manual" (default: ``None``)

  Description: The weights of the datasets for the "MANY_BEAM" normalization.

  - "calc"

  .. math::

    w^{(n)} = \left(\frac{\sum_i^m v^{(n)}_{i}}{\sum_j^n \sum_i^m v^{(j)}_i} \right)^2

  - "manual"

  :math:`w^{(n)}` is specified by ``spot_weight``.

- ``spot_weight``

  Format: list of float (mandatory when ``weight_type`` is "manual")

  Description: The weights of the beams in the calculation of R-factor.
  The weights are automatically normalized as the sum be 1.
  For example, [3,2,1] means :math:`w^{(1)}=1/2, w^{(2)}=1/3, w^{(3)}=1/6`.

- ``omega``

  Format: float (default: 0.5)

  Description: This parameter specifies the half-width of convolution.

- ``remove_work_dir``

  Format: boolean (default: false)

  Description: Whether to remove working directories ``Log%%%_###`` after reading R-factor or not

[``solver.param``] subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``string_list``

  Format: list of string. The length should match the value of dimension (default: ["value_01", "value_02"]).

  Description: List of placeholders to be used in the reference template file to create the input file for the solver. These strings will be replaced with the values of the parameters being searched for.


[``solver.reference``] subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``path``

  Format: string (default: ``experiment.txt``)

  Description: Path to the reference data file.
  
- ``reference_first_line``

  Format: integer

  Description: In the reference data file, the first line to be read as Dexp. The default value is 1, that is, the first line of the file.

- ``reference_last_line``

  Format: integer

  Description: In the reference data file, the last line to be read as Dexp. If omitted, all lines from the first line to the end of the file will be read.

- ``exp_number``

  Format: Integer or List of integers

  Description: In the reference data file, the column numbers to be read. Multiple columns can be specified (many-beam condition).


Reference file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input template file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The input template file ``template.txt`` is a template for creating an input file for ``surf.exe``.
The parameters to be moved in ``py2dmat`` (such as the atomic coordinates you want to find) should be replaced with the appropriate string, such as ``value_*``.
The strings to be used are specified by ``string_list`` in the ``[solver]`` - ``[param]`` section of the input file for the solver.
An example template is shown below.

.. code-block::

    2                                    ,NELMS,  -------- Ge(001)-c4x2
    32,1.0,0.1                           ,Ge Z,da1,sap
    0.6,0.6,0.6                          ,BH(I),BK(I),BZ(I)
    32,1.0,0.1                           ,Ge Z,da1,sap
    0.4,0.4,0.4                          ,BH(I),BK(I),BZ(I)
    9,4,0,0,2, 2.0,-0.5,0.5               ,NSGS,msa,msb,nsa,nsb,dthick,DXS,DYS
    8                                    ,NATM
    1, 1.0, 1.34502591	1	value_01   ,IELM(I),ocr(I),X(I),Y(I),Z(I)
    1, 1.0, 0.752457792	1	value_02
    2, 1.0, 1.480003343	1.465005851	value_03
    2, 1.0, 2	1.497500418	2.281675
    2, 1.0, 1	1.5	1.991675
    2, 1.0, 0	1	0.847225
    2, 1.0, 2	1	0.807225
    2, 1.0, 1.009998328	1	0.597225
    1,1                                  ,(WDOM,I=1,NDOM)

In this case, ``value_01``, ``value_02``, and ``value_03`` are the parameters to be moved in ``py2dmat``.


Target file
^^^^^^^^^^^^^^
This file (``experiment.txt``) contains the data to be targeted.
The first column contains the angle, and the second and following columns contain the calculated value of the reflection intensity multiplied by the weight.
An example of the file is shown below.

.. code-block::

    3.00000e-01 8.17149e-03 1.03057e-05 8.88164e-15 ...
    4.00000e-01 1.13871e-02 4.01611e-05 2.23952e-13 ...
    5.00000e-01 1.44044e-02 1.29668e-04 4.53633e-12 ...
    6.00000e-01 1.68659e-02 3.49471e-04 7.38656e-11 ...
    7.00000e-01 1.85375e-02 7.93037e-04 9.67719e-10 ...
    8.00000e-01 1.93113e-02 1.52987e-03 1.02117e-08 ...
    9.00000e-01 1.92590e-02 2.53448e-03 8.69136e-08 ...
    1.00000e+00 1.86780e-02 3.64176e-03 5.97661e-07 ...
    1.10000e+00 1.80255e-02 4.57932e-03 3.32760e-06 ...
    1.20000e+00 1.77339e-02 5.07634e-03 1.50410e-05 ...
    1.30000e+00 1.80264e-02 4.99008e-03 5.53791e-05 ...
    ...


Output file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For ``sim-trhepd-rheed``, the files output by ``surf.exe`` will be output in the ``Log%%%%%_#####`` folder under the folder with the rank number.
``%%%%%`` means an index of iteration in ``Algorithm`` (e.g., steps in Monte Carlo),
and ``#####`` means an index of group (e.g., replica index in Monte Carlo).
In large calculation, the number of these folders becomes too large to be written in the storage of the system.
For such a case, let ``solver.post.remove_work_dir`` parameter be ``true`` in order to remove these folders.
This section describes the own files that are output by this solver.

``stdout``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It contains the standard output of ``surf.exe``.
An example is shown below.

.. code-block::

     bulk-filename (end=e) ? :
     bulkP.b
     structure-filename (end=e) ? :
     surf.txt
     output-filename :
     surf-bulkP.s

``RockingCurve_calculated.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This file is located in the ``Log%%%%%_#####`` folder.
At the beginning of the file, the lines beginning with ``#`` are headers.
The header contains the values of the input variables, the objective function value ``f(x)``, the parameters ``Rfactor_type``, ``normalization``, ``weight_type``, ``cal_number``, ``spot_weight``, and what is marked in the data portion columns (e.g. ``# #0 glancing_angle``).

The header is followed by the data.
The first column shows the glancing angle, and the second and subsequent columns show the intensity of each data column.
You can see which data columns are marked in the header. For example,

.. code-block::

  # #0 glancing_angle
  # #1 cal_number=1
  # #2 cal_number=2
  # #3 cal_number=4

shows that the first column is the glancing angle, and the second, third, and fourth columns are the calculated data corresponding to the first, second, and fourth columns of the calculated data file, respectively.

Intencities in each column are normalized so that the sum of the intensity is 1.
To calculate the objective function value (R-factor and R-factor squared), the data columns are weighted by ``spot_weight`` and normalized by ``normalization``.

.. code-block::

  #value_01 =  0.00000 value_02 =  0.00000 
  #Rfactor_type = A
  #normalization = MANY_BEAM
  #weight_type = manual
  #fx(x) = 0.03686180462340505
  #cal_number = [1, 2, 4, 6, 8]
  #spot_weight = [0.933 0.026 0.036 0.003 0.002]
  #NOTICE : Intensities are NOT multiplied by spot_weight.
  #The intensity I_(spot) for each spot is normalized as in the following equation.
  #sum( I_(spot) ) = 1
  #
  # #0 glancing_angle
  # #1 cal_number=1
  # #2 cal_number=2
  # #3 cal_number=4
  # #4 cal_number=6
  # #5 cal_number=8
  0.30000 1.278160358686800e-02 1.378767858296659e-04 8.396046839668212e-14 1.342648818357391e-30 6.697979700048016e-53
  0.40000 1.778953628930054e-02 5.281839702773564e-04 2.108814173486245e-12 2.467220122612354e-28 7.252675318478533e-50
  0.50000 2.247181148723425e-02 1.671115124520428e-03 4.250758278908295e-11 3.632860054842994e-26 6.291667506376419e-47
  ...

