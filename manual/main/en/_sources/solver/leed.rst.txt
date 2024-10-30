``leed`` solver
***********************************************

``leed`` is a ``Solver`` made by M.A. Van Hove, which calculates the Rocking curve from atomic positions etc., using ``SATLEED``, and returns the error from the experimental Rocking curve as :math:`f(x)`.
For more information on ``SATLEED``, see [SATLEED]_.

.. [SATLEED] M.A. Van Hove, W. Moritz, H. Over, P.J. Rous, A. Wander, A. Barbieri, N. Materer, U. Starke, G.A. Somorjai, Automated determination of complex surface structures by LEED, Surface Science Reports, Volume 19, 191-229 (1993). https://doi.org/10.1016/0167-5729(93)90011-D

Preparation
~~~~~~~~~~~~
First, install  ``SATLEED`` .
Access to the following URL
http://www.icts.hkbu.edu.hk/VanHove_files/leed/leedsatl.zip
and download a zip file.
Depending on the details of the system you want to calculate, it is necessary to change the parameters in the source code for ``SATLEED``.
After changing parameters, compile programs to generate the executable files such as ``stal1.exe``, ``satl2.exe`` .

For trying the example at ``sample/py2dmat/leed``, a utility script file ``setup.sh`` for downloading ``SATLEED``, rewriting source codes, and compiling the program is available.::

    $ cd sample/py2dmat/leed
    $ sh ./setup.sh

After running ``setup.sh``, executable files ``satl1.exe`` and ``satl2.exe`` are generated in ``leedsatl`` directory.

Note that it is assumed that you have already executed ``satl1.exe`` before using ``py2dmat`` .
Therefore, the following files must be generated.

- Input files of ``satl1.exe`` : ``exp.d``, ``rfac.d``, ``tleed4.i``, ``tleed5.i``

- Output files of ``satl1.exe`` : ``tleed.o`` , ``short.t``

``py2dmat`` will run ``satl2.exe`` based on the above files.

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input parameters are specified in subsections in the ``solver`` section (``config`` and ``reference``).

[``config``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``path_to_solver``

  Format: string

  Description: Path to the solver ``satl2.exe`` .


[``reference``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``path_to_base_dir``

  Format: string

  Description: Path to the directory which stores ``exp.d``, ``rfac.d``, ``tleed4.i``, ``tleed5.i`` , ``tleed.o`` , ``short.t`` .

  
Reference file for Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Target reference file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file containing the data to be targeted to fit. Edit ``tleed4.i`` in ``path_to_base_dir`` in the [``reference``] section.
Add the number you want to optimize to ``optxxx`` (where xxx is a three-digit number in the format 000, 001, 002, ...). (where xxx is a three-digit integer in the form 000, 001, 002, ...).
Note that the number of xxx must match the order and number of variables in the list of ``py2dmat`` variables to be optimized.
Note that if IFLAG and LSFLAG are not set to 0, the satleed side is also optimized.

An example file is shown below.

.. code-block::

    1  0  0                          IPR ISTART LRFLAG
    1 10  0.02  0.2                  NSYM  NSYMS ASTEP VSTEP
    5  1  2  2                       NT0  NSET LSMAX LLCUT
    5                                NINSET
    1.0000 0.0000                  1      PQEX
    1.0000 2.0000                  2      PQEX
    1.0000 1.0000                  3      PQEX
    2.0000 2.0000                  4      PQEX
    2.0000 0.0000                  5      PQEX
    3                                 NDIM
    opt000 0.0000 0.0000  0           DISP(1,j)  j=1,3
    0.0000 opt001 0.0000  0           DISP(2,j)  j=1,3
    0.0000 0.0000 0.0000  1           DISP(3,j)  j=1,3
    0.0000 0.0000 0.0000  0           DISP(4,j)  j=1,3
    0.0000  0                         DVOPT  LSFLAG
    3  0  0                           MFLAG NGRID NIV
    ...
   
Output file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``leed``, the output files are stored in the folder with the rank number.
