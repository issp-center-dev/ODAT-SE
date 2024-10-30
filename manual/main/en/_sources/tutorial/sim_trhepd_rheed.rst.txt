TRHEPD Direct Problem Solver
============================

As one of the forward problem solvers, 2DMAT provides a wrapper for the program `sim-trhepd-rheed <https://github.com/sim-trhepd-rheed/sim-trhepd-rheed/>`_ , which calculates the intensity of reflection fast (positron) electron diffraction (RHEED, TRHEPD) (A. Ichimiya, Jpn. J. Appl. Phys. 22, 176 (1983); 24, 1365 (1985)). 
In this tutorial, we will show some examples which use some algorithms with sim-trhepd-rheed.
First, we will install and test sim-trhepd-rheed (for details, see the official web page for `sim-trhepd-rheed <https://github.com/sim-trhepd-rheed/sim-trhepd-rheed/>`_).

Download and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, in the tutorial, we assume that you are at the location where the ``2DMAT`` folder is located. ::

   $ ls -d 2DMAT
   2DMAT/

Get the source codes from the sim-trhepd-rheed repository on GitHub and build it. ::

   git clone http://github.com/sim-trhepd-rheed/sim-trhepd-rheed
   cd sim-trhepd-rheed/src
   make

If make is successful, ``bulk.exe`` and ``surf.exe`` will be created.
		

Calculation execution
~~~~~~~~~~~~~~~~~~~~~

In sim-trhepd-rheed, the bulk part of the surface structure is first calculated with ``bulk.exe``.
Then, using the results of the ``bulk.exe`` calculation (the ``bulkP.b`` file), 
the surface portion of the ``surf.exe`` surface structure is calculated.

In this tutorial, we will actually try to do the TRHEPD calculation.
The sample input files are located in ``sample/sim-trhepd-rheed`` in 2DMAT.
First, copy this folder to a suitable working folder ``work``.

.. code-block::

   cd ../../
   cp -r 2DMAT/sample/sim-trhepd-rheed/solver work
   cd work

Next, copy ``bulk.exe`` and ``surf.exe`` to ``work``.

.. code-block::

   cp ../sim-trhepd-rheed/src/bulk.exe .
   cp ../sim-trhepd-rheed/src/surf.exe .

Execute ``bulk.exe``.

.. code-block::

   ./bulk.exe

Then, the bulk file ``bulkP.b`` will be generated with the following output.

.. code-block::

   0:electron 1:positron ?
   P
   input-filename (end=e) ? :
   bulk.txt
   output-filename :
   bulkP.b

Next, execute ``surf.exe``.

.. code-block::

   ./surf.exe

Then, the following standard output will be seen.

.. code-block::

   bulk-filename (end=e) ? :
   bulkP.b
   structure-filename (end=e) ? :
   surf.txt
   output-filename :
   surf-bulkP.md
   surf-bulkP.s

After execution, the files ``surf-bulkP.md``, ``surf-bulkP.s`` and ``SURFYYYYMMDD-HHMMSSlog.txt`` will be generated.
(YYYYMMDD and HHMMSS are numbers corresponding to the execution date and time).

Visualization of calculation result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contents of ``surf-bulkP.s`` are shown as follow:

.. code-block::

   #azimuths,g-angles,beams
   1 56 13
   #ih,ik
   6 0 5 0 4 0 3 0 2 0 1 0 0 0 -1 0 -2 0 -3 0 -4 0 -5 0 -6 0
   0.5000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.1595E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.6000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.1870E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.7000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.2121E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.8000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.2171E-02, 0.1927E-01, 0.2171E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.9000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.4397E-02, 0.1700E-01, 0.4397E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.1000E+01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.6326E-02, 0.1495E-01, 0.6326E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   ...

From the above file, create a rocking curve from the angle on the vertical axis (first column of data after row 5) and the intensity of the (0,0) peak (eighth column of data after row 5).
You can use Gnuplot or other graphing software, but here we use the program ``plot_bulkP.py`` in the ``2DMAT/script`` folder.
Run it as follows.

.. code-block::

   python3 ../2DMAT/script/plot_bulkP.py

The following ``plot_bulkP.png`` will be created.

.. figure:: ../../../common/img/plot_bulkP.*

   Rocking curve of Si(001)-2x1 surface.

We will convolute and normalize the diffraction intensity data of the 00 peaks.
Prepare ``surf-bulkP.s`` and run ``make_convolution.py``.

.. code-block::

   python3 ../2DMAT/script/make_convolution.py

When executed, the following file ``convolution.txt`` will be created.

.. code-block::

   0.500000 0.010818010
   0.600000 0.013986716
   0.700000 0.016119093
   0.800000 0.017039022
   0.900000 0.017084666
     ... skipped ...
   5.600000 0.000728539
   5.700000 0.000530758
   5.800000 0.000412908
   5.900000 0.000341740
   6.000000 0.000277553

The first column is the viewing angle, and the second column is the normalized 00-peak diffraction intensity data 
written in ``surf-bulkP.s`` with a convolution of half-width 0.5.
