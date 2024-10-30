.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

(For developers) User-defined algorithm and solver
====================================================

``py2dmat`` solves the reverse problem by combination of ``Solver`` for the direct problem and ``Algorithm`` for the optimization problem.
Instead of some ``Solver`` and ``Algorithm`` which are served by ``py2dmat``, users can define and use their own components.
In this chapter, how to define ``Solver`` and ``Algorithm`` and to use them will be described.

.. toctree::
   :maxdepth: 1

   common
   solver
   algorithm
   usage
