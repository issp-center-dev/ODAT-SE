Developer Guide
====================

ODAT-SE is a framework that solves inverse problems by combining a forward problem solver ``Solver`` with a search algorithm ``Algorithm``.
In addition to the built-in ``Solver`` and ``Algorithm``, users can define and use their own.

.. code-block:: text

   ┌────────────┐     ┌────────┐     ┌────────────┐
   │ Algorithm  │────→│ Runner │────→│   Solver   │
   │ (search)   │←────│(bridge)│←────│(objective) │
   └────────────┘     └────────┘     └────────────┘
     proposes          coordinate      computes
     parameter x       transform       f(x) and
                       & constraints    returns it

This chapter first explains **how to add a custom solver** through a hands-on tutorial,
then provides API details.

:doc:`tutorial_solver`
    A tutorial for optimizing your own objective function with ODAT-SE. Provides a complete, copy-paste-ready example covering file creation, execution, and result verification. Start here if you are new to customization.

:doc:`solver`
    API reference for the ``Solver`` class. Details on inheriting ``SolverBase`` and implementing the ``evaluate`` method.

:doc:`algorithm`
    API reference for the ``Algorithm`` class. How to define custom search algorithms.

:doc:`common`
    Explanation of classes shared by Solver and Algorithm: ``Info``, ``Runner``, ``Mapping``, ``Limitation``.

:doc:`usage`
    Code example for combining custom Solver / Algorithm and running them.

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_solver
   solver
   algorithm
   common
   usage
