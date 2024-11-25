.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorials
==================================

In these tutorials, how to perform inverse problem analyses using ODAT-SE is explained by examples taken from minimization of analytical functions.
In ODAT-SE, the algorithms for solving the inverse problem can be selected from the following algorithms:

- ``minsearch``

   Nealder-Mead method.

- ``mapper_mpi``

   Entire search over a grid for a given parameter.

- ``bayes``

   Bayesian optimization.

- ``exchange``

   Sampling by the replica exchange Monte Carlo method.

- ``pamc``

   Sampling by the population annealing Monte Carlo method.

In the following sections, the procedures to run these algorithms are provided.
In addition, the usage of ``[runner.limitation]`` to apply limitations to the search region will be described. In the end of the section, how to define a direct problem solver wil be explained by a simple example.

.. toctree::
   :maxdepth: 1

   intro
   minsearch
   mapper
   bayes
   exchange
   pamc
   limitation
   solver_simple
