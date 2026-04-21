.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Search algorithms
====================

ODAT-SE searches the parameter space :math:`\mathbf{X}\ni x` by using the search algorithm ``Algorithm`` and the result of ``Solver`` :math:`f(x)`.
The following search algorithms are available in ODAT-SE.
Click each item for detailed usage, including input parameters and output files.

:doc:`minsearch`
    Performs optimization using the Nelder-Mead method (simplex method). A derivative-free direct search method that converges quickly for a small number of parameters. Requires scipy.

:doc:`mapper_mpi`
    Divides the parameter space into a grid and evaluates :math:`f(x)` at all grid points. Supports MPI parallelization and is suitable for obtaining an overview of the parameter space.

:doc:`exchange`
    Searches using the replica exchange Monte Carlo method (parallel tempering). By exchanging configurations between replicas at different temperatures, it avoids being trapped in local minima. Requires mpi4py.

:doc:`pamc`
    Searches using the population annealing Monte Carlo method. Efficiently explores the parameter space by gradually cooling a large number of replicas (walkers) while resampling.

:doc:`bayes`
    Searches using Bayesian optimization. Builds a surrogate model of :math:`f(x)` using Gaussian process regression and selects the next evaluation point based on an acquisition function. Can efficiently find optimal solutions with a small number of evaluations. Requires physbo.

:doc:`random_search`
    Evaluates the objective function at randomly selected parameters. Samples parameters uniformly at random from a specified range, and is suitable for obtaining a global overview of the parameter space. Supports MPI parallelization.

.. toctree::
   :maxdepth: 1
   :hidden:

   minsearch
   mapper_mpi
   random_search
   exchange
   pamc
   bayes
   random_search
