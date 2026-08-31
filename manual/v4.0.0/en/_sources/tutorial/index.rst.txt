.. 2dmat documentation master file, created by
  sphinx-quickstart on Tue May 26 18:44:52 2020.
  You can adapt this file completely to your liking, but it should at least
  contain the root `toctree` directive.

Tutorials
==================================

In these tutorials, how to perform inverse problem analyses using ODAT-SE is explained by examples taken from minimization of analytical functions.
The following algorithms are covered in these tutorials.
``global_search`` (global optimization with scipy.optimize) is also available;
see :doc:`../algorithm/index` for the complete list of the algorithms.

- ``minsearch``

  Nelder-Mead method.

- ``mapper``

  Exhaustive search over a grid of the given parameters.

- ``random_search``

  Random search.

- ``bayes``

  Bayesian optimization.

- ``ttopt``

  Tensor train optimization.

- ``exchange``

  Sampling by the replica exchange Monte Carlo method.

- ``pamc``

  Sampling by the population annealing Monte Carlo method.

In the following sections, the procedures to run these algorithms are provided.
In addition, the usage of ``[runner.limitation]`` to apply limitations to the search region will be described. At the end of the section, a description of how to implement a direct problem solver is provided, as well as tutorials describing some applications.

Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the tutorials, the sample files are required in addition to the installation of ODAT-SE (see :doc:`../start`).
The sample files are included in the source package.
Clone the repository and move into the obtained directory as follows:

.. code-block::

    $ git clone https://github.com/issp-center-dev/ODAT-SE.git
    $ cd ODAT-SE

The steps in each tutorial are assumed to start from this ODAT-SE directory.

.. note::
   To run the tutorials that use MPI parallelization (``mapper``, ``exchange``, ``pamc``, and so on),
   ``mpi4py`` (``python3 -m pip install 'ODAT-SE[all]'``) is required in addition to a working
   MPI implementation providing the ``mpiexec`` / ``mpirun`` command.

.. toctree::
   :maxdepth: 1

   intro
   minsearch
   mapper
   random_search
   bayes
   ttopt
   exchange
   pamc
   limitation
   solver_simple
   parallel_solver
