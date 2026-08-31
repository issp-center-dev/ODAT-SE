========================================
Related Resources
========================================

ODAT-SE Gallery
~~~~~~~~~~~~~~~~~~~~~~~~~

`ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ is a repository of sample data, working examples, and solver templates for various analysis techniques using ODAT-SE.

Analysis Examples
^^^^^^^^^^^^^^^^^^^^^^^^

Working examples for the following quantum beam diffraction experiments are included:

- **odatse-STR** -- Structure analysis by Total-reflection High-energy Positron Diffraction (TRHEPD)
- **odatse-SXRD** -- Surface X-ray Diffraction (SXRD) analysis
- **odatse-LEED** -- Low-energy Electron Diffraction (LEED) analysis
- **odatse-XAFS** -- Extended X-ray Absorption Fine Structure (XAFS) analysis

Each sample includes mesh data, input files (``input.toml``), execution scripts (``do.sh``), reference data, and visualization scripts.
They can be run as-is to reproduce results.

Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^

Tutorials illustrating custom solvers, Bayesian inference workflows, and optimization with the TTOpt (tensor train optimization) algorithm are also included:

- **gpr** -- Gaussian process regression: modeling an unknown function and estimating its minimum with a custom solver (``minsearch``)
- **linreg_with_noise** -- Linear regression and noise-level estimation via model evidence (``pamc``)
- **ttopt_benchmark** -- TTOpt with continuous parameters: minimizing analytical benchmark functions
- **qubo** -- TTOpt with discrete parameters: QUBO (Quadratic Unconstrained Binary Optimization) problems
- **mip** -- TTOpt with continuous and discrete parameters: mixed integer programs (MIP)

Each example provides a main script, ODAT-SE input files, sample data, and step-by-step tutorials.

Solver Templates
^^^^^^^^^^^^^^^^^^^^^^^^

Four types of templates are available for developing custom solvers.
See :doc:`customize/tutorial_solver` for details.

- **user_function** -- Minimal script for quickly optimizing a Python function
- **function_module** -- ``pip install``-able package template for analytical function solvers
- **solver_module** -- Template for solvers that compare against reference data
- **external_solver_module** -- Template for using external programs (C/Fortran, etc.) as solvers

How to Obtain
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ git clone https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery.git


Related Links
~~~~~~~~~~~~~~~~~~~~~~~~~

- `ODAT-SE GitHub Repository <https://github.com/issp-center-dev/ODAT-SE>`_
- `Project for Advancement of Software Usability in Materials Science <https://www.pasums.issp.u-tokyo.ac.jp/>`_
- `ODAT-SE Publications and Presentations <https://www.pasums.issp.u-tokyo.ac.jp/odat-se/paper>`_
