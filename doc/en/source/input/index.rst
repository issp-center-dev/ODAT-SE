================================
Input file
================================

As the input file format, `TOML <https://toml.io/ja/>`_ format is used.
The input file consists of the following four sections.

- ``base``

  - Specifies the basic settings for ODAT-SE, such as the dimension of the search space and directory configuration.

- ``solver``

  - Specifies the type and settings of the forward problem solver (the module that computes the objective function).

- ``algorithm``

  - Specifies the type and settings of the search algorithm (e.g., Nelder-Mead, Bayesian optimization, replica exchange Monte Carlo). This section also includes random seed and checkpoint settings.

- ``runner``

  - Configures the ``Runner`` that bridges the algorithm and solver. Includes settings for parameter coordinate transformations (affine mapping), search space constraints, and logging.


.. toctree::
   :maxdepth: 2

   algorithm
   runner

``[base]`` section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``dimension``

  Format: Integer

  Description: Dimension of the search space (number of parameters to search)

- ``root_dir``

  Format: string (default: The directory where the program was executed)

  Description: Name of the root directory. The origin of the relative paths to input files.

- ``output_dir``

  Format: string (default: The directory where the program was executed)

  Description: Name of the directory to output the results.

``[solver]`` section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``name`` determines the type of solver. Each parameter is defined for each solver.

- ``name``

  Format: String

  Description: Name of the solver. The following solvers are available.

    - ``analytical`` : Solver to provide analytical solutions (mainly used for testing).

    The following are solvers for 2D material structure analysis distributed as separate modules:

    - ``sim-trhepd-rheed`` :
      Solver to calculate Total-reflection high energy positron diffraction (TRHEPD) or Reflection High Energy Electron Diffraction (RHEED) intensities.

    - ``sxrd`` : Solver for Surface X-ray Diffraction (SXRD)

    - ``leed`` : Solver for Low-energy Electron Diffraction (LEED)

- ``dimension``

  Format: Integer (default: ``base.dimension``)

  Description:
  Number of input parameters for Solvers

See :doc:`/solver/index` for details of the various solvers and their input/output files.
