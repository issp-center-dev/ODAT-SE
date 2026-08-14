odatse command
================

NAME
----
odatse - perform inverse problem analyses combining a search algorithm and a direct problem solver

SYNOPSIS
--------

.. code-block:: bash

   odatse [-h] [--version] [--init | --resume | --cont] [--reset_rand]
          [--nalg NALG] [--nsolve NSOLVE] inputfile

DESCRIPTION
-----------

Reads an input file in TOML format and performs an inverse problem analysis combining the search algorithm specified in the ``[algorithm]`` section with the direct problem solver specified in the ``[solver]`` section.
See :doc:`/input/index` for the specification of the input file and :doc:`/output` for the output files.

For MPI parallel execution, launch the command via ``mpiexec``:

.. code-block:: bash

   mpiexec -np N odatse [OPTION]... inputfile

The available command-line options are listed below.

**inputfile**
    Input file in TOML format.

**--init**
    Start the calculation from the initial state. This is the default behavior.

**--resume**
    Restore the state at the interruption point from the checkpoint file and resume the calculation.
    Applies to runs executed with the checkpoint feature enabled (``checkpoint = true``) in the ``[algorithm]`` section.

**--cont**
    Take over the results of a finished calculation and continue from where it ended.
    Use this to extend a calculation with more steps or more temperature points.
    As with ``--resume``, the preceding calculation must have been run with the checkpoint
    feature enabled (``checkpoint = true``) so that a checkpoint file of the final state exists.

    .. note::
       ``--init`` / ``--resume`` / ``--cont`` are mutually exclusive.
       The support for the execution modes varies by algorithm; see the "Restart" or "Limitation" section of each algorithm page (:doc:`/algorithm/index`).

**--reset_rand**
    Use together with ``--resume`` or ``--cont`` to start a new random number series on restart.

**--nalg NALG**
    Number of MPI processes assigned to the search algorithm layer.
    Used together with ``--nsolve`` to split the MPI communicator; ``NALG × NSOLVE`` must equal the total number of processes.
    If omitted, it is determined from the total number of processes and ``--nsolve``.

**--nsolve NSOLVE**
    Number of MPI processes per solver group.
    If both ``--nalg`` and ``--nsolve`` are omitted, all processes are assigned to the algorithm layer (``NSOLVE = 1``).
    See :doc:`/tutorial/parallel_solver` for details of the two-level parallelization.

**--version**
    Show the version and exit.

**-h, --help**
    Show the help message and exit.

USAGE
-----

1. Run from the initial state

   .. code-block:: bash

      odatse input.toml

2. Run in parallel with MPI (4 processes)

   .. code-block:: bash

      mpiexec -np 4 odatse input.toml

3. Resume an interrupted calculation from the checkpoint

   .. code-block:: bash

      odatse --resume input.toml

4. Extend a finished calculation with a new random number series

   .. code-block:: bash

      odatse --cont --reset_rand input.toml

   Examples 3 and 4 assume that the preceding calculation was run with
   ``checkpoint = true`` and that its checkpoint file is still available.

5. Split 8 processes into 2 algorithm processes × 4 solver processes

   .. code-block:: bash

      mpiexec -np 8 odatse --nalg 2 --nsolve 4 input.toml

ENVIRONMENT
-----------

**ODATSE_NOMPI**
    When set to a value other than ``0``, run with MPI disabled without loading mpi4py.
    Use this to run without MPI (without importing mpi4py) in an environment where mpi4py is installed.

.. code-block:: bash

   ODATSE_NOMPI=1 odatse input.toml

SEE ALSO
--------

- :doc:`/input/index` -- Specification of the input file
- :doc:`/output` -- Specification of the output files
- :doc:`/algorithm/index` -- Search algorithms and their support for the execution modes
- :doc:`/tutorial/parallel_solver` -- Two-level MPI parallelization
