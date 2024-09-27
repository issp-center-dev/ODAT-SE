``Solver``
================================

``Solver`` is a class that describes the direct problem, providing a method ``evaluate`` that returns the value of the objective function from the input parameters.

- ``Solver`` is define as a derived class of ``odatse.solver.SolverBase``.

  .. code-block:: python

     import odatse

     class Solver(odatse.solver.SolverBase):
         pass

- Constructor

  Solver class should have a constructor that takes an ``Info`` class object as an argument:

  .. code-block:: python

     def __init__(self, info: odatse.Info):
         super().__init__(info)

  It is required to call the constructor of the base class with the info object.
  There the following instance variables are introduced:

  - ``self.root_dir: pathlib.Path`` : Root directory

    This parameter is taken from ``info.base["root_dir"]``, and represents the directory in which ``odatse`` is executed. It can be referred as a root location when the external programs or data files are read.

  - ``self.output_dir: pathlib.Path`` : Output directory

    This parameter is taken from ``info.base["output_dir"]``, and used for the directory in which the result files are written. Usually, when the MPI parallelization is applied, the accumulated results are stored.

  - ``self.proc_dir: pathlib.Path`` : Working directory for each MPI process by the form ``self.output_dir / str(mpirank)``

    The ``evaluate`` method of Solver is called from Runner with the ``proc_dir`` directory set as the current directory, in which the intermediate results produced by each rank are stored. When the MPI parallelization is not used, the rank number is treated as 0.

  The parameters for the Solver class can be obtained from ``solver`` field of ``info`` object.
  The required parameters should be taken and stored.

- ``evaluate`` method

  The form of ``evaluate`` method should be as follows:

  .. code-block:: python

     def evaluate(self, x, args=(), nprocs=1, nthreads=1) -> float:
         pass

  This method evaluates the objective function at a given parameter value `x` and returns the result. It takes the following arguments:

  - ``x: np.ndarray``

    The parameter value in :math:`N` dimensional vector of numpy.ndarray type.

  - ``args: Tuple = ()``

    The additional arguments passed from the Algorithm in the form of a Tuple of two integers.
    One is the step count that corresponds to the Monte Carlo steps for MC type algorithms, or the index of the grid point for grid search algorithm.
    The other is the set number that represents :math:`n`-th iteration.

  - ``nprocs: int = 1``

  - ``nthreads: int = 1``

    The number of processes and threads that specify how to run the solver in MPI/thread parallelisation. In the current version, ``nprocs=1`` and ``nthreads=1`` are accepted.
  
  The ``evaluate`` method returns the value of the objective function as a float number.
