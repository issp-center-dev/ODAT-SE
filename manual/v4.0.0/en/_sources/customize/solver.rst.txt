``Solver``
================================

``Solver`` is a class that describes the direct problem, providing a method ``evaluate`` that returns the value of the objective function from the input parameters.

- ``Solver`` is defined as a derived class of ``odatse.solver.SolverBase``.

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
  The following instance variables are set by the constructor of the base class:

  - ``self.root_dir: pathlib.Path`` : Root directory

    This parameter is taken from ``info.base["root_dir"]``, and represents the directory in which ``odatse`` is executed. It can be used as the base location when external programs or data files are read.

  - ``self.output_dir: pathlib.Path`` : Output directory

    This parameter is taken from ``info.base["output_dir"]``, and used for the directory in which the result files are written. Usually, when the MPI parallelization is applied, the accumulated results are stored.

  - ``self.proc_dir: pathlib.Path`` : Working directory for each MPI process by the form ``self.output_dir / str(odatse.mpi.algrank())``

    The ``evaluate`` method of Solver is called from Runner with the ``proc_dir`` directory set as the current directory, in which the intermediate results produced by each rank are stored. When the MPI parallelization is not used, the rank number is treated as 0.

  - ``self.work_dir: pathlib.Path`` : An alias of ``self.proc_dir``.

  - ``self.dimension: int`` : The dimension of the input parameter. It is taken from ``info.solver["dimension"]`` if specified, or from ``info.base["dimension"]`` otherwise.

  - ``self.timer: dict`` : A dictionary for recording execution times, with the keys ``"prepare"``, ``"run"``, and ``"post"``.

  - ``self._name: str`` : The name of the solver. It is initialized to an empty string in the base class; set an appropriate name in the constructor. It is referred to through the ``name`` property.

  The parameters for the Solver class can be obtained from the ``solver`` field of the ``info`` object.
  The required parameters should be taken and stored.

- ``evaluate`` method

  The form of ``evaluate`` method should be as follows:

  .. code-block:: python

     def evaluate(self, x, args=()) -> float:
         pass

  This method evaluates the objective function at a given parameter value ``x`` and returns the result. It takes the following arguments:

  - ``x: np.ndarray``

    The parameter value as an :math:`N`-dimensional vector of type ``numpy.ndarray``.

  - ``args: Tuple = ()``

    The additional arguments passed from the Algorithm in the form of a Tuple of two integers.
    One is the step count that corresponds to the Monte Carlo steps for MC type algorithms, or the index of the grid point for the grid search algorithm.
    The other is the set number that represents the :math:`n`-th iteration.

  The ``evaluate`` method returns the value of the objective function as a float.

  .. note::
     If ``evaluate`` raises a ``RuntimeError`` and ``ignore_error = true`` is specified in the ``[runner]`` section, the Runner ignores the exception and treats the objective function value as ``np.nan``.
     If the search point does not satisfy the constraints (``[runner.limitation]``), the solver is not called and the objective function value becomes ``np.inf``.

  .. note::
     When the solver parallelization is used (``--nsolve`` greater than 1), ``evaluate`` is called on all MPI ranks in the solver group with the same ``x`` and ``args``.
     The division of roles among the ranks should be implemented within the solver. See :doc:`../tutorial/parallel_solver` for details.
