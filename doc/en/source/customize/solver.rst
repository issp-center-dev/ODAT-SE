.. _customize_solver:

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
     An exception raised by ``evaluate`` on any rank of the solver group is reported to the controller rank (``solrank() == 0``) and handled there: ``ignore_error`` applies when every failing rank raised a ``RuntimeError``; otherwise the job terminates with an error. Collectives inside ``evaluate`` must stay matched across the group even when a rank fails.

- Optional gradient support

  If the gradient of the objective function can be computed, the following members can be implemented to allow the ``Algorithm`` to use the gradient (e.g. ``use_gradient = true`` in ``minsearch``).

  - ``supports_gradient`` (class attribute, default: False)

    Set this to True when the solver can compute gradients. Subclasses may
    declare support as a class attribute (``supports_gradient = True``), an
    instance attribute (``self.supports_gradient = True`` in ``__init__``), or
    a property returning True.

  - ``gradient(self, x, args=()) -> np.ndarray``

    Returns the gradient of the objective function at ``x`` with the same shape as ``x``.
    The Runner converts the return value to a float ndarray and raises ``ValueError``
    if its shape differs from that of ``x``, even with ``ignore_error = true``.
    Called only when ``supports_gradient`` is True. The base class raises ``NotImplementedError``.
    If ``supports_gradient`` is True, subclasses must override this method.
    Inheriting the default ``SolverBase.gradient`` implementation makes
    gradients unavailable; the Runner detects this before calling the solver.
    After checking the shape, the Runner returns non-finite values (NaN or
    ±inf) as returned by the solver, regardless of ``ignore_error``.

    Gradient failures follow the same rules as objective-function failures.
    If ``gradient()`` raises ``RuntimeError`` (including ``NotImplementedError``)
    and ``ignore_error = true`` is specified in ``[runner]``, the Runner returns
    an all-NaN vector with the same shape as ``x``. Otherwise, the exception
    propagates. Other exceptions, such as ``ValueError`` and ``TypeError``,
    always propagate.

    The optimization method determines the outcome after a non-finite gradient;
    it may report ``success=False`` or return NaN results. If a gradient contains
    ±inf, L-BFGS-B may report ``success=True`` at the initial point. Raise an
    exception where the gradient cannot be computed, rather than returning
    non-finite values.

    When L-BFGS-B terminates abnormally (``ABNORMAL``, ``success=False``), scipy's
    returned objective value may be from the last evaluated point and may not
    match the value at the returned parameters. ODAT-SE uses scipy's result as
    returned. Do not take the result at face value when ``success=False``.
    See :doc:`../algorithm/minsearch` for result handling.

  At a point violating the Runner's constraints, ``submit_gradient()`` returns
  an all-NaN vector without calling the solver. In ``minsearch``, points outside
  the search region or violating algorithm-unit constraints also produce NaN
  gradients without a solver call. The Runner itself does not check search bounds.

  ``runner.log`` counts calls to ``evaluate()`` only; gradient calls are not recorded or counted.

  Gradients are not supported when ``[runner.mapping]`` is specified or when the solver parallelization is used (``--nsolve`` greater than 1).
