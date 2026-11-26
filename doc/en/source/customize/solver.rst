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

    .. note::
      With ``--nsolve`` greater than 1, the current directory when ``evaluate`` or ``gradient`` is called
      is ``proc_dir`` on the controller but the directory where ``odatse`` was launched on the workers.
      A solver that reads or writes files should use ``self.work_dir`` explicitly.

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
    Error handling when called from ``Runner``:

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

    It takes the same arguments as the ``evaluate`` method.
    With ``function.Solver``, ``set_gradient()`` can be used instead of implementing this method (see :doc:`function_solver`).

    .. note::
      Error handling when called from ``Runner``:

      If ``gradient`` raises a ``RuntimeError`` and ``ignore_error = true`` is specified in the ``[runner]`` section, the Runner ignores the exception and treats the gradient as an array of ``np.nan``.
      If the search point does not satisfy the constraints (``[runner.limitation]``), the solver is not called and the gradient becomes an array of ``np.nan``.

    ``gradient`` must return the derivative of the value returned by ``evaluate`` with respect to the same ``x`` that ``evaluate`` receives.
    Its correctness is the responsibility of the solver's author. ODAT-SE checks only the shape of the returned array and whether the call failed,
    so a gradient with a wrong sign or scale shows up as a premature termination or as convergence to a wrong point, which cannot be told from the output.
    Use ``check_gradient`` below to compare it with finite differences beforehand.

  - ``check_gradient(self, x, args=(0, 0), step=None) -> tuple[np.ndarray, np.ndarray]``

    A method provided by ``SolverBase`` for checking (subclasses do not need to implement it).
    It returns the pair ``(g, d)``: ``g`` is the value of ``gradient`` at ``x``, and ``d`` is the gradient estimated by central differences of ``evaluate``.
    It does not judge whether they agree and prints nothing; the tolerance is up to the caller.

    - ``x``: one-dimensional sequence of finite real numbers, in the same space as the argument of ``evaluate``.
    - ``args``: passed unchanged to every call to ``evaluate`` and ``gradient``.
    - ``step``: step of the differences; positive and finite, added to and subtracted from the coordinates as it is (not relative to ``x``). A scalar or a sequence as long as ``x``.
      When omitted, ``eps**(1/3) * max(1, |x_i|)`` is used for each component (``eps`` is the double-precision machine epsilon).
      This is a guide for smooth functions; specify ``step`` when the evaluation is noisy or the scale of a variable is far from 1.
    - An invalid ``x`` or ``step`` raises ``ValueError`` before the solver is called.
      Exceptions raised by ``evaluate`` or ``gradient`` propagate unchanged.

    The calls are made in this order: ``evaluate(x)``, ``gradient(x)``, and then ``evaluate(x + h e_i)`` and ``evaluate(x - h e_i)`` for each component.
    ``evaluate`` is called 2 × dimension + 1 times and ``gradient`` once.
    ``g`` keeps the shape returned by ``gradient``, which is not checked; ``d`` has the shape of ``x``.
    A component for which the step is too small to make ``x + h`` and ``x - h`` differ is not evaluated, and its value in ``d`` is ``np.nan``.

    In a test of a solver it can be used as follows.

    .. code-block:: python

      import numpy as np

      def test_gradient(solver):
          solver.work_dir.mkdir(parents=True, exist_ok=True)
          x = np.array([0.3, -1.2])
          g, d = solver.check_gradient(x)
          assert g.shape == d.shape
          np.testing.assert_allclose(g, d, rtol=1e-6, atol=1e-8)

    .. note::
      - ``Runner`` is not involved, so the caller must prepare whatever ``evaluate`` needs, such as creating the working directory.
        The search region and the constraints are not taken into account either; choose a point and a step that stay inside them.
      - After the call, the internal state of the solver and the files in the working directory are those of the last evaluation for the differences.
      - ``np.testing.assert_allclose`` accepts a comparison between a scalar and an array, so check the shape separately.
      - Agreement with the differences is meaningful only where the objective function is differentiable and its evaluation is reproducible
        (the Ackley function, for example, is not differentiable at the origin). When in doubt about the step, compare several values of ``step``.
      - Only the value at the given point is checked. Checking at several points is recommended.
      - With the solver parallelization, all ranks of the group must make the call with the same arguments. Exceptions are not aggregated across the ranks.


  ``runner.log`` counts calls to ``evaluate()`` only; gradient calls are not recorded or counted.

  Gradients are not supported when ``[runner.mapping]`` is specified.
