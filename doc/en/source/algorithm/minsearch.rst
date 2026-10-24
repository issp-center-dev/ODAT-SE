============================================================
Optimization by local optimization algorithms ``minsearch``
============================================================

.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _scipy.optimize.basinhopping: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

``minsearch`` performs optimization by local optimization algorithms.
It is implemented on top of SciPy's `scipy.optimize.minimize`_ function.
The optimization method is selected by the ``method`` parameter of the
``[algorithm.minimize]`` section. The default is the
`Nelder-Mead method <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_
(a.k.a. downhill simplex method), and the other methods accepted by
`scipy.optimize.minimize`_ (Powell, COBYLA, ...) can also be chosen.
In the Nelder-Mead method, assuming the dimension of the parameter space is
:math:`D`, the optimal solution is searched by systematically moving a set of
:math:`D+1` coordinate points (the vertices of a simplex) according to the
value of the objective function at each point.

An important hyperparameter is the initial value of the coordinates.
Since these local optimization methods have the problem of being trapped in a
local optimum, it is recommended to repeat the calculation with different
initial values several times to check the results, or to enable the optional
global optimization by basin hopping (`scipy.optimize.basinhopping`_), which
repeats a random hop followed by a local optimization with the selected method
(see the ``basinhopping`` parameter below).


Preparation
~~~~~~~~~~~

You will need to install `scipy <https://docs.scipy.org/doc/scipy/reference>`_.

.. code-block::

   $ python3 -m pip install scipy

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It has subsections ``param`` and ``minimize``.

.. _minsearch_input_param:

``[algorithm.param]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``initial_list``

  Format: List of float. The length should match the value of dimension.

  Description: Initial value of the parameter. If not defined, it will be initialized uniformly and randomly.

- ``unit_list``

  Format: List of float. The length should match the value of dimension.

  Description:
  Units for each parameter.
  In the search algorithm, each parameter is divided by each of these values to perform simple nondimensionalization and normalization.
  If not defined, the value is 1.0 for all dimensions.

- ``min_list``

  Format: List of float. Length should be equal to ``dimension``.

  Description:
  Minimum value of each parameter.
  When a parameter falls below this value during the optimization,
  the solver is not evaluated and the value is considered infinite.

- ``max_list``

  Format: List of float. Length should be equal to ``dimension``.

  Description:
  Maximum value of each parameter.
  When a parameter exceeds this value during the optimization,
  the solver is not evaluated and the value is considered infinite.

``[algorithm.minimize]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the optimization method and its hyperparameters.
See the documentation of `scipy.optimize.minimize`_ for details.

All parameters other than the ODAT-SE-specific keys ``method``,
``initial_scale_list``, ``use_gradient``, and ``basinhopping`` (described below) are passed
verbatim to the ``options`` argument of `scipy.optimize.minimize`_.
If a parameter name not accepted by the selected method is given,
the program stops with an error before the optimization starts.
The default values of ``xatol``, ``fatol``, ``maxiter``, and ``maxfev``
listed below apply only when ``method`` is "Nelder-Mead".

Legacy callbacks reuse objective values at points evaluated since the previous
callback and at the previous callback's iterate; other points are evaluated with
``iset = 1``. Cached values are discarded at each callback, so memory use does not
grow with the number of evaluations. The callback form depends on the scipy
version and method (all methods use the legacy form with scipy < 1.11;
with scipy 1.18, TNC is one example).
For solvers that separate working directories by ``(step, iset)``, per-iteration
``iset = 1`` output is usually not created, because the iterate is normally
evaluated within the same iteration. It can still appear when a callback reports
a point from an earlier iteration, e.g. COBYLA in older scipy ending at its best point.

After optimization, scipy's ``success`` and ``message`` are printed on one line,
with a warning when ``success`` is false. For basin hopping, the available
status information is printed.

Each walker's result uses scipy's returned parameters and objective value as they
are. When multiple walkers finish with a mixture of NaN and finite objective
values, the final result selects the walker with the smallest finite value.

Parameters containing NaN, which an optimizer may propose after a failure, are
handled without calling the solver: the objective value is NaN, a warning is
printed, no step number is consumed, and no row is added to
``History_FunctionCall.txt``. Parameters containing ±inf but no NaN remain out of
range, with an objective value of inf. These rules apply with or without
``use_gradient``.

- ``method``

  Format: String (default: "Nelder-Mead")

  Description:
  Name of the optimization method, passed as-is to the ``method`` argument of `scipy.optimize.minimize`_.
  Examples: "Nelder-Mead", "Powell", "COBYLA".
  Note that for gradient-based methods (BFGS, CG, ...) the gradient is evaluated by numerical differentiation,
  which costs dimension+1 solver evaluations per gradient evaluation,
  unless the gradient provided by the solver is used (see ``use_gradient``).
  The search region (``min_list`` / ``max_list``) is passed as the ``bounds`` argument of scipy
  for methods that support it (Powell, L-BFGS-B, TNC, SLSQP, trust-constr, COBYLA, COBYQA).
  For the Nelder-Mead method, out-of-range points are handled as before
  by treating the objective function value as infinity.
  If the optimum may lie on the boundary, use a bounds-aware method such as
  L-BFGS-B, TNC, SLSQP, or trust-constr.
  Methods without bounds, such as BFGS and CG, treat out-of-range objective values
  as infinity and may stop before reaching the boundary.

- ``initial_scale_list``

  Format: List of float. The length should match the value of dimension.

  Description:
  The difference value that is shifted from the initial value in order to create the initial simplex for the Nelder-Mead method.
  The ``initial_simplex`` consists of ``initial_list`` together with the ``dimension`` points obtained by adding each single component of ``initial_scale_list`` to ``initial_list``.
  If not defined, scales at each dimension are set to 0.25.
  Used only when ``method`` is "Nelder-Mead".

- ``use_gradient``

  Format: Boolean (default: false)

  Description:
  When true, the gradient computed by the solver is passed to
  `scipy.optimize.minimize`_ as its ``jac`` argument instead of numerical differentiation
  (also for the local optimizations of ``basinhopping``).
  To use gradients with a custom solver, ``supports_gradient`` must be True and the
  ``gradient`` method must be implemented (see :ref:`Solver <customize_solver>`).
  With ``function.Solver``, ``set_gradient()`` can be used instead.
  A ``SolverBase`` subclass that declares support but inherits the default
  ``gradient()`` implementation, or a ``function.Solver`` (including its
  subclasses such as ``analytical``) that neither overrides ``gradient()``
  nor has a gradient set, cannot use
  gradients and is rejected before any solver evaluation.

  A non-boolean value (including strings such as ``"false"`` and integers such as ``0``)
  raises ``InputValueError`` regardless of whether the value evaluates to true or false.
  When ``use_gradient = true``, the following settings also raise ``InputValueError``
  before any solver evaluation:

  - A method that does not use gradients: Nelder-Mead, Powell, COBYLA, or COBYQA
    (case-insensitive; omitting ``method`` selects Nelder-Mead).
  - A method requiring a Hessian: dogleg, trust-ncg, trust-krylov, or trust-exact
    (case-insensitive). ODAT-SE does not supply a Hessian; Newton-CG is allowed.
  - A solver that does not support gradients (``supports_gradient`` is False).
  - A ``[runner.mapping]`` section.

  ``InputValueError`` inherits from both ``InputError`` and ``ValueError``.
  The ``odatse`` command reports these input errors on an ``ERROR:`` line without
  a traceback; under ``mpirun``, each rank reports one line.
  When false or omitted, the solver's gradient is not used and these restrictions do not apply.

  At points outside the search region or violating constraints, the solver's
  ``gradient()`` is not called and the gradient is a vector of NaN values.
  Gradient failures follow the same rules as objective-function failures.
  With ``ignore_error = true`` in ``[runner]``, a ``RuntimeError`` raised by
  ``gradient()`` (including ``NotImplementedError``) is treated as an all-NaN
  gradient. Otherwise, the exception propagates. Exceptions other than
  ``RuntimeError`` always propagate. Non-finite return values (NaN or ±inf)
  are used as returned, regardless of ``ignore_error``.

  The outcome depends on the optimization method: it may report
  ``success=False`` or return a NaN objective value or non-finite parameters
  (see the handling of results and NaN parameters above).

  If the solver returns a gradient containing ±inf, L-BFGS-B may report
  ``success=True`` while remaining at the initial point. To make failures
  detectable, raise an exception where the gradient cannot be computed,
  rather than returning non-finite values.

  When L-BFGS-B terminates abnormally (``ABNORMAL``, ``success=False``), scipy's
  returned objective value may be from the last evaluated point and may not
  match the value at the returned parameters. ODAT-SE uses scipy's result as
  returned. Do not take the result at face value when ``success=False``.

  Gradients and gradient failures are remembered only for points computed since
  the previous callback and for the previous callback's iterate; points from earlier
  iterations (e.g. revisited by basinhopping) are computed again.
  Callbacks reuse only finite remembered gradients. At a remembered point whose
  gradient contains non-finite values and has not recovered, they record NaN
  without calling ``gradient()`` again.
  Scipy's gradient requests may retry the point; after recovery, callbacks record
  the finite cached value.
  Gradient evaluations receive step numbers shared with objective evaluations,
  but their step numbers are not recorded in output files; see the histories below.

  Example:

  .. code-block:: toml

      [algorithm.minimize]
      method = "L-BFGS-B"
      use_gradient = true

- ``xatol``

  Format: Float (default: 1e-4)

  Description: Parameter used to determine convergence of the Nelder-Mead method.

- ``fatol``

  Format: Float (default: 1e-4)

  Description: Parameter used to determine convergence of the Nelder-Mead method.

- ``maxiter``

  Format: Integer (default: 10000)

  Description: Maximum number of iterations for the Nelder-Mead method.

- ``maxfev``

  Format: Integer (default: 100000)

  Description: Maximum number of times to evaluate the objective function.

- ``basinhopping``

  Format: Boolean or table (default: false)

  Description:
  Enables global optimization by `scipy.optimize.basinhopping`_ (basin hopping),
  with the method specified by ``method`` used as the local minimizer of each hop.
  ``basinhopping = true`` runs with the scipy default parameters.
  Defining a ``[algorithm.minimize.basinhopping]`` sub-table also enables it;
  its entries (``niter``, ``stepsize``, ``T``, ...) are passed verbatim as
  arguments of `scipy.optimize.basinhopping`_.
  If an argument name not accepted by scipy is given, the program stops with
  an error before the optimization starts.
  Arguments managed by ODAT-SE (``take_step``, ``seed``, ...) cannot be set.

  Random hops are clipped into the search region (``min_list`` / ``max_list``).
  The random numbers are initialized from ``seed`` in the ``[algorithm]`` section.
  Note that the local optimization runs ``niter`` + 1 times in total, so the
  total number of solver evaluations is roughly
  (``niter`` + 1) x (evaluations per local optimization).
  When enabled, ``initial_scale_list`` (the initial simplex) is not used.

  Example:

  .. code-block:: toml

      [algorithm.minimize]
      method = "Nelder-Mead"

      [algorithm.minimize.basinhopping]
      niter = 50
      stepsize = 0.5


Output files
~~~~~~~~~~~~~~~~~

``SimplexData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Outputs information about the process of finding the minimum value.
The first line is a header, the second and subsequent lines are step,
the values of the variables in the order defined by ``label_list`` in the ``[algorithm]`` section of the input file (``x1``, ``x2``, ... by default),
and finally the value of the function.

The following is an example of the output.

.. code-block::

    #step z1 z2 z3 R-factor
    0 5.25 4.25 3.5 0.015199251773721183
    1 5.25 4.25 3.5 0.015199251773721183
    2 5.229166666666666 4.3125 3.645833333333333 0.013702918021532375
    3 5.225694444444445 4.40625 3.5451388888888884 0.012635279378225261
    4 5.179976851851851 4.348958333333334 3.5943287037037033 0.006001660077530159
    5 5.179976851851851 4.348958333333334 3.5943287037037033 0.006001660077530159

.. note::
   The ``R-factor`` column in the header contains the values of the objective function (the name comes from the former 2DMAT package).

``History_FunctionCall.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Records every call of the objective function during the optimization.
Each line contains the step number, the values of the variables, and the value of the objective function, in that order.
The first column is a step number shared by objective function and gradient evaluations.
Gradient evaluations also advance this number, so gaps may appear when gradients are used.
The number identifies an evaluation; it is not guaranteed to be consecutive.
The variables here are in solver units (``x / unit_list``), whereas both the
variables and gradients in ``History_Gradient.txt`` are in algorithm units.
Gradient evaluation step numbers are not recorded in either file.


``BasinHoppingData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written only when ``basinhopping`` is enabled.
For each local optimization (one from the initial point plus ``niter`` hops),
it outputs the hop number, the values of the variables at the local minimum,
the value of the objective function, and whether the hop was accepted (1/0),
in that order.

``History_Gradient.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written only when the gradient is used (``use_gradient = true`` with a solver that provides gradients).
The first line is a header (``#No``, variable names, ``fx``, and ``grad_`` followed by each variable name).
Each of the following lines contains the index, the values of the variables, the value of the objective function, and the gradient, in that order:
the first line is for the initial point, and the others are for each iteration of the local optimization.
Both variables and gradients are in algorithm units: for the solver objective
``f``, the recorded gradient is that of ``F(x) = f(x / unit_list)``, namely
``gradient(x / unit_list) / unit_list``. In contrast, the variables in
``History_FunctionCall.txt`` are in solver units (``x / unit_list``).
The index here counts history rows, not evaluation steps; gradient evaluation
step numbers are not recorded in output files.

``res.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the final objective function and the value of the parameters at that time are described.
The objective function is listed first, followed by the values of the variables in the order defined by ``label_list`` in the ``[algorithm]`` section of the input file (``x1``, ``x2``, ... by default).

The following is an example of the output.

.. code-block::

    fx = 7.382680568652868e-06
    z1 = 5.230524973874179
    z2 = 4.370622919269477
    z3 = 3.5961444501081647

Restart
~~~~~~~~~~~
Restarting is not supported by the ``minsearch`` algorithm,
regardless of the selected method and whether basin hopping is enabled.
