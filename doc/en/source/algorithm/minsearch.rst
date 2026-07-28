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
:math:`D`, the optimal solution is searched by systematically moving pairs of
:math:`D+1` coordinate points according to the value of the objective function
at each point.

An important hyperparameter is the initial value of the coordinates.
Since these local optimization methods have the problem of being trapped in a
local optimum, it is recommended to repeat the calculation with different
initial values several times to check the results, or to enable the optional
global optimization by basin hopping (`scipy.optimize.basinhopping`_), which
repeats a random hop followed by a local optimization with the selected method
(see the ``basinhopping`` parameter below).


Preparation
~~~~~~~~~~~

You will need to install `scipy <https://docs.scipy.org/doc/scipy/reference>`_ .

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
  In the search algorithm, each parameter is divided by each of these values to perform a simple dimensionless and normalization.
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
``initial_scale_list``, and ``basinhopping`` (described below) are passed
verbatim to the ``options`` argument of `scipy.optimize.minimize`_.
If a parameter name not accepted by the selected method is given,
the program stops with an error before the optimization starts.
The default values of ``xatol``, ``fatol``, ``maxiter``, and ``maxfev``
listed below apply only when ``method`` is "Nelder-Mead".

- ``method``

  Format: String (default: "Nelder-Mead")

  Description:
  Name of the optimization method, passed as-is to the ``method`` argument of `scipy.optimize.minimize`_.
  Examples: "Nelder-Mead", "Powell", "COBYLA".
  Note that for gradient-based methods (BFGS, CG, ...) the gradient is evaluated by numerical differentiation,
  which costs dimension+1 solver evaluations per gradient evaluation.
  The search region (``min_list`` / ``max_list``) is passed as the ``bounds`` argument of scipy
  for methods that support it (Powell, L-BFGS-B, TNC, SLSQP, trust-constr, COBYLA, COBYQA).
  For the Nelder-Mead method, out-of-range points are handled as before
  by treating the objective function value as infinity.

- ``initial_scale_list``

  Format: List of float. The length should match the value of dimension.

  Description:
  The difference value that is shifted from the initial value in order to create the initial simplex for the Nelder-Mead method.
  The ``initial_simplex`` is given by the sum of ``initial_list`` and the dimension of the ``initial_list`` plus one component of the ``initial_scale_list``.
  If not defined, scales at each dimension are set to 0.25.
  Used only when ``method`` is "Nelder-Mead".

- ``xatol``

  Format: Float (default: 1e-4)

  Description: Parameters used to determine convergence of the Nelder-Mead method.

- ``fatol``

  Format: Float (default: 1e-4)

  Description: Parameters used to determine convergence of the Nelder-Mead method.

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
the values of variables defined in ``string_list`` in the ``[solver]`` - ``[param]`` sections of the input file,
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

``History_FunctionCall.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Records every call of the objective function during the optimization.
Each line contains the call number, the values of the variables, and the value of the objective function, in that order.

``BasinHoppingData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written only when ``basinhopping`` is enabled.
For each local optimization (one from the initial point plus ``niter`` hops),
it outputs the hop number, the values of the variables at the local minimum,
the value of the objective function, and whether the hop was accepted (1/0),
in that order.

``res.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the final objective function and the value of the parameters at that time are described.
The objective function is listed first, followed by the values of the variables defined in ``string_list`` in the ``[solver]`` - ``[param]`` sections of the input file, in that order.

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
