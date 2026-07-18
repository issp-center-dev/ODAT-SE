=========================================
Global optimization ``global_search``
=========================================

.. _scipy.optimize.differential_evolution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
.. _scipy.optimize.shgo: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html

``global_search`` minimizes :math:`f(x)` using the global optimization
routines of scipy.optimize.
The following methods are currently available (direct is planned):

- Differential evolution (`scipy.optimize.differential_evolution`_):
  an evolutionary algorithm that maintains a population of candidate
  solutions and generates new candidates from difference vectors between
  population members. It is derivative-free and robust for multimodal
  problems.
- shgo (simplicial homology global optimization, `scipy.optimize.shgo`_):
  a deterministic method that builds a simplicial complex over sampling
  points and systematically selects starting points of local optimizations
  from its topological structure. It can report the list of **all local
  minima** found.

The search region is defined by ``min_list`` / ``max_list`` of
``[algorithm.param]`` and passed as the ``bounds`` argument of scipy.
The initial value (``initial_list``) is not used.

MPI parallelization
~~~~~~~~~~~~~~~~~~~~~

Under MPI, algorithm rank 0 drives the optimizer while the other ranks act
as evaluation servers. For differential evolution, the candidate points of a
whole generation are distributed to the ranks at once; for shgo, the
evaluation points of the sampling phase are. Each rank evaluates its share
with its own solver group. The point-level parallelism (number of algorithm
ranks) composes with the solver-side parallelism (``nsolve``), giving two
levels of parallelization.

For differential evolution, the number of objective function evaluations per
generation is ``popsize`` x dimension, and the total is roughly bounded by
(``maxiter`` + 1) x ``popsize`` x dimension (the run may stop earlier upon
convergence). The local refinements of shgo (its internal local
optimizations) run serially on rank 0.

Preparation
~~~~~~~~~~~

You will need to install `scipy <https://docs.scipy.org/doc/scipy/reference>`_ .

.. code-block::

   $ python3 -m pip install scipy

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It has subsections ``param`` and ``global_search``.

``[algorithm.param]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: Minimum value of each parameter.

- ``max_list``

  Format: List of float. The length should match the value of dimension.

  Description: Maximum value of each parameter.

- ``unit_list``

  Format: List of float. The length should match the value of dimension.

  Description:
  Units for each parameter.
  In the search algorithm, each parameter is divided by each of these values
  to perform a simple dimensionless and normalization.
  If not defined, the value is 1.0 for all dimensions.

``[algorithm.global_search]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the optimization method and its hyperparameters.

All parameters other than ``method`` are passed verbatim as arguments of the
selected scipy routine. If an argument name not accepted by the routine is
given, the program stops with an error before the optimization starts.
Arguments managed by ODAT-SE (``bounds``, ``workers``, ``seed``, ...) cannot
be set.

- ``method``

  Format: String (default: "DE")

  Description: Name of the optimization method (case-insensitive).
  "DE" or "differential_evolution" selects differential evolution;
  "shgo" selects shgo. "direct" is planned.

- other parameters

  Arguments of the selected scipy routine can be given directly.
  See the scipy documentation for details.

  - Differential evolution: ``popsize``, ``maxiter``, ``tol``,
    ``mutation``, ``recombination``, ``strategy``, ``polish``, ...
    The random numbers are initialized from ``seed`` in the ``[algorithm]``
    section (the random number sequence of algorithm rank 0 is used).
  - shgo: ``n``, ``iters``, ``sampling_method``, ...
    The sub-tables ``[algorithm.global_search.options]`` and
    ``[algorithm.global_search.minimizer_kwargs]`` are passed as the
    ``options`` / ``minimizer_kwargs`` arguments of scipy, respectively.
    shgo is deterministic and does not use random numbers.

Example:

.. code-block:: toml

    [algorithm]
    name = "global_search"
    seed = 12345

    [algorithm.param]
    min_list = [-5.0, -5.0]
    max_list = [ 5.0,  5.0]

    [algorithm.global_search]
    method = "DE"
    popsize = 15
    maxiter = 100

Remarks
~~~~~~~~~~~~~~~~~

- When ``polish`` (default: true) is enabled for differential evolution, a
  local optimization by L-BFGS-B runs after it finishes. It runs serially on
  rank 0 and evaluates gradients by numerical differentiation
  (dimension+1 solver evaluations per gradient). Consider ``polish = false``
  when solver evaluations are expensive.
- The local refinements of shgo also run serially on rank 0. Its default
  local minimizer is SLSQP, whose gradients are evaluated by numerical
  differentiation.
- The parallel evaluation of shgo (``workers``) requires scipy >= 1.11;
  older versions stop with an error before the optimization starts.
- Constraints given by ``[runner.limitation]`` are handled by treating the
  objective function value of violating points as infinity.
- Restarting (checkpointing) is not supported.

Output files
~~~~~~~~~~~~~~~~~

``GenerationData.txt`` / ``IterationData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Records the best point of each iteration (rank 0 only).
For differential evolution, ``GenerationData.txt`` contains the generation
number, the values of the variables of the best point, the value of the
objective function, and the convergence measure, in that order.
For shgo, ``IterationData.txt`` contains the iteration number, the values of
the variables of the best point, and the value of the objective function.

``LocalMinimaData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Written only for shgo (rank 0 only).
Lists all local minima found: the index, the values of the variables, and
the value of the objective function, sorted in ascending order of the
objective function.

``History_FunctionCall.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Records the history of the objective function calls evaluated by each rank,
in the per-rank output directory. Each line contains the call number, the
values of the variables, and the value of the objective function, in that
order.

``res.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the final objective function and the values of the parameters
at that time.

.. code-block::

    fx = 4.119494492750836e-11
    x1 = 6.135735138280041e-06
    x2 = 1.4614801832975428e-06

Restart
~~~~~~~~~~~

Restarting is not supported for ``global_search``.
