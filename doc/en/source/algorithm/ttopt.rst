Tensor Train Optimization ``ttopt``
***********************************

``ttopt`` is an ``Algorithm`` that performs parameter search by modeling the objective function as a large tensor indexed by the parameters and representing it as a network of 3-rank tensors (matrix product state MPS, tensor train) and searching the MPS using a gradient-free optimization method based on repeated cross-approximation.

Preparation
~~~~~~~~~~~

`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ should be installed when using MPI parallelization.

.. code-block::

  $ python3 -m pip install mpi4py

Input Parameters
~~~~~~~~~~~~~~~~

[``algorithm.param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: The minimum value that each parameter can take.

- ``max_list``

  Format: List of float. The length should match the value of dimension.

  Description: The maximum value that each parameter can take.

[``algorithm.ttopt``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following hyperparameters are supported:

- ``p_points``

  Format: Integer or list of integer. The length should match the value of dimension.

  Description: Bond dimension along each dimension. Each parameter is discretized into :math:`N_i = P_i^{q_i}` uniformly spaced points, which are represented by :math:`P_i` values taken by :math:`q_i` tensor legs. If an integer is provided, the same value is used for all dimensions. The default value is 2.

- ``q_points``

  Format: Integer or list of integer. The length should match the value of dimension.

  Description: Number of tensor legs along each dimension. Each parameter is discretized into :math:`N_i = P_i^{q_i}` uniformly spaced points, which are represented by :math:`P_i` values taken by :math:`q_i` tensor legs. If an integer is provided, the same value is used for all dimensions. The default value is 1.

- ``r_max``

  Format: Integer (default: 4)

  Description: Maximum bond dimension used to connect small tensors. Larger values result in more function evaluations at each step of the optimization.

- ``max_f_eval``

  Format: Integer (default: 10000)

  Description: Maximum number of cost function evaluations. This corresponds to the computational budget for the optimization. The counter is updated at the end of each sweep in the algorithm, and the parameter search terminates when the counter is greater than or equal to this variable.

- ``maxvol_tol``

  Format: Float (default: 1.001)

  Description: Tolerance used in computing the maximum volume submatrix. Exact decomposition of the matrix is done when the tolerance is set to 1. Values lower than 1 are clipped to 1.

- ``maxvol_max_it``

  Format: Integer (default: 1000)

  Description: Maximum number of iterations used in computing the maximum volume submatrix.

- ``init_points``

  Format: List of lists of float (default: [])

  Description: Initial guesses that are evaluated at the beginning of the optimization. Each inner list must have the same length as the dimension. This parameter is optional, and is used to inform the optimizer of existing candidate regions.

- ``save_eval_history``

  Format: Boolean (default: ``True``)

  Description: If ``True``, each evaluated candidate is appended to ``ttopt_eval_history.txt`` (MPI rank 0 only). Rows are flushed to disk whenever the in-memory buffer reaches ``eval_history_buffer_rows`` evaluations.

- ``eval_history_buffer_rows``

  Format: Integer (default: 256)

  Description: Frequency :math:`N_{\mathrm{flush}}` of writing to ``ttopt_eval_history.txt``. :math:`N_{\mathrm{flush}}` evaluations are written together.

Output Files
^^^^^^^^^^^^

``ttopt_hyperparameters.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At the end of the preparation phase (``prepare``), rank 0 writes the main hyperparameters, one field per line:
.. code-block::

    nprocs = 1
    bounds = [[-6.0, 6.0], [-6.0, 6.0]]
    p_points = [2 2]
    q_points = [20 20]
    r_max = 4
    max_f_eval = 10000
    maxvol_tol = 1.001
    maxvol_max_it = 1000
    save_eval_history = True
    eval_history_buffer_rows = 256

``ttopt_history.txt``
^^^^^^^^^^^^^^^^^^^^^

After each optimization step (after a batch of candidates is evaluated at one MPS core and the running best point is updated), the cumulative function evaluation count, the best point ``x_opt`` so far, and the best value ``fx_opt`` are appended. Leading ``#`` lines describe the columns.

.. code-block::

    # $1: count
    # $2: x_opt[0]
    # $3: x_opt[1]
    # $4: fx_opt
    8 3.420030040769616e+00 -9.735097632501253e-01 7.005409201321578e+00
    24 3.420030040769616e+00 -2.098510836134754e+00 2.643948442798083e+00
    ...

For dimension :math:`D`, data columns are: column 1 is the evaluation count, columns 2 through :math:`D+1` are ``x_opt[0], ..., x_opt[D-1]``, and the last column is ``fx_opt``.

``res.txt``
^^^^^^^^^^^

After the run, rank 0 writes the global best solution (best across ranks when MPI is used) as text: objective ``fx`` followed by each parameter (default labels ``x1``, ``x2``, ...).

.. code-block::

    fx = 3.188892404355571e-08
    x1 = 3.584424576210571
    x2 = -1.8480795365138398

``ttopt_eval_history.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Created only if ``save_eval_history`` is ``True``: ``OUTPUT/ttopt_eval_history.txt``.
Each row is one actually evaluated candidate with its coordinates and ``f(x)`` (leading ``#`` lines give column labels; parameter names follow ``label_list`` when set).

.. code-block::

    # $1: row index (order of batches in the run)
    # $2: x1
    # $3: x2
    # $4: f(x)
    1 3.420030040769616e+00 -5.098513697160432e+00 5.218032809779967e+02
    2 3.420030040769616e+00 -9.735097632501253e-01 7.005409201321578e+00
    ...

``time.log``
^^^^^^^^^^^^

Total timing for the algorithm is written per rank to ``OUTPUT/<rank>/time.log``.

Algorithm Description
~~~~~~~~~~~~~~~~~~~~~

Tensor Train Optimization (TTOpt) [1] is a method for finding the optimum of a discrete function and its location in the parameter space. It can be easily adapted to continuous optimization problems and is able to handle high-dimensional problems and functions whose parameters are a combination of discrete and continuous quantities.

TTOpt is a gradient-free optimization scheme based on repeated cross-approximation of a large tensor whose elements correspond to values of the objective function :math:`f(x_1, x_2, ..., x_n)` indexed by parameter combinations :math:`(x_1, x_2, ..., x_n)`. The cross-approximation is computed using approximate maximum-volume submatrices.
Each parameter :math:`x_i` is discretized into :math:`N_i = P_i^{q_i}` uniformly spaced points, which are represented by :math:`P_i` values taken by :math:`q_i` tensor legs. This way, the objective function is represented as a :math:`\prod_i q_i`-rank tensor.
The TTOpt algorithm decomposes this high-dimensional tensor into a network of 3-rank tensors (MPS, tensor train).

The algorithm is designed such that only a small part of the whole large tensor needs to be explicitly computed. Thus, this approach is advantageous when the objective function is computationally costly or when the search space is very large. Furthermore, by representing the data in MPS form, we can avoid having to form exponentially large matrices in the optimization process.

For this algorithm, execution using multiple MPI processes is possible. When MPI is used, evaluation of the objective function values at the sampled points is divided across the different ranks.

References
^^^^^^^^^^

[1] K. Sozykin et al., `arXiv:2205.00293 <https://arxiv.org/abs/2205.00293>`_ (2022).
