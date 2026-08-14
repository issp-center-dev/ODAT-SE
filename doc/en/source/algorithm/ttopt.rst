Tensor Train Optimization ``ttopt``
***********************************

``ttopt`` is an ``Algorithm`` that performs parameter search using Tensor Train Optimization (TTOpt).
It treats the objective function as a high-dimensional tensor indexed by the parameter combinations, and searches for the optimal value and its location without gradients by selectively evaluating only a small fraction of the tensor via cross approximation.
See "Algorithm Description" below for an overview of the method.

Preparation
~~~~~~~~~~~

TTOpt requires scipy. In addition, `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ should be installed when using MPI parallelization.

.. code-block::

  $ python3 -m pip install scipy
  $ python3 -m pip install mpi4py

Input Parameters
~~~~~~~~~~~~~~~~

``[algorithm.param]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: The minimum value that each parameter can take.

- ``max_list``

  Format: List of float. The length should match the value of dimension.

  Description: The maximum value that each parameter can take.

``[algorithm.ttopt]`` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following hyperparameters are supported:

- ``p_points``

  Format: Integer or list of integer. The length should match the value of dimension.

  Description: Base :math:`P_i` used to discretize parameter :math:`x_i` (the number of values each decomposed index takes; 2 for a binary representation). Each parameter is discretized into :math:`N_i = P_i^{q_i}` uniformly spaced points. If an integer is provided, the same value is used for all dimensions. The default value is 2.

- ``q_points``

  Format: Integer or list of integer. The length should match the value of dimension.

  Description: Number of decomposed indices :math:`q_i` for parameter :math:`x_i` (the exponent that determines the number of grid points :math:`N_i = P_i^{q_i}`; the number of bits for a binary representation). If an integer is provided, the same value is used for all dimensions. The default value is 1.

.. note::
   With the default settings (``p_points = 2``, ``q_points = 1``), each parameter is discretized into only :math:`2^1 = 2` points.
   When optimizing continuous parameters, set ``q_points`` to around 10--20
   (:math:`2^{10} \approx 10^3` to :math:`2^{20} \approx 10^6` grid points) to obtain sufficient resolution.

- ``r_max``

  Format: Integer (default: 4)

  Description: Maximum rank of the tensor approximation (the maximum bond dimension connecting the small tensors). Larger values can capture more complicated functions, but result in more objective-function evaluations at each step of the optimization.

- ``max_f_eval``

  Format: Integer (default: 10000)

  Description: Maximum number of objective-function evaluations. This corresponds to the computational budget for the optimization. The counter is updated after each batch of candidate points is evaluated, and the limit is checked after the whole batch, so the actual count can slightly exceed ``max_f_eval``. The counter counts the requested candidate points, including those already cached.

- ``maxvol_tol``

  Format: Float (default: 1.001)

  Description: Stopping threshold of the iterative maximum-volume submatrix search (the maxvol method described below). Values closer to 1 make the iteration continue until a stricter maximum-volume condition is reached; setting it to exactly 1 requires the exact maximum-volume condition. Values below 1 are not meaningful: the stopping condition can never be satisfied, so the iteration always runs up to ``maxvol_max_it``. The default value is usually sufficient.

- ``maxvol_max_it``

  Format: Integer (default: 1000)

  Description: Maximum number of iterations used in computing the maximum volume submatrix.

- ``init_points``

  Format: List of lists of float (default: [])

  Description: Initial guesses that are evaluated at the beginning of the optimization. Each inner list must have the same length as the dimension. This parameter is optional, and is used to inform the optimizer of existing candidate regions.

- ``save_eval_history``

  Format: Boolean (default: ``true``)

  Description: If ``true``, each evaluated candidate is appended to ``ttopt_eval_history.txt`` (MPI rank 0 only). Rows are flushed to disk whenever the in-memory buffer reaches ``eval_history_buffer_rows`` evaluations.

- ``eval_history_buffer_rows``

  Format: Integer (default: 256)

  Description: Flush threshold :math:`N_{\mathrm{flush}}` for writing to ``ttopt_eval_history.txt``. The whole pending buffer is written out once the number of buffered rows reaches or exceeds :math:`N_{\mathrm{flush}}`.

Output Files
~~~~~~~~~~~~

In the following, ``OUTPUT`` denotes ``output_dir`` specified in the ``[base]`` section (when given as a relative path, ``root_dir/output_dir`` is used).
Each MPI rank uses a rank-numbered subdirectory under ``OUTPUT`` (e.g. ``OUTPUT/0/``) as its working directory. The main files written by TTOpt are placed directly under ``OUTPUT`` (created by MPI rank 0 only, unless stated otherwise).

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

After each optimization step (after a batch of candidates is evaluated and the running best point is updated), the cumulative function evaluation count, the best point ``x_opt`` so far, and the best value ``fx_opt`` are appended. Leading ``#`` lines describe the columns.

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

Created only if ``save_eval_history`` is ``true``: ``OUTPUT/ttopt_eval_history.txt``.
Each row is one requested candidate (including cached points for which the solver was not called again) with its coordinates and ``f(x)`` (leading ``#`` lines give column labels; parameter names follow ``label_list`` when set). Column 1 is the sequential row index.

.. code-block::

    # $1: row index
    # $2: x1
    # $3: x2
    # $4: f(x)
    1 3.420030040769616e+00 -5.098513697160432e+00 5.218032809779967e+02
    2 3.420030040769616e+00 -9.735097632501253e-01 7.005409201321578e+00
    ...

``time.log``
^^^^^^^^^^^^

Total timing for the algorithm is written to ``OUTPUT/0/time.log`` by the rank-0 process of the algorithm layer only.

Algorithm Description
~~~~~~~~~~~~~~~~~~~~~

Tensor Train Optimization (TTOpt) [1] is a gradient-free black-box optimization method for high-dimensional discrete optimization problems.
It can also be applied to continuous variables by discretizing each of them, and thus handles problems with a mixture of discrete and continuous variables.

TTOpt regards the objective function :math:`f(x_1, \dots, x_n)` as the elements of a high-dimensional tensor indexed by the parameter combinations.
This huge tensor is never computed explicitly.
Instead, TTOpt exploits a low-rank representation in the Tensor Train (TT, equivalent to the matrix product state MPS) format and selectively evaluates only the required tensor elements using a cross-approximation technique called TT-cross.

In TT-cross, the maxvol (maximum-volume) method is used to select sample points corresponding to the rows and columns that capture the structure of the tensor efficiently.
Repeating this operation along each dimension of the Tensor Train, the algorithm searches for the optimal value and its location while keeping the number of objective-function evaluations small.

In addition, when each parameter :math:`x_i` has :math:`N_i = P_i^{q_i}` discrete points, the index can be "quantized", i.e., decomposed into :math:`q_i` smaller indices each taking :math:`P_i` states.
This allows even very finely discretized parameter spaces to be treated as a higher-order tensor with small local dimensions.
The input parameters ``p_points`` and ``q_points`` correspond to :math:`P_i` and :math:`q_i`, respectively.

Since this method does not need to store or evaluate the whole search space, it is effective when the parameter space is huge or when a single objective-function evaluation is computationally expensive.
Moreover, the evaluations of the objective function at the sample points are independent of each other, so they can be distributed over multiple processes using MPI.

Restart
~~~~~~~~~~~~~~~~~

When ``algorithm.checkpoint`` is set to true, the intermediate state is stored
to ``status.pickle`` on the following occasions:

#. a double sweep (right-to-left followed by left-to-right) has completed and
   the ``checkpoint_steps`` or ``checkpoint_interval`` condition is met.
#. the run ends because ``max_f_eval`` has been reached (the final state).

``run_mode`` corresponds to the ``--init``, ``--resume``, and ``--cont`` options
of the ``odatse`` command.

- ``"initial"``

  Run from the beginning.

- ``"resume"``

  Restart an interrupted calculation from the checkpoint.

- ``"continue"``

  Extend a finished calculation. For TTOpt it behaves exactly as ``"resume"``:
  the evaluation budget ``max_f_eval`` is not checkpointed and is re-read from
  the input file on every run, so raising ``max_f_eval`` and restarting
  continues the search from the stored state.

.. note::
   The result of an extended run is identical to that of a single run started
   with the larger ``max_f_eval``. Some evaluations are repeated on restart, but
   the cache of evaluated values is part of the checkpoint, so the solver is not
   called again for them.

References
^^^^^^^^^^

[1] K. Sozykin et al., `arXiv:2205.00293 <https://arxiv.org/abs/2205.00293>`_ (2022).
