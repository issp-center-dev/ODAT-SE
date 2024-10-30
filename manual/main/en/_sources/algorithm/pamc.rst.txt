Population Annealing Monte Carlo ``pamc``
===============================================

``pamc`` explores the parameter space by using the Population Annealing Monte Carlo (PAMC) method.

Preparation
~~~~~~~~~~~~~~~~

`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ should be installed for the MPI parallelization ::

  python3 -m pip install mpi4py

Input parameters
~~~~~~~~~~~~~~~~~~~

This has two subsections ``algorithm.param`` and ``algorithm.pamc`` .

[``algorithm.param``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This defines a space to be explored.
When ``mesh_path`` key is defined the discrete space is used.
Otherwise, continuous space is used.

- Continuous space

  - ``initial_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Initial value of parameters.
    If not defined, these will be initialize randomly.

  - ``unit_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Unit length of each parameter.
    ``Algorithm`` makes parameters dimensionless and normalized by dividing these by ``unit_list``.
    If not defined, each component will be 1.0.

  - ``min_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Minimum value of each parameter.
                 When a parameter falls below this value during the Monte Carlo search,
                 the solver is not evaluated and the value is considered infinite.


  - ``max_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Maximum value of each parameter.
                 When a parameter exceeds this value during the Monte Carlo search,
                 the solver is not evaluated and the value is considered infinite.

- Discrete space

  - ``mesh_path``

    Format: string

    Description: Path to the mesh definition file.

  - ``neighborlist_path``

    Format: string

    Description: Path to the neighborhood-list file.

[``algorithm.pamc``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``numsteps``

  Format: Integer

  Description: The number of Monte Carlo steps.

- ``numsteps_annealing``

  Format: Integer

  Description: The number of interval Monte Carlo steps between lowering "temperature".

- ``numT``

  Format: Integer

  Description: The number of "temperature" points.

- ``Tmin``

  Format: Float

  Description: The minimum value of the "temperature" (:math:`T`).

- ``Tmax``

  Format: Float

  Description: The maximum value of the "temperature" (:math:`T`).

- ``bmin``

  Format: Float

  Description: The minimum value of the "inverse temperature" (:math:`\beta = 1/T`).
  One of the "temperature" and "inverse temperature" should be defined.

- ``bmax``

  Format: Float

  Description: The maximum value of the "inverse temperature" (:math:`\beta = 1/T`).
  One of the "temperature" and "inverse temperature" should be defined.

- ``Tlogspace``

  Format: Boolean (default: true)

  Description: Whether to assign "temperature" to replicas equally spaced in the logarithmic space or not.

- ``nreplica_per_proc``

  Format: Integer (default: 1)

  Description: The number of replicas in a MPI process.

- ``resampling_interval``

  Format: Integer (default: 1)

  Description: The number of annealing processes between resampling of the replicas.

- ``fix_num_replicas``

  Format: Boolean (default: true)

  Description: Whether to fix the number of replicas or not on resampling.


About the number of steps
********************************

Specify just two of ``numstep``, ``numsteps_annealing``, and ``numT``.
The value of the remaining one will be determined automatically.

Reference file
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh definition file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the grid space to be explored in this file.
The first column is the index of the mesh, and the second and subsequent columns are the values of variables.
Note that the index of the mesh will be ignored for this "algorithm".

Below, a sample file is shown.

.. code-block::

    1 6.000000 6.000000
    2 6.000000 5.750000
    3 6.000000 5.500000
    4 6.000000 5.250000
    5 6.000000 5.000000
    6 6.000000 4.750000
    7 6.000000 4.500000
    8 6.000000 4.250000
    9 6.000000 4.000000
    ...


Neighborhood-list file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before searching in the discrete space by Markov Chain Monte Carlo method,
we should define "neighborhoods" for each point :math:`i`, which are points that a walker can move from :math:`i`
A neighborhood-list file defines the list of neighborhoods.
In this file, the index of an initial point :math:`i` is specified by the first column,
and the indices of final points :math:`j` are specified by the second and successive columns.

An utility tool, ``py2dmat_neighborlist`` is available for generating a neighborhood-list file from a mesh file. For details, please see :doc:`../tool`.

.. code-block::

    0 1 2 3
    1 0 2 3 4
    2 0 1 3 4 5
    3 0 1 2 4 5 6 7
    4 1 2 3 5 6 7 8
    5 2 3 4 7 8 9
    ...

Output files
~~~~~~~~~~~~~~~~~~~~~

``RANK/trial_T#.txt``
^^^^^^^^^^^^^^^^^^^^^
This file stores the suggested parameters and the corresponding value returned from the solver for each temperature point (specified by ``#``).
The first column (``step``) is the index of the MC step.
The second column (``walker``) is the index of the walker in the process.
The third column (``beta``) is the inverse temperature of the replica.
The fourth column (``fx``) is the value of the solver.
The fifth - (4+dimension)-th columns are the coordinates.
The last two columns (``weight`` and ``ancestor``) are the Neal-Jarzynsky weight and the grand-ancestor of the replica.

Example::

    # step walker beta fx x1 weight ancestor
    0 0 0.0 73.82799488298886 8.592321856342956 1.0 0
    0 1 0.0 13.487174782058675 -3.672488908364282 1.0 1
    0 2 0.0 39.96292704464803 -6.321623766458111 1.0 2
    0 3 0.0 34.913851603463 -5.908794428939206 1.0 3
    0 4 0.0 1.834671825646121 1.354500581633733 1.0 4
    0 5 0.0 3.65151610695736 1.910894059585031 1.0 5
    ...


``RANK/trial.txt``
^^^^^^^^^^^^^^^^^^^^^

This is a combination of all the ``trial_T#.txt`` in one.

``RANK/result_T#.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^
This file stores the sampled parameters and the corresponding value returned from the solver for each replica and each temperature.
This has the same format as ``trial.txt``.

.. code-block::

    # step walker beta fx x1 weight ancestor
    0 0 0.0 73.82799488298886 8.592321856342956 1.0 0
    0 1 0.0 13.487174782058675 -3.672488908364282 1.0 1
    0 2 0.0 39.96292704464803 -6.321623766458111 1.0 2
    0 3 0.0 34.913851603463 -5.908794428939206 1.0 3
    0 4 0.0 1.834671825646121 1.354500581633733 1.0 4
    0 5 0.0 3.65151610695736 1.910894059585031 1.0 5
    ...

``RANK/result.txt``
^^^^^^^^^^^^^^^^^^^^^

This is a combination of all the ``result_T#.txt`` in one.

``best_result.txt``
^^^^^^^^^^^^^^^^^^^^
The optimal value of the solver and the corresponding parameter among the all samples.

.. code-block::

    nprocs = 4
    rank = 2
    step = 65
    fx = 0.008233957976993406
    z1 = 4.221129370933539
    z2 = 5.139591716517661


``fx.txt``
^^^^^^^^^^^^^^

This file stores statistical metrics over the all replicas for each temperature.
The first column is inverse temperature.
The second and third column are the expectation value and the standard error of the solver's output (:math:`f(x)`), respectively.
The fourth column is the number of replicas.
The fifth column is the logarithmic of the ratio between the normalization factors (partition functions)

.. math::

   \log\frac{Z}{Z_0} = \log\int \mathrm{d}x e^{-\beta f(x)} - \log\int \mathrm{d}x e^{-\beta_0 f(x)},

where :math:`\beta_0` is the minimum value of :math:`\beta` used in the calculation.
The sixth column is the acceptance ratio of MC updates.

.. code-block::

    # $1: 1/T
    # $2: mean of f(x)
    # $3: standard error of f(x)
    # $4: number of replicas
    # $5: log(Z/Z0)
    # $6: acceptance ratio
    0.0 33.36426034198166 3.0193077565358273 100 0.0 0.9804
    0.1 4.518006242920819 0.9535301415484388 100 -1.2134775491597027 0.9058
    0.2 1.5919146358616842 0.2770369776964151 100 -1.538611313376179 0.9004
    ...


Algorithm
~~~~~~~~~~

Goal
^^^^^

When the weight of the configuration :math:`x` under some parameter :math:`\beta_i` is given as :math:`f_i(x)`
(e.g., the Bolzmann factor :math:`f_i(x) = \exp[-\beta_i E(x)]` ),
the expectation value of :math:`A` is defined as

.. math::

   \langle A\rangle_i
   = \frac{\int \mathrm{d}xA(x)f_i(x)}{\int \mathrm{d}x f_i(x)}
   = \frac{1}{Z}\int \mathrm{d}xA(x)f_i(x)
   = \int \mathrm{d}xA(x)\tilde{f}_i(x),

where :math:`Z = \int \mathrm{d} x f_i(x)` is the normalization factor (partition function)
and :math:`\tilde{f}(x) = f(x)/Z` is the probability of :math:`x`.

Our goal is to numerically calculate the expectation value for each :math:`\beta_i` and the (ratio of) the normalization factor.

Annealed Importance Sampling (AIS) [1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we introduce a series of configurations :math:`\{x_i\}` obeying the following joint probability

.. math::

   \tilde{f}(x_0, x_1, \dots, x_n) = \tilde{f}_n(x_n) \tilde{T}_n(x_n, x_{n-1}) \tilde{T}_{n-1}(x_{n-1}, x_{n-2}) \cdots \tilde{T}_1(x_1, x_0),

with

.. math::

   \tilde{T}_i(x_i, x_{i-1}) = T_i(x_{i-1}, x_i) \frac{\tilde{f}_i(x_{i-1})}{\tilde{f}_i(x_i)},

where :math:`T_i(x, x')` is a transition probability from :math:`x` to :math:`x'` under :math:`\beta_i`
holding the balance condition,

.. math::


   \int \mathrm{d}x \tilde{f}_i(x) T_i(x, x') = \tilde{f}_i(x').

It turns out that :math:`\tilde{f}_n(x_n)` is the marginal distribution of :math:`\tilde{f}(x_0, x_1, \dots, x_n)`, that is,

.. math::


   \tilde{f}_n(x_n) = \int \prod_{i=0}^{n-1} \mathrm{d} x_i \tilde{f}(x_0, x_1, \dots, x_n),

from 

.. math::

   \int \mathrm{d} x_{i-1} \tilde{T}_i(x_i, x_{i-1})
   = \int \mathrm{d} x_{i-1} \tilde{f}_i(x_{i-1}) T_i(x_{i-1}, x_i) / \tilde{f}_i(x_i)
   = 1.

Consequently,
:math:`\langle A \rangle_n` is represented by using the extended configuration :math:`\{x_i\}` as

.. math::


   \begin{split}
   \langle A \rangle_n
   &\equiv
   \int \mathrm{d} x_n A(x_n) \tilde{f}_n(x_n) \\
   &= \int \prod_i \mathrm{d} x_i A(x_n) \tilde{f}(x_0, x_1, \dots, x_n).
   \end{split}


Unfortunately, it is difficult to generate directly a series of configurations :math:`\{x_i\}`
following the distribution :math:`\tilde{f}(x_0, x_1, \dots, x_n)`.
Then, instead of :math:`\tilde{f}(x_0, x_1, \dots, x_n)`, we consider :math:`\{x_i\}` obeying the joint distribution

.. math::

   \tilde{g}(x_0, x_1, \dots, x_n) = \tilde{f}_0(x_0) T_1(x_0, x_1) T_2(x_1, x_2) \dots T_n(x_{n-1}, x_n),


by using the following the following scheme:

1. Generete :math:`x_0` from the initial distribution :math:`\tilde{f}_0(x)`

2. Generate :math:`x_{i+1}` from :math:`x_i` through :math:`T_{i+1}(x_i, x_{i+1})`


By using the reweighting method (or importance sampling method), 
:math:`\langle A \rangle_n` is rewritten as

.. math::


   \begin{split}
   \langle A \rangle_n
   &= \int \prod_i \mathrm{d} x_i A(x_n) \tilde{f}(x_0, x_1, \dots, x_n) \\
   &= \int \prod_i \mathrm{d} x_i A(x_n) \frac{\tilde{f}(x_0, x_1, \dots, x_n)}{\tilde{g}(x_0, x_1, \dots, x_n)} \tilde{g}(x_0, x_1, \dots, x_n) \\
   &= \left\langle A\tilde{f}\big/\tilde{g} \right\rangle_{g, n}
   \end{split}.

Because the ratio between :math:`\tilde{f}` and :math:`\tilde{g}` is 

.. math::


   \begin{split}
   \frac{\tilde{f}(x_0, \dots, x_n)}{\tilde{g}(x_0, \dots, x_n)}
   &= 
   \frac{\tilde{f}_n(x_n)}{\tilde{f}_0(x_0)}
   \prod_{i=1}^n \frac{\tilde{T}_i(x_i, x_{i-1})}{T(x_{i-1}, x_i)} \\
   &=
   \frac{\tilde{f}_n(x_n)}{\tilde{f}_0(x_0)}
   \prod_{i=1}^n \frac{\tilde{f}_i(x_{i-1})}{\tilde{f}_i(x_i)} \\
   &=
   \frac{Z_0}{Z_n}
   \frac{f_n(x_n)}{f_0(x_0)}
   \prod_{i=1}^n \frac{f_i(x_{i-1})}{f_i(x_i)} \\
   &=
   \frac{Z_0}{Z_n}
   \prod_{i=0}^{n-1} \frac{f_{i+1}(x_{i})}{f_i(x_i)} \\
   &\equiv
   \frac{Z_0}{Z_n} w_n(x_0, x_1, \dots, x_n),
   \end{split}

the form of the expectation value will be

.. math::

   \langle A \rangle_n = \left\langle A\tilde{f}\big/\tilde{g} \right\rangle_{g, n}
   = \frac{Z_0}{Z_n} \langle Aw_n \rangle_{g,n}.


Finally, the ratio between the normalization factors :math:`Z_n/Z_0` can be evaluated as

.. math::

   \frac{Z_n}{Z_0} = \langle w_n \rangle_{g,n},

and therefore the expectation value of :math:`A` can be evaluated as a weighted arithmetic mean:

.. math::

   \langle A \rangle_n = \frac{\langle Aw_n \rangle_{g,n}}{\langle w_n \rangle_{g,n}}.

This weight :math:`w_n` is called as the Neal-Jarzynski weight.

population annealing (PA) [2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the AIS method can estimate the expectation values of :math:`A` for each parameter :math:`\beta` as the form of weighted arithmetic mean,
the variance of weights :math:`w` is generally large and then the accuracy of the result gets worse.
In order to overcome this problem, the population annealing Monte Carlo (PAMC) method resamples all the replicas according to the probability
:math:`p^{(k)} = w^{(k)} / \sum_k w^{(k)}` at some periods and resets all the weights to unity.

The following pseudo code describes the scheme of PAMC:

.. code-block:: python

    for k in range(K):
        w[0, k] = 1.0
        x[0, k] = draw_from(β[0])
    for i in range(1, N):
        for k in range(K):
            w[i, k] = w[i-1, k] * ( f(x[i-1,k], β[i]) / f(x[i-1,k], β[i-1]) )
        if i % interval == 0:
            x[i, :] = resample(x[i, :], w[i, :])
            w[i, :] = 1.0
        for k in range(K):
            x[i, k] = transfer(x[i-1, k], β[i])
        a[i] = sum(A(x[i,:]) * w[i,:]) / sum(w[i,:])

There are two resampling methods: one with a fixed number of replicas[2] and one without[3].

References
^^^^^^^^^^^^^

[1] R. M. Neal, Statistics and Computing **11**, 125-139 (2001).

[2] K. Hukushima and Y. Iba, AIP Conf. Proc. **690**, 200 (2003).

[3] J. Machta, PRE **82**, 026704 (2010).
