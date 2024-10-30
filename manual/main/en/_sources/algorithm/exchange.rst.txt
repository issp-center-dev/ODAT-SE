Replica exchange Monte Carlo ``exchange``
=============================================

``exchange`` explores the parameter space by using the replica exchange Monte Carlo (RXMC) method.

Preparation
~~~~~~~~~~~~~~~~

`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ should be installed. ::

  python3 -m pip install mpi4py

Input parameters
~~~~~~~~~~~~~~~~~~~

This has two subsections ``algorithm.param`` and ``algorithm.exchange`` .

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

[``algorithm.exchange``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``numsteps``

  Format: Integer

  Description: The number of Monte Carlo steps.

- ``numsteps_exchange``

  Format: Integer

  Description: The number of interval Monte Carlo steps between replica exchange.

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

``RANK/trial.txt``
^^^^^^^^^^^^^^^^^^^^^
This file stores the suggested parameters and the corresponding value returned from the solver for each replica.
The first column is the index of the MC step.
The second column is the index of the walker in the process.
The third column is the temperature of the replica.
The fourth column is the value of the solver.
The remaining columns are the coordinates.

Example::

    # step walker T fx z1 z2
    0 0 0.004999999999999999 0.07830821484593968 3.682008067401509 3.9502750191292586 
    1 0 0.004999999999999999 0.0758494287185766 2.811346329442423 3.691101784194861 
    2 0 0.004999999999999999 0.08566823949124412 3.606664760390988 3.2093903670436497 
    3 0 0.004999999999999999 0.06273922648753057 4.330900869594549 4.311333132184154 


``RANK/result.txt``
^^^^^^^^^^^^^^^^^^^^^
This file stores the sampled parameters and the corresponding value returned from the solver for each replica.
This has the same format as ``trial.txt``.

.. code-block::

    # step walker T fx z1 z2
    0 0 0.004999999999999999 0.07830821484593968 3.682008067401509 3.9502750191292586 
    1 0 0.004999999999999999 0.07830821484593968 3.682008067401509 3.9502750191292586 
    2 0 0.004999999999999999 0.07830821484593968 3.682008067401509 3.9502750191292586 
    3 0 0.004999999999999999 0.06273922648753057 4.330900869594549 4.311333132184154 


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


``result_T#.txt``
^^^^^^^^^^^^^^^^^^^
This file stores samples for each temperature ( ``#`` is replaced with the index of temperature ).
The first column is the index of the MC step.
The second column is the index of the walker.
The third column is the value of the solver.
The remaining columns are the coordinates.

.. code-block::

    # T = 1.0
    0 15 28.70157662892569 3.3139009347685118 -4.20946994566609
    1 15 28.70157662892569 3.3139009347685118 -4.20946994566609
    2 15 28.70157662892569 3.3139009347685118 -4.20946994566609
    3 15 28.98676409223712 3.7442621319489637 -3.868754990884034


Algorithm
********************

Markov chain Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Markov chain Monte Carlo (MCMC) sampling explores the parameter space by moving walkers :math:`\vec{x}` stochastically according to the weight function :math:`W(\vec{x})`.
For the weight function, the Boltzmann factor :math:`W(\vec{x}) = e^{-f(\vec{x})/T}` is generally adopted, where :math:`T>0` is the "temperature."
It is impossible in the many cases, unfortunately, to sample walkers according to :math:`W` directly.
Insteadly, the MCMC method moves walkers slightly and generates a time series :math:`\{\vec{x}_t\}` such that the distribution of the walkers obeys :math:`W` .
Let us call the transision probability from :math:`\vec{x}` to :math:`\vec{x}'` as :math:`p(\vec{x}' | \vec{x})`.
When :math:`p` is determined by the following condition ("the balance condition")

.. math::

  W(\vec{x}') = \sum_{\vec{x}} p(\vec{x}' | \vec{x}) W(\vec{x}),

the distribution of the generated time series :math:`\{\vec{x}_t\}` will converges to :math:`W(\vec{x})` [#mcmc_condition]_.
Practically, the stronger condition ("the detailed balance condition")

.. math::

  p(\vec{x} | \vec{x}') W(\vec{x}') =  W(\vec{x})p(\vec{x}' | \vec{x})


is usually imposed.
The detailed balance condition returns to the balance condition by taking the summation of :math:`\vec{x}`.

2DMAT adopts the Metropolis-Hasting (MH) method for solving the detailed balance condition.
The MH method splits the transition process into the suggestion process and the acceptance process.

1. Generate a candidate :math:`\vec{x}` with the suggestion probability :math:`P(\vec{x} | \vec{x}_t)`.

   - As :math:`P`, use a simple distribution such as the normal distribution with centered at x.

2. Accept the candidate :math:`\vec{x}` with the acceptance probability :math:`Q(\vec{x} | \vec{x}_t)`.

   - If accepted, let :math:`\vec{x}_{t+1}` be `\vec{x}`.
   - Otherwise, let :math:`\vec{x}_{t+1}` be `\vec{x}_t`.

The whole transision probability is the product of these two ones, :math:`p(\vec{x} | \vec{x_t}) = P(\vec{x} | \vec{x}_t) Q(\vec{x} | \vec{x}_t)`.
The acceptance probability :math:`Q(\vec{x} | \vec{x}_t)` is defined as

.. math::

  Q(\vec{x} | \vec{x}_t) = \min\left[1, \frac{W(\vec{x})P(\vec{x}_t | \vec{x}) }{W(\vec{x}_t) P(\vec{x} | \vec{x}_t)} \right].

It is easy to verify that the detailed balance condition is satisfied by substituting it into the detailed balance condition equation.

When adopting the Boltzmann factor for the weight and a symmetry distribution
:math:`P(\vec{x} | \vec{x}_t) = P(\vec{x}_t | \vec{x})` for the suggestion probability,
the acceptance probability :math:`Q` will be the following simple form:

.. math::

  Q(\vec{x} | \vec{x}_t) = \min\left[1, \frac{W(\vec{x})}{W(\vec{x}_t)} \right]
                         = \min\left[1, \exp\left(-\frac{f(\vec{x}) - f(\vec{x}_t)}{T}\right) \right].

By saying :math:`\Delta f = f(\vec{x}) - f(\vec{x}_t)` and using the fact :math:`Q=1` for :math:`\Delta f \le 0`,
the procedure of MCMC with the MH algorithm is the following:

1. Choose a candidate from near the current position and calculate :math:`f` and :math:`\Delta f`.
2. If :math:`\Delta f \le 0`, that is, the walker is descending, accept it.
3. Otherwise, accept it with the probability :math:`Q=e^{-\Delta f/T}`.
4. Repeat 1-3.

The solution is given as the point giving the minimum value of :math:`f(\vec{x})`.
The third process of the above procedure endures that walkers can climb over the hill with a height of :math:`\Delta f \sim T`, the MCMC sampling can escape from local minima.

Replica exchange Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "temperature" :math:`T` is one of the most important hyper parameters in the MCMC sampling.
The MCMC sampling can climb over the hill with a height of :math:`T` but cannot easily escape from the deeper valley than :math:`T`.
It is why we should increase the temperature in order to avoid stuck to local minima.
On the other hand, since walkers cannot see the smaller valleys than :math:`T`, the precision of the obtained result :math:`\min f(\vec{x})` becomes about :math:`T`, and it is necessary to decrease the temperature in order to achieve more precise result.
This dilemma leads us that we should tune the temperature carefully.

One of the ways to overcome this problem is to update temperature too.
For example, simulated annealing decreases temperature as the iteration goes.
Another algorithm, simulated tempering, treats temperature as another parameter to be sampled, not a fixed hyper parameter,
and update temperature after some iterations according to the (detailed) balance condition.
Simulated tempering studies the details of a valley by cooling and escapes from a valley by heating.
Replica exchange Monte Carlo (RXMC), also known as parallel tempering, is a parallelized version of the simulated tempering.
In this algorithm, several copies of a system with different temperature, called as replicas, will be simulated in parallel.
Then, with some interval of steps, each replica exchanges temperature with another one according to the (detailed) balance condition.
As the simulated tempering does, RXMC can observe the details of a valley and escape from it by cooling and heating.
Moreover, because each temperature is assigned to just one replica, the temperature distribution will not be biased.
Using more replicas narrows the temperature interval, and increases the acceptance ratio of the temperature exchange.
This is why this algorithm suits for the massively parallel calculation.

It is recommended that users perform ``minsearch`` optimization starting from the result of ``exchange``, because the RXMC result has uncertainty due to temperature.

.. only:: html

  .. rubric:: footnote

.. [#mcmc_condition] To be precisely, the non-periodicality and the ergodicity are necessary for convergence.
