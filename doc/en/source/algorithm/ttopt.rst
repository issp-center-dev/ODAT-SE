Tensor Train Optimization ``ttopt``
***********************************

``ttopt`` is an ``Algorithm`` that performs parameter search using a method based on modeling the cost function as a large tensor indexed by the parameters and considering an implicit tensor train decomposition of the large tensor.

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

  Format: List of integer

  Description: Mode count along each dimension. In the implicit tensor train representation, each parameter corresponds to a dimension, which is represented by :math:`q` open legs in the tensor train with bond dimension :math:`p`. The interval is discretized into :math:`p^q` uniformly spaced points. In practice, for approximaions of continuous intervals, it is advisable to set :math:`p` to a small value like 2, adjusting :math:`q` appropriately.

- ``q_points``

  Format: List of integer

  Description: Submode count along each dimension. Without mode compression, :math:`q` is set to 1 across dimensions. A bond dimension whose original size would be :math:`p^q` can be compressed by representing it as :math:`q` legs with bond dimension :math:`p`. The interval for the parameter is discretized into :math:`p^q` points. In practice, for approximaions of continuous intervals, it is advisable to set :math:`p` to a small value like 2, adjusting :math:`q` appropriately.

- ``r_max``

  Format: Integer (default: 4)

  Description: Maximum rank used in the implicit tensor train approximation. Larger values result in more function evaluations at each step of the optimization.

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

Output Files
~~~~~~~~~~~~

``ttopt_history.txt`` 
^^^^^^^^^^^^^^^^^^^^^

After each sweep of the optimization process (i.e. traversal across each MPS tensor in one direction), the best estimate and the best combination of parameters are recorded. The following is a sample output file:

.. code-block::

    nprocs = 8
    bounds = [[0.0, 157.5], [0.0, 157.5], [0.0, 157.5], [0.0, 157.5], [0.0, 157.5], [0.0, 157.5], [0.0, 157.5], [0.0, 157.5]]
    p_points = [8 8 8 8 8 8 8 8]
    q_points = [1 1 1 1 1 1 1 1]
    r_max = -16.09442984075178
    max_f_eval = 1000000
    maxvol_tol = 1.001
    maxvol_max_it = 1000
    f_eval, x_opt, f_opt
    1024, [112.5  90.    0.    0.  157.5 112.5   0.   90. ], -14.562713869686608
    9216, [ 90.   22.5   0.   90.  112.5   0.   22.5  90. ], -15.638476150205186
    ...

Algorithm Description
~~~~~~~~~~~~~~~~~~~~~

`Tensor Train Optimization <https://arxiv.org/abs/2205.00293>`_ [1] (TTOpt) is a method for finding the optimum of a discrete function and its location in the parameter space. It can be easily adapted to continuous optimization problems and is able to handle high-dimensional problems and functions whose parameters are a combination of discrete and continuous quantities.

TTOpt is a gradient-free optimization scheme based on repeated cross-approximation of a large tensor whose elements correspond to values of the cost function :math:`f(x_1, x_2, ..., x_n)` indexed by parameter combinations :math:`(x_1, x_2, ..., x_n)`. The cross-approximation is computed using approximate maximum-volume submatrices.

The TTOpt algorithm assumes that the large high-dimensional tensor representing the parameter space can be decomposed into a matrix product state (MPS) form. The open legs of the MPS are associated with parameter combinations in the space. The algorithm is designed such that only a small part of the whole large tensor needs to be explicitly computer. Thus, this approach is advantageous when the cost function is computationally costly or when the search space is very large. Furthermore, by representing the data in MPS form (up to a maximum bond dimensions along the internal legs), we can avoid having to form exponentially large matrices in the optimization process.

A detailed description of the algorithm can be found in [1]. Broadly speaking, the algorithm works by considering unfoldings along axes of the large tensor and sampling at points corresponding to elements of maximum-volume submatrices. While the algorithm assumes a discretized interval for each parameter, continuous function optimization can be approximated by using a sufficiently fine discretization of the interval.

For this algorithm, execution using multiple MPI processes is possible. When MPI is used, evaluation of the cost function values at the sampled points is divided across the different ranks.

References
^^^^^^^^^^

[1] K. Sozykin et al., arXiv:2205.00293 (2022).