================================
Choosing an Algorithm
================================

Which algorithm should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose based on the number of parameters and the nature of your objective function.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Algorithm
     - Suitable for
     - Characteristics
   * - ``minsearch``
     - Few parameters (~10), quick local minimum search
     - Nelder-Mead method. Fast but may get trapped in local minima. Gradient-free
   * - ``global_search``
     - Global optimization of continuous parameters
     - Global optimizers from scipy.optimize (differential evolution, SHGO, DIRECT, dual annealing). Requires scipy. Supports MPI-parallel candidate evaluation
   * - ``bayes``
     - Expensive objective function, minimize evaluation count
     - Surrogate model via Gaussian process regression. Minimizes evaluations. Requires physbo
   * - ``mapper``
     - Overview of parameter space. Few parameters (2-3)
     - Evaluates all grid points. Computational cost grows explosively with the number of parameters
   * - ``exchange``
     - Multimodal objective function, broad search
     - Replica exchange method. Avoids local minima. MPI parallelism recommended (requires mpi4py; single-process runs also work)
   * - ``pamc``
     - Posterior distribution estimation, model evidence calculation
     - Population annealing. Suitable for statistical estimation
   * - ``random_search``
     - Quick overview of parameter space
     - Random sampling. Simple and robust
   * - ``ttopt``
     - Finding minima in problems with relatively many parameters or with discrete variables
     - Tensor train optimization. Gradient-free cross approximation. Supports MPI
