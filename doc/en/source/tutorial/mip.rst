Tensor train optimization for MIP applications
==============================================

This tutorial describes how to use the TTOpt algorithm when solving optimization problems formulated as mixed integer programs (MIP). This example is concerned with the optimization of functions with both continuous and discrete parameters. For a tutorial describing an application to functions with only discrete or continuous parameters, please see :doc:`ttopt` or :doc:`qubo`, respectively.

Here, we demonstrate the solution of three MIP problem instances taken from the MINLP benchmark library (MINLPLib) [1]. These instances are briefly described as follows:

- ``alan``: a portfolio optimization problem with 4 continuous variables and 4 binary variables in its prescribed formulation.

- ``synthes2``: an optimization problem in process synthesis with 6 continuous variables and 5 binary variables in its prescribed formulation.

- ``cvxnonsep_normcon20``: an example of a convex non-separable problem with 10 continuous variables and 10 binary variables in its prescribed formulation.

These problems were selected for their relative small problem size, their broad applicabililty, and the availability of their global optima as reported in the MINLPLib database .

Sample files
~~~~~~~~~~~~

Sample files are available from ``sample/mip/ttopt`` .
This directory includes the following files:

- ``alan.py``

  Implementation of a solver for the ``alan`` problem instance.

- ``synthes2.py``

  Implementation of a solver for the ``synthes2`` problem instance.

- ``cvxnonsep_normcon20.py``

  Implementation of a solver for the ``cvxnonsep_normcon20`` problem instance.

Input files
~~~~~~~~~~~

For all examples, the input TOML files are of a similar form. This subsection describes the input file template.

.. code-block::

    [base]
    dimension = {dim}
    output_dir = "{output_dir}"

    [solver]
    name = "custom"

    [algorithm]
    name = "ttopt"
    seed = 12345

    [algorithm.param]
    max_list = {max_list}
    min_list = {min_list}

    [algorithm.ttopt]
    p_points = {p_points}
    q_points = {q_points}
    r_max = 4
    max_f_eval = 100000
    init_points = {init_points}

Parameter values of the form ``{...}`` are problem-specific.

The ``[base]`` section specifies some global parameter values used in the ODAT-SE run:

- ``dimension`` denotes the number of dimensions in the optimization. This corresponds to the number of scalar parameters needed to specify a point in the state space.

- ``output_dir`` is the location where the results of the ODAT-SE run will be saved to.

The ``[algorithm]`` section specifies general parameters for the algorithm.

- ``name`` is the name of the algorithm (``ttopt`` in this case)

- ``seed`` is the initial seed used in the PRNG.

The ``[algorithm.param]`` section sets the parameter space to be explored.

- ``max_list`` is a list of upper bounds along each dimension in the optimization.

- ``min_list`` is a list of lower bounds along each dimension in the optimization.

The ``[algorithm.ttopt]`` section specifies parameters specific to the TTOpt algorithm.

- ``p_points`` is the number of modes for each dimension, which corresponds to the bond dimension along the tensor legs allotted for the parameter. Together with the ``q_points`` parameter, this is related to the number of possible values each parameter can take. The total number of possible values for a parameter is :math:`p^q`.

- ``q_points`` is the number of submodes for each dimension, which corresponds to the number of tensor legs allotted for the parameter. Together with the ``p_points`` parameter, this is related to the number of possible values each parameter can take. The total number of possible values for a parameter is :math:`p^q`.

- ``r_max`` is the maximum rank of the implicit MPS representation used to model the state space.

- ``max_f_eval`` is the maximum number of function evaluations used in the optimization process.

- ``init_points`` is a list of initial points used in the optimization process, which is updated as the optimization proceeds to the next stage.

Calculation
~~~~~~~~~~~

We assume that the current working directory is the directory containing the relevant tutorial files.

.. code-block::

   $ cd sample/mip/ttopt

The instances that we consider in this tutorial are formulated as constrained optimization problems. However, the ``ttopt`` algorithm does not support constraints directly, so we use the penalty method to enforce inequality constraints. The ``ttopt`` algorithm is limited in the sense that it struggles with equality constraints (since equality constraints implicitly reduce the dimension of the problem, but the optimizer does not know about this reduction), so we consider instances where the equality constraints (if present) can be eliminated algebraically.

The penalty method
------------------

The penalty method introduces penalty terms to the objective function that penalize constraint violations. Thus, it only approximately enforces the constraints according to the magnitude of a parameter. The penalty parameter controls the strength of the penalty term, and it is increased iteratively until the constraints are satisfied to within a desired tolerance. The output of the optimization at each stage is used as a starting point for the next stage. Essentially, we transform the following constrained optimization problem with :math:`M` equality constraints and :math:`N` inequality constraints:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   g_i(\mathbf{x})=0, i\in\{1,\ldots,M\} \\
   h_j(\mathbf{x})\leq0, j\in\{1,\ldots,N\} \\
   \end{gather*}

into the following unconstrained problem:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})+p\left(\sum_{i=1}^Mc_{eq}(g_i(\mathbf{x}))+\sum_{j=1}^Nc_{ineq}(h_j(\mathbf{x}))\right)

where :math:`p` is the penalty parameter. The functions :math:`c_{eq}` and :math:`c_{ineq}` are the penalty functions for equality and inequality constraints, respectively. We use the quadratic penalty functions:

.. math::
   :nowrap:

   \begin{gather*}
   c_{eq}(g_i(\mathbf{x}))=g_i(\mathbf{x})^2 \\
   c_{ineq}(h_j(\mathbf{x}))=\max(0,h_j(\mathbf{x}))^2
   \end{gather*}

Since the ``ttopt`` algorithm can support multiple initial points, we keep track of all candidate minima found so far and use them as initial points for the next stage.

Note that there are other approaches to transform a constrained optimization into an unconstrained optimization, such as methods involving Lagrange multipliers and the barrier method. However, introducing Lagrange multipliers effectively increases the dimensionality of the problem, which may not be desired for a derivative-free optimization algorithm. Also, since the ``ttopt`` algorithm performs better with some notion of smoothness in the objective, the barrier method may not be a good choice since the objective function blows up outside the feasible region.

Solving the ``alan`` instance
-----------------------------

To run the solver for the ``alan`` instance, use the following command:

.. code-block::

   $ export OMP_NUM_THREADS=1; mpiexec -np 8 python3 ./alan.py | tee log_alan.txt

Here, the optimization is performed using 8 processes communicating over MPI. The log file ``log_alan.txt`` contains the output of the solver.

The ``alan`` instance is an example of a portfolio optmization problem with 8 variables (4 continuous and 4 binary). It is formulated in the following manner:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   f(\mathbf{x}) = x_0(4x_0+3x_1-x_2)+x_1(3x_0+6x_1+x_2)+x_2(x_1-x_0+10x_2) \\
   x_0+x_1+x_2+x_3=1 \\
   8x_0+9x_1+12x_2+7x_3=10 \\
   b_4+b_5+b_6+b_7\leq3 \\
   x_0-b_4\leq0 \\
   x_1-b_5\leq0 \\
   x_2-b_6\leq0 \\
   x_3-b_7\leq0 \\
   x_0,x_1,x_2,x_3\in[0,1] \\
   b_4,b_5,b_6,b_7\in\{0,1\}
   \end{gather*}

Here, the variables :math:`b_4,b_5,b_6,b_7` are binary variables representing whether or not a security is included in the portfolio, while the variables :math:`x_0,x_1,x_2,x_3` are continuous variables representing the proportion (as shown through the first equality constraint) of each security in the portfolio. The second equality constraint is a budget constraint, and the first inequality constraints are capacity constraints on the number of types of securities included in the portfolio.

Since the original formulation contains two linear equality constraints, we can eliminate them by introducing intermediate variables :math:`y_0,y_1` and using the following formulation of the problem:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   f(\mathbf{x}) = x_0(4x_0+3x_1-y_0)+x_1(3x_0+6x_1+y_0)+y_0(x_1-x_0+10y_0) \\
   b_2+b_3+b_4+b_5\leq3 \\
   x_0-b_2\leq0 \\
   x_1-b_3\leq0 \\
   y_0-b_4\leq0 \\
   y_1-b_5\leq0 \\
   x_0,x_1\in[0,1] \\
   b_2,b_3,b_4,b_5\in\{0,1\} \\
   y_0=\frac{3-x_0-x_1}{5} \\
   y_1=1-x_0-x_1-y_0
   \end{gather*}

This form reduces the MIP to a problem with 2 continuous variables :math:`x_0,x_1` and 4 binary variables :math:`b_2,b_3,b_4,b_5`. Furthermore, all constraints are expressed as inequality constraints that can be handled by the penalty method using a quadratic penalty function. To encode the reduced problem into the ``ttopt`` optimizer, we use the following parameters:

- ``dimension = 6``
- ``min_list = [0, 0, 0, 0, 0, 0]``
- ``max_list = [1, 1, 1, 1, 1, 1]``
- ``p_points = [2, 2, 2, 2, 2, 2]``
- ``q_points = [20, 20, 1, 1, 1, 1]``
- ``init_points = []``

This formulation treats the first two variables as continuous variables (discretized into :math:`2^{20}` points) and the remaining variables as binary variables. As this is a multi-stage optmization, the initial penalty is set to :math:`10` and the initial list of points is initialized to be empty. The penalty function is doubled at each stage, for a total of 20 steps. In practice, drastically increasing the penalty function can lead to an unstable optimization due to the ill-conditioning of the problem.

The obtained minimum for the objective function is :math:`f(x^*)\approx2.9250` at :math:`x^*\approx(0.3750,0,1,0,1,1)` (rounded to 4 decimal places). The known global minimum is :math:`f(x^*)=2.925` at :math:`x^*=(0.375,0,1,0,1,1)`, which matches the obtained minimum.

Solving the ``synthes2`` instance
---------------------------------

Next, to run the solver for the ``synthes2`` instance, use the following command:

.. code-block::

   $ export OMP_NUM_THREADS=1; mpiexec -np 8 python3 ./synthes2.py | tee log_synthes2.txt

Here, the optimization is performed using 8 processes communicating over MPI. The log file ``log_synthes2.txt`` contains the output of the solver.

The ``synthes2`` instance is an example of a process synthesis problem with 11 variables (6 continuous and 5 binary). It is formulated in the following manner:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   f(\mathbf{x}) = \exp(x_0)-10x_0+\exp(\frac{5x_1}{6})-15x_1-60\log(1+x_3+x_4)+15x_3+5x_4 \\
   -15x_2-20x_5+5b_6+8b_7+6b_8+10b_9+6b_{10}+140 \\
   -\log(1+x_3+x_4)\leq0 \\
   \exp(x_0)-10b_6\leq1 \\
   \exp(\frac{5x_1}{6})-10b_7\leq1 \\
   \frac{5x_2}{4}-10b_8\leq0 \\
   x_3+x_4-10b_9\leq0 \\
   -2x_2+2x_5-10b_{10}\leq0 \\
   -x_0-x_1-2x_2+x_3+2x_5\leq0 \\
   -x_0-x_1-\frac{3x_2}{4}+x_3+2x_5\leq0 \\
   x_2-x_5\leq0 \\
   2x_2-x_3-2x_5\leq0 \\
   -\frac{x_3}{2}+x_4\leq0 \\
   -\frac{x_3}{5}-x{4}\leq0 \\
   b_6+b_7=1 \\
   b_9+b_{10}\leq1 \\
   x_3,x_4\in[0,1] \\
   x_0,x_1,x_2\in[0,2] \\
   x_5\in[0,3] \\
   b_6,b_7,b_8,b_9,b_{10}\in\{0,1\}
   \end{gather*}

In this problem, the variables :math:`b_6,b_7,b_8,b_9,b_{10}` are binary variables, while the variables :math:`x_0,x_1,x_2,x_3,x_4,x_5` are continuous variables.

While the problem contains a large number of constraints, it can be reformulated to eliminate the equality constraint and the redundant inequality constraints, of which there are a few.

Since the variables are restricted to be positive, the first and twelfth constraints can be safely eliminiated because they are always true. The seventh constraint is implied by the eighth constraint, and similarly, the tenth constraint is implied by the ninth constraint. The equality constraint allows us to introduce an intermediate variable :math:`c_0`.

The reduced optimization problem can be expressed as follows:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   f(\mathbf{x}) = \exp(x_0)-10x_0+\exp(\frac{5x_1}{6})-25x_1-60\log(1+x_3+x_4)+15x_3+5x_4 \\
   -15x_2-20x_5+5b_6+8c_0+6b_7+10b_8+6b_9+140 \\
   \exp(x_0)-10b_6\leq1 \\
   \exp(\frac{5x_1}{6})-10b_7\leq1 \\
   \frac{5x_2}{4}-10b_8\leq0 \\
   x_3+x_4-10b_9\leq0 \\
   -2x_2+2x_5-10b_{10}\leq0 \\
   -x_0-x_1-\frac{3x_2}{4}+x_3+2x_5\leq0 \\
   x_2-x_5\leq0 \\
   -\frac{x_3}{2}+x_4\leq0 \\
   b_9+b_{10}\leq1 \\
   x_3,x_4\in[0,1] \\
   x_0,x_1,x_2\in[0,2] \\
   x_5\in[0,3] \\
   b_6,b_7,b_8,b_9\in\{0,1\} \\
   c_0=1-b_6
   \end{gather*}

Now, the MIP is written as a 10-dimensional problem with 6 continuous variables and 4 discrete variables. The number of inequality constraints has also decreased.

As before, all constraints are expressed as inequality constraints that can be handled by the penalty method using a quadratic penalty function. To encode the reduced problem into the ``ttopt`` optimizer, we use the following parameters:

- ``dimension = 10``
- ``min_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
- ``max_list = [2, 2, 2, 1, 1, 3, 1, 1, 1, 1]``
- ``p_points = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]``
- ``q_points = [20, 20, 20, 20, 20, 20, 1, 1, 1, 1]``
- ``init_points = []``

The first four dimensions are chosen to represent continuous variables (discretized into :math:`2^{20}` points) and the remaining dimensions represent the binary variables. The initial penalty is set to :math:`10` and the initial list of points is initialized to be empty. We apply the penalty method for 20 steps, doubling the penalty parameter each time.

The obtained minimum for the objective function is: 

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)\approx73.0422 \\
   x^*\approx(0,2,1.0625,0.67188,0.33594,1.0625,0,1,1,0)
   \end{gather*}

The known global minimum is:

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)\approx73.0353 \\
   x^*\approx(0,2,1.0784,0.65201,0.32601,1.0784,0,1,1,0)
   \end{gather*}

This is very close to the obtained minimum.

Solving the ``cvxnonsep_normcon20`` instance
--------------------------------------------

Finally, to run the solver for the ``cvxnonsep_normcon20`` instance, use the following command:

.. code-block::

   $ export OMP_NUM_THREADS=1; mpiexec -np 8 python3 ./cvxnonsep_normcon20.py | tee log_cvxnonsep_normcon20.txt

Here, the optimization is performed using 8 processes communicating over MPI. The log file ``log_cvxnonsep_normcon20.txt`` contains the output of the solver.

The ``cvxnonsep_normcon20`` instance is a synthetic test function with 20 variables (10 continuous and 10 integer). It is formulated in the following manner:

.. math::

   \min_{\mathbf{x}} f(\mathbf{x})

subject to the following constraints:

.. math::
   :nowrap:

   \begin{gather*}
   f(\mathbf{x}) = -0.175i_0-0.39i_1-0.83i_2-0.805i_3-0.06i_4-0.4i_5-0.52i_6-0.415i_7-0.655i_8-0.63i_9 \\
   -0.29x_{10}-0.43x_{11}-0.015x_{12}-0.985x_{13}-0.165x_{14}-0.105x_{15}-0.37x_{16}-0.2x_{17}-0.49x_{18}-0.34x_{19} \\
   \sqrt{0.0001+\sum_{k=0}^9i_k^2+\sum_{k=10}^{19}x_k^2}\le10 \\
   x_{10},x_{11},x_{12},x_{13},x_{14},x_{15},x_{16},x_{17},x_{18},x_{19}\in[0,5] \\
   i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7,i_8,i_9,\in\{0,1,2,3,4,5\}
   \end{gather*}

In this problem, the variables :math:`i_0,\cdots,i_9` are integer variables, while the variables :math:`x_{10},\cdots,x_{19}` are continuous variables.

This problem contains a large number of variables. In its current form, it is nonseparable, but a power transform can be used to make the constraint separable. In this example, we consider the prescribed form to illustrate the versatility of the approach.

To encode the reduced problem into the ``ttopt`` optimizer, we use the following parameters:

- ``dimension = 20``
- ``min_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
- ``max_list = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]``
- ``p_points = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]``
- ``q_points = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]``
- ``init_points = []``

The first ten dimensions are chosen to represent the integer variables and the remaining dimensions represent the continuous variables (discretized into :math:`2^{20}` points as before). The initial penalty is set to :math:`1` and the initial list of points is initialized to be empty. Like in previous examples, we apply the penalty method for 20 steps, doubling the penalty parameter each time.

The obtained minimum for the objective function is:

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)=-21.438 \\
   x^*=(1,2,5,4,0,2,2,2,3,3,1.0499,1.5595,0.055924,3.6254,0.61767,0.38695,1.2500,0.62500,1.7920,1.2743)
   \end{gather*}

However, the known global minimum is:

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)\approx-21.749 \\
   x^*=(1,2,4,4,0,2,2,2,3,3,1.2382,1.8359,0.064043,4.2055,0.70447,0.44830,1.5797,0.85391,2.0921,1.4516)
   \end{gather*}

which is close to the obtained minimum, but the two do not match since a discrete variable differs.

The obtained minimum is a good candidate optimum, but we can supply an initial point assuming that the optimization problem has been partially solved through some other means. We then supply the following initial point in the input file (which corresponds to knowledge of the optimal values that the discrete variables can take): 

- ``init_points = [[1, 2, 4, 4, 0, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]``

If we run the solver again, we will obtain an estimated minimum of:

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)\approx-21.749 \\
   x^*\approx(1,2,4,4,0,2,2,2,3,3,1.2525,1.8542,0.043941,4.1973,0.69840,0.44467,1.5870,0.85935,2.0702,1.4647)
   \end{gather*}

This can be compared with the known global minimum:

.. math::
   :nowrap:

   \begin{gather*}
   f(x^*)\approx-21.749 \\
   x^*\approx(1,2,4,4,0,2,2,2,3,3,1.2382,1.8359,0.064043,4.2055,0.70447,0.44830,1.5797,0.85391,2.0921,1.4516)
   \end{gather*}

Now, the obtained minimum lines up with the true global minimum.