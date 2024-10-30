Nelder-Mead method ``minsearch``
********************************

.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html

When ``minsearch`` is selcted, the optimization by the `Nelder-Mead method <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_ (a.k.a. downhill simplex method) will be done. In the Nelder-Mead method, the dimension of the parameter space is :math:`D`, and the optimal solution is searched by systematically moving pairs of :math:`D+1` coordinate points according to the value of the objective function at each point.

An important hyperparameter is the initial value of the coordinates.
Although it is more stable than the simple steepest descent method, it still has the problem of being trapped in the local optimum solution, so it is recommended to repeat the calculation with different initial values several times to check the results.

In 2DMAT, the Scipy's function ``scipy.optimize.minimize(method="Nelder-Mead")`` is used.
For details, see `the official document <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ .


Preparation
~~~~~~~~~~~

You will need to install `scipy <https://docs.scipy.org/doc/scipy/reference>`_ .::

  python3 -m pip install scipy

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It has subsections ``param`` and ``minimize``.

.. _minsearch_input_param:

[``param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``initial_list``

  Format: List of float. The length should match the value of dimension.

  Description: Initial value of the parameter. If not defined, it will be initialized uniformly and randomly.

- ``unit_list``

  Format: List of float. The length should match the value of dimension.

  Description: Units for each parameter.
        In the search algorithm, each parameter is divided by each of these values to perform a simple dimensionless and normalization.
        If not defined, the value is 1.0 for all dimensions.
	
  - ``min_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Minimum value of each parameter.
                 When a parameter falls below this value during the Nelson-Mead method,
                 the solver is not evaluated and the value is considered infinite.


  - ``max_list``

    Format: List of float. Length should be equal to ``dimension``.

    Description: Maximum value of each parameter.
                 When a parameter exceeds this value during the Nelson-Mead method,
                 the solver is not evaluated and the value is considered infinite.

[``minimize``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the hyperparameters for the Nelder-Mead method.
See the documentation of `scipy.optimize.minimize`_ for details.

- ``initial_scale_list``

  Format: List of float. The length should match the value of dimension. 

  Description: The difference value that is shifted from the initial value in order to create the initial simplex for the Nelder-Mead method.
  The ``initial_simplex`` is given by the sum of ``initial_list`` and the dimension of the ``initial_list`` plus one component of the ``initial_scale_list``.
  If not defined, scales at each dimension are set to 0.25.

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
