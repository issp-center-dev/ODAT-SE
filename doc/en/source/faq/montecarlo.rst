================================
Monte Carlo Tuning
================================

Where can I check the acceptance ratio?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``exchange`` and ``pamc``, the acceptance ratio can be checked as follows:

**Standard output during execution:**

``pamc`` displays the acceptance ratio at each temperature step.

.. code-block:: text

    # beta  mean[f]  Err[f]  nreplica  log(Z/Z0)  acceptance_ratio

**Output file ``fx.txt``:**

Column 6 of ``output/fx.txt`` contains the acceptance ratio.

.. code-block:: text

    # $1: beta (= 1/T)
    # $2: mean of f(x)
    # $3: std err of f(x)
    # $4: number of replicas
    # $5: log(Z/Z0)
    # $6: acceptance ratio


What is an appropriate acceptance ratio?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General guidelines:

- **Too high (> 0.8)**: Step size is too small. Exploration of parameter space is slow.
- **Too low (< 0.1)**: Step size is too large. Most proposals are rejected and the search stagnates.
- **Appropriate range**: Around 0.2-0.5 is generally efficient.

The acceptance ratio varies with temperature (inverse temperature :math:`\beta`). It is normal for it to be higher at high temperatures (low :math:`\beta`) and lower at low temperatures (high :math:`\beta`).


What should I do if the acceptance ratio is too low?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the following remedies:

1. **Reduce the step size**

   Adjust the step size in the ``[algorithm.param]`` section using ``step_list``.

   .. code-block:: toml

       [algorithm.param]
       step_list = [0.1, 0.1, 0.1]

   Smaller values increase the acceptance ratio. Set an appropriate scale relative to the search range of each parameter.

2. **Increase the number of temperature points**

   For ``exchange``, if the temperature difference between replicas is too large, exchanges become unlikely. Increase the number of temperature points to reduce the temperature interval.

   For ``pamc``, increasing ``Tnum`` (number of temperature points) similarly makes the temperature change at each step more gradual.

3. **Increase the number of MCMC steps**

   If equilibration at each temperature is insufficient, increase the number of MCMC steps per temperature.


What should I do if the acceptance ratio is too high?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increase the ``step_list`` values to make larger moves per step.
This commonly occurs when the step size is too small relative to the search range (``max_list`` - ``min_list``).


Monte Carlo results differ each run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monte Carlo methods are stochastic, so results differ with different random seeds.
To reproduce results, fix the ``seed`` in the ``[algorithm]`` section.

.. code-block:: toml

    [algorithm]
    seed = 12345

With sufficient steps, statistically equivalent results will be obtained.


Optimization does not converge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the following:

1. **Search range**: Verify that ``min_list`` / ``max_list`` include the optimal solution.

2. **Initial values**: For ``minsearch``, convergence is difficult if ``initial_list`` is extremely far from the optimal solution.

3. **Algorithm choice**: Using ``minsearch`` on problems with many local minima tends to get trapped. Try ``exchange`` or ``bayes`` which can perform global search.

4. **Insufficient steps**: Monte Carlo methods need enough steps for thorough exploration. Increase the number of steps and re-run.
