Bayesian optimization ``bayes``
*******************************

.. _PHYSBO: https://www.pasums.issp.u-tokyo.ac.jp/physbo/en
.. _PHYSBO-Core-cython: https://github.com/issp-center-dev/PHYSBO-Core-cython

``bayes`` is an ``Algorithm`` that uses Bayesian optimization to perform parameter search.
The implementation is based on `PHYSBO`_.

Preparation
~~~~~~~~~~~~
You will need to install `PHYSBO`_ beforehand.

.. code-block:: bash

   $ python3 -m pip install physbo

If PHYSBO version is 3 or higher, we recommend you to install `PHYSBO-Core-cython`_ too for better performance.

.. code-block:: bash

   $ python3 -m pip install physbo-core-cython

If `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ is installed, MPI parallel computing is possible.

.. _bayes_input:

Input parameters
~~~~~~~~~~~~~~~~~~~~~

[``algorithm.param``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, the search parameter space is defined.

If ``mesh_path`` is defined, it will be read from a mesh file.
In a mesh file, one line gives one point in the parameter space,
the first column is the data number, and the second and subsequent columns are the coordinates of each dimension.

If ``mesh_path`` is not defined, candidate points are automatically generated from the search space defined by ``min_list`` and ``max_list``.
If ``num_list`` is defined, ``num_list`` points are sampled evenly for each parameter.
If ``num_points`` is defined, ``num_points`` points are randomly sampled from the search space.
``num_list`` and ``num_points`` cannot be defined at the same time.

- ``mesh_path``

  Format: String

  Description: The path to a reference file that contains information about the mesh data.

- ``min_list``

  Format: List of float. The length should match the value of dimension.

  Description: The minimum value the parameter can take.

- ``max_list``

  Format: List of float.The length should match the value of dimension.

  Description: The maximum value the parameter can take.

- ``num_list``

  Format: List of integer. The length should match the value of dimension.

  Description: The number of grids the parametar can take at each dimension.

- ``num_points``

  Format: Integer.

  Description: The number of points to be randomly sampled.


[``algorithm.bayes``] section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hyper parameters are defined.

- ``random_max_num_probes``

  Format: Integer (default: 20)

  Description: Number of random samples to be taken before Bayesian optimization (random sampling is needed if parameters and scores are not available at the beginning).

- ``bayes_max_num_probes``

  Format: Integer (default: 40)

  Description: Number of times to perform Bayesian optimization.

- ``score``

  Format: String (default: ``TS`` )

  Description: Parameter to specify the score function.
  ``EI`` (expected improvement), ``PI`` (probability of improvement), and ``TS`` (Thompson sampling) can be chosen.

- ``interval``

  Format: Integer (default: 5)

  Description: The hyperparameters are learned at each specified interval. If a negative value is specified, no hyperparameter learning will be performed.
  If a value of 0 is specified, hyperparameter learning will be performed only in the first step.

- ``num_rand_basis``

  Format: Integer (default: 5000)

  Description: Number of basis functions; if 0 is specified, the normal Gaussian process is performed without using the Bayesian linear model.


Reference file
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh definition file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the grid space to be explored in this file.
The first column is the index of the mesh, and the second and subsequent columns are the values of variables defined in ``string_list`` in the ``[solver.param]`` section.

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

Output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BayesData.txt`` 
^^^^^^^^^^^^^^^^^^^^^^

At each step of the optimization process, the values of the parameters and the corresponding objective functions are listed in the order of the optimal parameters so far and the searched parameters at that step.

.. code-block::

    #step z1 z2 R-factor z1_action z2_action R-factor_action
    0 4.75 4.5 0.05141906746102885 4.75 4.5 0.05141906746102885
    1 4.75 4.5 0.05141906746102885 6.0 4.75 0.06591878368102033
    2 5.5 4.25 0.04380131351780189 5.5 4.25 0.04380131351780189
    3 5.0 4.25 0.02312528177606794 5.0 4.25 0.02312528177606794
    ...


Restart
~~~~~~~~~~~~~~~~~~~~~~
The execution mode is specified by the ``run_mode`` parameter to the constructor.
The operation of each mode is described as follows.
The parameter values correspond to ``--init``, ``--resume``, and ``--cont`` options of ``odatse`` command, respectively.

- ``"initial"`` (default)

  The program is started from the initial state.
  First, it performs the random sampling for the number of times specified by ``random_max_num_probes`` parameter.
  Then, it performs the Bayes optimization for the number of times specified by ``bayes_max_num_probes``.

  If the checkpointing is enabled, the intermediate states will be stored at the folloing occasions:

  #. when the random sampling is finished.
  #. during the Bayesian optimization, the specified number of iteration has been done, or the specified period of time has passed.
  #. at the end of the execution.

- ``"resume"``

  The program execution is resumed from the latest checkpoint.
  The conditions such as the number of MPI processes should be kept the same.

  It is noted that the results obtaind from the resumed run from the interruption and those obtained from the uninterrupted run do not exactly match.

- ``"continue"``

  The program execution of the Bayes optimization is continued from the previous run.
  The value of ``bayes_max_num_probes`` should be increased. The step counter is taken over.

  For example: in the first run, the calculation is carried out for 100 Bayesian optimization steps with ``bayes_max_num_probes=100``. In the next run, the calculation is continued with ``bayes_max_num_probes=200``, where the calculations from 101st step to 200th step are carried out.


Algorithm Description
~~~~~~~~~~~~~~~~~~~~~~

`Bayesian optimization (BO) <https://en.wikipedia.org/wiki/Bayesian_optimization>`_ is an optimization algorithm that uses machine learning as an aid, and is particularly powerful when it takes a long time to evaluate the objective function. 

In BO, the objective function :math:`f(\vec{x})` is approximated by a model function (often a Gaussian process) :math:`g(\vec{x})` that is quick to evaluate and easy to optimize.
The :math:`g` is trained to reproduce well the value of the objective function :math:`\{\vec{x}_i\}_{i=1}^N` at some suitably predetermined points (training data set) :math:`\{f(\vec{x}_i)\}_{i=1}^N`.

At each point in the parameter space, we propose the following candidate points for computation :math:`\vec{x}_{N+1}`, where the expected value of the trained :math:`g(\vec{x})` value and the "score" (acquition function) obtained from the error are optimal.
The training is done by evaluating :math:`f(\vec{x}_{N+1})`, adding it to the training dataset, and retraining :math:`g`.
After repeating these searches, the best value of the objective function as the optimal solution will be returned.

A point that gives a better expected value with a smaller error is likely to be the correct answer,
but it does not contribute much to improving the accuracy of the model function because it is considered to already have enough information.
On the other hand, a point with a large error may not be the correct answer,
but it is a place with little information and is considered to be beneficial for updating the model function.
Selecting the former is called "exploition," while selecting the latter is called "exploration," and it is important to balance both.
The definition of "score" defines how to choose between them.

In ODAT-SE, we use `PHYSBO`_ as a library for Bayesian optimization.
PHYSBO, like ``mapper_mpi``, computes a "score" for a predetermined set of candidate points, and proposes an optimal solution.
MPI parallel execution is possible by dividing the set of candidate points.
In addition, we use a kernel that allows us to evaluate the model function and thus calculate the "score" with a linear amount of computation with respect to the number of training data points :math:`N`.
In PHYSBO, "expected improvement (EI)", "probability of improvement (PI)", and "Thompson sampling (TS)" are available as "score" functions.

