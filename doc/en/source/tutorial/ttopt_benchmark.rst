Tensor train optimization for analytical test functions
=======================================================

This tutorial describes how to use the TTOpt algorithm for minimizing various high-dimensional benchmark functions used in optimization. This example is concerned with the optimization of functions with continuous parameters. A related tutorial describing an application to functions with discrete parameters, please see :doc:`qubo`.

Sample files
~~~~~~~~~~~~

Sample files are available from ``sample/analytical/ttopt_benchmark`` .
This directory includes the following files:

- ``run_benchmarks.py``

  Main program file.

Input files
~~~~~~~~~~~

In this example, the input TOML files are dynamically generated for each test function. This subsection describes the input file template.

.. code-block::

    [base]
    dimension = {dim}
    output_dir = "{output_dir}"

    [solver]
    name = "analytical"
    function_name = "{func_name}"

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

Parameter values of the form ``{...}`` are metavariables that are replaced with the appropriate quantity as specified in ``run_benchmarks.py``. For details, see the manual entries :doc:`../input/index` and :doc:`../algorithm/ttopt`.

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


Calculation
~~~~~~~~~~~

We assume that the current working directory is the directory containing the relevant tutorial files.

.. code-block::

   $ cd sample/analytical/ttopt_benchmark

In this tutorial, we search for the minimum of various benchmark functions in 2 and 10 dimensions. The benchmark minimization is run by executing the following command together with an optional argument that sets the number of MPI processes to use:

.. code-block::

   $ python3 ./run_benchmarks.py 8 | tee log.txt

Here, the calculation is performed using 8 processes communicating over MPI. Each optimization should take a few seconds to complete.

For each benchmark function, a folder of the form ``output/output_{func_name}`` is first created, and all output files for the ODAT-SE run are stored in this folder. Each MPI process is given a subfolder indexed by the process rank. Within each subfolder is a log file containing execution time details for the process. The log file containing the optimization history for the entire optimization process is the ``ttopt_history.txt`` file located in the output folder.

The file ``ttopt_history.txt`` contains a list of parameters set when invoking the optimization method, as well as a record of the number of function evaluations, the best point found so far, and the best objective function value found so far.

.. code-block::

    nprocs = 8
    bounds = [[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]
    p_points = [2 2 2 2 2 2 2 2 2 2]
    q_points = [25 25 25 25 25 25 25 25 25 25]
    r_max = 3.906300918732342e-06
    max_f_eval = 100000
    maxvol_tol = 1.001
    maxvol_max_it = 1000
    f_eval, x_opt, f_opt
    8, [ -5.58609489  10.23980011  25.61853104  22.65200946 -13.55066544
     -10.49954426  28.96521473 -19.50112851 -17.7153394   -9.98021612], 21.450699050623328
    24, [ -5.58609489  10.23980011  25.61853104  22.65200946 -13.55066544
     -10.49954426  28.96521473 -19.50112851 -17.7153394   -9.98021612], 21.450699050623328
    ...

By piping the output to an output file (here, it is ``log.txt``), the optimization results can be examined.

In the following table, for each optimization function, the global minimum is provided together with the estimate of the global minimum.

.. raw:: html

        <style>
        .centered {
                text-align: center;
                vertical-align: middle;
        }
        table {
                width: 60%;
                margin-left: auto;
                margin-right: auto;
                border-collapse: collapse;
        }
        thead {
                border-bottom: 2px solid black;
        }
        </style>
        <table>
                <colgroup>
                        <col>
                        <col>
                        <col style="border-right: 2px solid black;">
                        <col>
                        <col>
                </colgroup>
                <thead>
                        <tr>
                                <td class="centered">Function</td>
                                <td class="centered">Dimension</td>
                                <td class="centered">Bounds</td>
                                <td class="centered">Minimum</td>
                                <td class="centered">Estimated Minimum</td>
                        </tr>
                </thead>
                <tbody>
                        <tr>
                                <td class="centered">Ackley</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-32.768, 32.768]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$3.906\times10^{-6}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Alpine</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-10, 10]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$2.879\times10^{-7}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Exponential</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-1, 1]$$</td>
                                <td class="centered">$$-1$$</td>
                                <td class="centered">$$-1$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Griewank</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-600, 600]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$2.466\times10^{-6}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Himmelblau</td>
                                <td class="centered">$$d=2$$</td>
                                <td class="centered">$$[-6, 6]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$6.312\times10^{-13}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Michalewicz</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[0, \pi]$$</td>
                                <td class="centered">$$-9.66015$$</td>
                                <td class="centered">$$-9.578$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Qing</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-500, 500]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$1.500\times10^{-8}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Rastrigin</td>
                                <td class="centered;">$$d=10$$</td>
                                <td class="centered;">$$[-5.12, 5.12]$$</td>
                                <td class="centered;">$$0$$</td>
                                <td class="centered;">$$4.620\times10^{-11}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Rosenbrock</td>
                                <td class="centered">$$d=2$$</td>
                                <td class="centered">$$[-5, 5]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$0.01175$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Schaffer</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-100, 100]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$3.606\times10^{-2}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Schwefel</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-500, 500]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$1.273\times10^{-4}$$</td>
                        </tr>
                </tbody>
        </table>

.. raw:: latex

        \begin{tabular}{ccc|cc}
        Function & Dimension & Bounds & Minimum & Estimated Minimum \\
        \hline
        Ackley & $10$ & $[-32.768, 32.768]$ & $0$ & $3.906\times10^{-6}$ \\
        Alpine & $10$ & $[-10, 10]$ & $0$ & $2.879\times10^{-7}$ \\
        Exponential & $10$ & $[-1, 1]$ & $-1$ & $-1$ \\
        Griewank & $10$ & $[-600, 600]$ & $0$ & $2.466\times10^{-6}$ \\
        Himmelblau & $2$ & $[-6, 6]$ & $0$ & $6.312\times10^{-13}$ \\
        Michalewicz & $10$ & $[0, \pi]$ & $-9.66015$ & $-9.578$ \\
        Qing & $10$ & $[-500, 500]$ & $0$ & $1.500\times10^{-8}$ \\
        Rastrigin & $10$ & $[-5.12, 5.12]$ & $0$ & $4.620\times10^{-11}$ \\
        Rosenbrock & $2$ & $[-5, 5]$ & $0$ & $0.01175$ \\
        Schaffer & $10$ & $[-100, 100]$ & $0$ & $3.606\times10^{-2}$ \\
        Schwefel & $10$ & $[-500, 500]$ & $0$ & $1.273\times10^{-4}$ \\
        \end{tabular}

It can be seen that TTOpt provides good estimates of the global minimum for all the considered functions.
