Tutorial
========================================

This tutorial explains the workflow for analyzing PAMC calculation results using a concrete example.
For detailed options and output formats of each tool, see :doc:`tools/index`.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~

- Python 3.9 or later
- matplotlib (required for histogram and model evidence plots)
- Post-processing scripts are included in the ``script/`` directory of ODAT-SE

Workflow Overview
~~~~~~~~~~~~~~~~~~~~~~~~~

The overall workflow for analyzing PAMC calculation results is as follows:

1. **Run PAMC calculation** to obtain MCMC logs and partition function values
2. **Calculate model evidence** to identify the optimal inverse temperature :math:`\beta`
3. **Aggregate data by temperature point** to collect replica configurations at each temperature
4. **Create histograms** to visualize posterior probability distributions

The output of each step serves as input for the next step.

.. code-block:: text

   PAMC calculation
     ├─ output/fx.txt ──────────────→ (2) model evidence calculation
     └─ output/{rank}/result_T*.txt ─→ (3) aggregate by temperature
                                           └─ summarized/ ─→ (4) histogram creation


1. Running PAMC Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, we use a calculation from the TRHEPD forward problem solver (odatse-STR).
The parameter space is 3-dimensional, with 51 temperature points logarithmically spaced from T=1.0 to 1.0e-6.
Each annealing step consists of 20 MCMC steps.
The number of replicas is set to 100 per process with 4 MPI processes.

Results are output under the output directory.
The main output files are the following two types.

**output/{rank}/result_T{index}.txt** -- MCMC calculation log (per temperature point)

.. code-block:: text

   # step  replica_id  T  fx  x1  x2  x3
   0  0  1.000000e+00  1.234567e+01  4.500  3.200  5.100
   1  0  1.000000e+00  1.198765e+01  4.520  3.180  5.080
   ...

Each row corresponds to one MCMC step, recording the temperature T, objective function value fx, and parameter values x1--x3.

**output/fx.txt** -- Partition function and f(x) statistics

.. code-block:: text

   # beta  fx_mean  fx_var  nreplica  logZ/Z0  acceptance
   1.000000e+00  1.234e+01  5.678e+00  400  0.000000e+00  0.850
   ...

Each row corresponds to a temperature point, recording the inverse temperature beta, mean and variance of f(x), number of replicas, log ratio of partition functions, and acceptance rate.

.. note::

   If export_combined_files is set to True, logs are consolidated in combined.txt.
   Use :doc:`tools/extract_combined` to extract result.txt.

   .. code-block:: bash

      python3 extract_combined.py -t result.txt -d output

.. note::

   If separate_T is False, logs are output to result.txt.
   Use :doc:`tools/separateT` to split into files by temperature point.

   .. code-block:: bash

      python3 separateT.py -d output


2. Calculating Model Evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model evidence :math:`\log P(D;\beta)` is expressed as:

.. math::

   \log P(D;\beta) = \log\left(\dfrac{Z_\beta}{Z_{\beta_0}}\right) - \log V_\Omega + \sum_\mu \dfrac{n_\mu}{2}\log\left(\dfrac{\beta w_\mu}{\pi}\right)

Calculate model evidence using the partition function values :math:`\log Z/Z_0` from output/fx.txt. This requires specifying the search space volume :math:`V_\Omega` (normalization factor for prior probability) and the number of data points :math:`n`.

In this example, the search space spans [3.0, 6.0] for each of z1, z2, z3. The number of data points (rows in experiment.txt) is 70.

.. code-block:: bash

   python3 plt_model_evidence.py -V 27.0 -n 70 output/fx.txt

Model evidence values are written to model_evidence.txt, and a plot against beta is output to model_evidence.png.
For detailed options, see :doc:`tools/plt_model_evidence`.

.. figure:: ../../../common/img/post/model_evidence.*

   Plot of model evidence. Maximum value occurs at beta= :math:`1.91\times 10^5` (Tstep=44).

The :math:`\beta` that maximizes the model evidence corresponds to the inverse temperature at which the model best explains the data.
When :math:`\beta` is too small, the prior distribution dominates (underfitting); when too large, the model fits noise in the data (overfitting).
By visualizing the posterior distribution at the optimal :math:`\beta`, you can evaluate the parameter estimation results.


3. Summarizing Search Data by Temperature Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract and combine replica configurations at the end of annealing from MCMC step information in output/{rank}/result_T{index}.txt.

.. code-block:: bash

   python3 summarize_each_T.py -d output -o summarized

Results are written to summarized/result_T{index}_summarized.txt.
For detailed options, see :doc:`tools/summarize_each_T`.


4. Creating 1D and 2D Marginalized Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot weighted posterior probability distributions :math:`P(z_i|D;\beta) = \dfrac{P(D|z_i\beta) P(z_i)}{P(D;\beta)}` using replica configuration data.

Focus on temperature points near the optimal :math:`\beta` identified in Step 2 to examine the parameter distributions.

To create 1D histograms marginalized along each :math:`z_i`:

.. code-block:: bash

   python3 plt_1D_histogram.py -d summarized -o 1dhist -r 3.0,6.0

This creates histograms for each data file in summarized/, with output to 1dhist/.
For detailed options, see :doc:`tools/plt_1D_histogram`.

.. figure:: ../../../common/img/post/1Dhistogram_result_T22.*

   Example 1D marginalized histogram output (Tstep=22, :math:`\beta=4.365\times 10^2`).


To create 2D marginalized histograms:

.. code-block:: bash

   python3 plt_2D_histogram.py -d summarized -o 2dhist -r 3.0,6.0

This creates 2D histograms for combinations (z1,z2), (z1,z3), (z2,z3), with output to 2dhist/.
The 2D histograms allow you to examine correlations between parameters.
For detailed options, see :doc:`tools/plt_2D_histogram`.

.. figure:: ../../../common/img/post/2Dhistogram_result_T22_x1_vs_x2.*

   Example 2D marginalized histogram output (Tstep=22, z1-z2 axis plot).
