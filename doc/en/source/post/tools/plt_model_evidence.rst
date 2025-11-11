plt_model_evidence.py
=====================

NAME
----
Calculate model evidence

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_model_evidence.py [OPTION]... -n NDATA FILEs


DESCRIPTION
-----------

Extracts beta and partition function values from PAMC output files (FILE) and calculates the model evidence.
Results are written to standard output and a plot of the results is saved to a file.

When multiple FILEs are specified, the average and variance of their model evidence values are calculated and a plot with error bars is generated.

.. note::
   * Python 3.6 or higher is required (due to the use of f-strings).
   * All calculations are performed on a logarithmic scale for numerical stability.
   * The x-axis (beta) in plots is always displayed on a logarithmic scale.

**FILE**
    PAMC output filename(s) (fx.txt). Multiple files can be specified.
    
**-n NDATA, \-\-ndata NDATA**
    Specifies the number of data points for each dataset as comma-separated integers. This is a required parameter. Examples: "100" (one dataset with 100 points), "50,100,75" (three datasets with 50, 100, and 75 points respectively)
    
**-w WEIGHT, \-\-weight WEIGHT**
    Specifies the relative weights between datasets as comma-separated values. Weights are automatically normalized to sum to 1.0. The number of weight values must match the number of data points.
    
**-V VOLUME, \-\-Volume VOLUME**
    Specifies the normalization of the prior probability distribution (volume of the domain :math:`V_\Omega`). Default is 1.0.
    
**-f RESULT, \-\-result RESULT**
    Specifies the filename for outputting model evidence values. Default is model_evidence.txt.
    
**-o OUTPUT, \-\-output OUTPUT**
    Specifies the filename for the model evidence plot. The output format is determined by the file extension, and any format supported by matplotlib can be specified. Default is model_evidence.png.
    
**--auto-focus**
    Sets a flag that applies a method for determining a suitable plot window for the model evidence data based on the maximum value of of the model evidence. The tightness of the focused window is controlled by the ``--focus-factor`` option argument.
    
**--focus-factor**
    Sets the tightness for the ``--auto-focus`` option. It should be a float between 0 and 1 (lower is tighter), and when ``--auto-focus`` is not set, this argument does nothing. The default value is 0.5.
    
**-h, \-\-help**
    Displays help message and exits the program.

USAGE
-----

1. Basic usage (one data file and one dataset)

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 fx.txt

   Calculates the model evidence for a dataset with 100 data points,
   and outputs model_evidence.txt and model_evidence.png.

2. When there are multiple datasets

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 50,100,75 -w 0.2,0.5,0.3 fx.txt

   Calculates the model evidence for three spots (with 50, 100, and 75 data points respectively,
   and relative weights of 0.2, 0.5, and 0.3).

3. When using multiple data files

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 -o evidence_plot.pdf -f evidence_data.txt fx_1.txt fx_2.txt fx_3.txt

   Calculates the model evidence from three data files and determines the mean and variance.
   Outputs the results to evidence_data.txt and generates a plot with error bars in evidence_plot.pdf.

NOTES
-----

Calculation of Model Evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R-factor is defined as follows:

.. math::

   R(X;D)^2 = \sum_\mu w_\mu \sum_i \left( I_\mu(\theta_i) - I^{\text{(cal)}}_\mu(\theta_i;X) \right)^2

where :math:`I_\mu(\theta_i)` represents the measured data points in dataset :math:`\mu`, and :math:`I^{\text{(cal)}}_\mu(\theta_i;X)` is the theoretical calculated value under parameter :math:`X`.
:math:`w_\mu` is the relative weight of each dataset, normalized so that their sum equals 1.

The model evidence :math:`P(D|\beta)` is calculated using the following formula:

.. math::

   \log P(D|\beta) = \log Z(D;\beta) - \log V_\Omega + \frac{n}{2} \log \beta + \sum_{\mu} \frac{n_{\mu}}{2} \log w_{\mu} - \frac{n}{2} \log \pi

where :math:`Z(D;\beta)` is the partition function:

.. math::

   Z(D;\beta) = \int_\Omega \exp\left(-\beta\,R(X;D)^2\right) dX

and:

 * :math:`V_\Omega`: Normalization factor of the prior probability distribution
 * :math:`n_\mu`: Number of data points in each dataset
 * :math:`n`: Total number of data points (sum of all datasets)
 * :math:`\beta`: Inverse temperature

Input File Format
~~~~~~~~~~~~~~~~~

The input file (PAMC output file) is expected to have the following format:

.. code-block:: text

   # Comment line (optional)
   beta_value  fx_mean  fx_var  nreplica  logz_value  acceptance
   ...

The script reads the following values from each line:
 * Column 1 (index 0): beta value (inverse temperature)
 * Column 5 (index 4): logz value (logarithm of the partition function)

Output File Format
~~~~~~~~~~~~~~~~~~

The output file (model_evidence.txt) has the following format:

.. code-block:: text

   # max log_P(D;beta) = {maximum_value} at Tstep = {index}, beta = {corresponding_beta_value}
   # $1: Tstep
   # $2: beta
   # $3: model_evidence
   0  beta0  model_evidence0
   1  beta1  model_evidence1
   ...

When processing multiple input files, a variance column is added:

.. code-block:: text

   # max log_P(D;beta) = {maximum_value} at Tstep = {index}, beta = {corresponding_beta_value}
   # $1: Tstep
   # $2: beta
   # $3: average model_evidence
   # $4: variance
   0  beta0  avg_model_evidence0  variance0
   1  beta1  avg_model_evidence1  variance1
   ...

Processing Mechanism
~~~~~~~~~~~~~~~~~~~~~

This script processes data in the following steps:

1. Reads beta values and logz values from input files
2. Obtains the number of data points and weights for each dataset
3. Calculates the logarithm of the model evidence
4. Calculates the mean and variance for multiple files
5. Outputs the results to a file
6. Plots the model evidence as a function of beta

Plot Characteristics
~~~~~~~~~~~~~~~~~~~~

* X-axis (beta) is always displayed on a logarithmic scale
* For a single file, only points are displayed; for multiple files, error bars are included
* Markers are displayed as red "x"
* Grid lines are displayed to make it easier to identify data positions

Error Handling
~~~~~~~~~~~~~~

* If the input file does not exist: A file open error occurs
* If the data format is invalid: An error occurs in numpy.loadtxt
* If the lengths of NDATA and WEIGHT do not match: An AssertionError occurs

In particular, the number of data points list and their weights must always match.
