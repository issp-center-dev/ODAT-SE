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
    
**-n NDATA, --ndata NDATA**
    Specifies the number of data points for each spot as comma-separated integers. This is a required parameter. Examples: "100" (one spot with 100 points), "50,100,75" (three spots with 50, 100, and 75 points respectively)
    
**-w WEIGHT, --weight WEIGHT**
    Specifies the relative weights of spots as comma-separated values. Weights are automatically normalized to sum to 1.0. The number of weight values must match the number of data points. If not specified, equal weights are assigned to all spots.
    
**-V VOLUME, --Volume VOLUME**
    Specifies the normalization of the prior probability distribution (volume of the domain :math:`V_\Omega`). Default is 1.0.
    
**-f RESULT, --result RESULT**
    Specifies the filename for outputting model evidence values. Default is model_evidence.txt.
    
**-o OUTPUT, --output OUTPUT**
    Specifies the filename for the model evidence plot. The output format is determined by the file extension, and any format supported by matplotlib can be specified. Default is model_evidence.png.
    
**-h, --help**
    Displays help message and exits the program.

USAGE
-----

1. Basic usage (one data file and one spot)

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 fx.txt

   Calculates the model evidence for a spot with 100 data points,
   and outputs model_evidence.txt and model_evidence.png.

2. When there are multiple spots

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 50,100,75 -w 0.2,0.5,0.3 fx.txt

   Calculates the model evidence for three spots (with 50, 100, and 75 data points respectively,
   and relative weights of 0.2, 0.5, and 0.3).

3. When using multiple data files

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 -o evidence_plot.pdf -f evidence_data.txt fx_1.txt fx_2.txt fx_3.txt

   Calculates the model evidence from three data files and determines the mean and variance.
   Outputs the results to evidence_data.txt and generates a plot with error bars in evidence_plot.pdf.

Calculation of Model Evidence
-----------------------------

The model evidence P(D|β) is calculated using the following formula:

.. math::

   \log P(D|\beta) = \log Z - \log V + \frac{n}{2} \log \beta + \sum_{\mu} \frac{n_{\mu}}{2} \log w_{\mu} - \frac{n}{2} \log \pi

where:
 * Z: Partition function (result of PAMC calculation)
 * V: Normalization factor of the prior probability distribution
 * n: Total number of data points (sum of all spots)
 * n_μ: Number of data points for each spot
 * w_μ: Relative weight of each spot (normalized to sum to 1)
 * β: Inverse temperature
 * π: Pi (the mathematical constant)

Input File Format
-----------------

The input file (PAMC output file) is expected to have the following format:

.. code-block:: text

   # Comment line (optional)
   beta_value  value2  value3  value4  logz_value  ...
   ...

The script reads the following values from each line:
 * Column 1 (index 0): beta value (inverse temperature)
 * Column 5 (index 4): logz value (logarithm of the partition function)

Output File Format
------------------

The output file (model_evidence.txt) has the following format:

.. code-block:: text

   # max log_P(D;beta) = {maximum_value} at Tstep = {index}, beta = {corresponding_beta_value}
   # $1: Tstep
   # $2: beta
   # $3: model_evidence
   0  beta1  model_evidence1
   1  beta2  model_evidence2
   ...

When processing multiple input files, a variance column is added:

.. code-block:: text

   # max log_P(D;beta) = {maximum_value} at Tstep = {index}, beta = {corresponding_beta_value}
   # $1: Tstep
   # $2: beta
   # $3: average model_evidence
   # $4: variance
   0  beta1  avg_model_evidence1  variance1
   1  beta2  avg_model_evidence2  variance2
   ...

Processing Mechanism
---------------------

This script processes data in the following steps:

1. Reads beta values and logz values from input files
2. Obtains the number of data points and weights for each spot
3. Calculates the logarithm of the model evidence
4. Calculates the mean and variance for multiple files
5. Outputs the results to a file
6. Plots the model evidence as a function of beta

Plot Characteristics
---------------------

* X-axis (beta) is always displayed on a logarithmic scale
* For a single file, only points are displayed; for multiple files, error bars are included
* Markers are displayed as red "x"
* Grid lines are displayed to make it easier to identify data positions

Error Handling
--------------

* If the input file does not exist: A file open error occurs
* If the data format is invalid: An error occurs in numpy.loadtxt
* If the lengths of n_mu and w_mu do not match: An AssertionError occurs

In particular, the number of spots and the number of their weights must always match.
