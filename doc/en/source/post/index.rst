.. post-tools documentation master file, created by
   sphinx-quickstart on Wed Mar  5 21:21:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Post-Processing Tools
========================

This section explains the post-processing tools and analysis workflow for PAMC calculations.

The following tools are available in the script directory.
The next section introduces the analysis workflow using examples.
For detailed information about individual tools, please refer to their respective reference sections.

:doc:`tools/extract_combined`
    Extracts specific items from log files output in combined format.

:doc:`tools/plt_1D_histogram`
    Creates marginalized 1D histograms.

:doc:`tools/plt_2D_histogram`
    Creates marginalized 2D histograms.

:doc:`tools/plt_model_evidence`
    Calculates model evidence.

:doc:`tools/separateT`
    Splits MCMC log files by temperature points.

:doc:`tools/summarize_each_T`
    Extracts and summarizes replica information after annealing from MCMC log files for each temperature point.

.. toctree::
   :maxdepth: 2

   tutorial
   tools/index
