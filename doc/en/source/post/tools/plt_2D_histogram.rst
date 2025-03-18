plt_2D_histogram.py
====================

NAME
----
Create 2D Marginalized Histograms

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_2D_histogram.py [OPTION]... [FILE]...

DESCRIPTION
-----------

Creates 2D marginalized histograms from data files specified in FILE.

Data files should be in text format containing numerical data in multiple columns.
In the standard format, values are space-separated with columns for beta, fx, x1, ..., xN, weight.
Beta is the inverse temperature, x1, ... xN are parameter values (N is the dimension of parameters), fx is the function value at that point, and weight is the weight value.
Field names can be specified with the field_list option, or you can use parameters (label_list) from the input file used in PAMC calculations.

If FILE is not specified, files named result_*_summarized.txt in the directory specified by the data_dir option will be used as data files.

The axes for creating histograms are specified with the columns option. 2D plots will be created for all combinations of the specified axes. If not specified, all axes x1, ..., xN will be used. Specify field names as a comma-separated list. For example, ``--column x1,x2,x3`` will create histograms marginalized to the ``x1 vs x2`` axis, ``x1 vs x3`` axis, and ``x2 vs x3`` axis.

The histogram range can be specified with the range option. In that case, the same range will be used for all axes being displayed. To specify ranges for each axis individually, provide a list of ``[xmin, xmax]`` pairs in the config file, or use the ``min_list`` and ``max_list`` from the input parameter file.

.. note::
   * Python 3.6 or higher is required (due to the use of f-strings).
   * The tqdm library is required for progress bar display. If not installed, regular messages will be displayed.
   * Be mindful of memory usage when processing large datasets.
   * 2D histograms are displayed on a logarithmic scale (LogNorm), allowing visualization of low-density regions.

The following command line options are available.
These options can also be provided collectively in a config file. The config file uses TOML format, with options specified in the format option_name = value.

**-b BINS, --bins BINS**
    Specifies the number of bins. Default value is 60.
    
**-c COLUMNS, --columns COLUMNS**
    Specifies the field names for creating histograms. Multiple field names can be specified as a comma-separated list. If omitted, all axes will be used.
			
**-d DATA_DIR, --data_dir DATA_DIR**
    Specifies the directory to retrieve data files from (when ``file`` is not specified). If not specified, the current directory is used.
			
**-f FORMAT, --format FORMAT**
    Specifies the format of the output histogram files. Any format supported by matplotlib can be specified. Multiple formats can be specified as a comma-separated list. Default value is ``png``.

**-o OUTPUT_DIR, --output_dir OUTPUT_DIR**
    Specifies the directory to output histogram files. If not specified, files will be written to the current directory. If the directory does not exist, it will be created automatically.

**-r RANGE, --range RANGE**
    Specifies the histogram range in the format xmin,xmax. When specified via the range command line option, it applies to all axes. To vary by axis, specify in the parameter file or config file. If not specified in any of these, ranges will be set automatically for each axis.
    
**-w WEIGHT_COLUMN, --weight_column WEIGHT_COLUMN**
    Specifies the column number (0-based) for the weight value. Default value is -1 (last column).

**--config CONFIG**
    Specifies a config file. The config file is in TOML format and specifies options equivalent to command line options. Option priority is: parameter file < config file < command line options.
    
**--params PARAMS**
    Specifies the input parameter file used when running PAMC. Range information (min_list, max_list) and field_list information (label_list) are obtained from the parameter file.
    
**--field_list FIELD_LIST**
    Specifies field names. If not specified, the standard format is assumed: beta, fx, x1, .. xN, weight (where N is the parameter dimension). When obtained from a parameter file, label_list values are used for x1 .. xN.
    Used for field name specification in columns.
    
**--progress**
    Displays a progress bar during execution. The tqdm library is required for display. If tqdm is not installed, messages about the processing status of each file will be displayed instead.
    
**-h, --help**
    Displays a help message and exits the program.

USAGE
-----

1. Run with a specified input data file file.txt. Output to the 2dhist directory.

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -o 2dhist file.txt

   2dhist/2Dhistogram_file_x1_vs_x2.png,
   2dhist/2Dhistogram_file_x1_vs_x3.png,
   2dhist/2Dhistogram_file_x2_vs_x3.png are output.

2. When input data files are prepared in the data directory as result_T0_summarized.txt to result_T10_summarized.txt. Set the output destination to the 2dhist directory.

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -d data -o 2dhist

   2Dhistogram_result_T0_beta_{beta}_x1_vs_x2.png to 2Dhistogram_result_T10_beta_{beta}_x2_vs_x3.png are output to the 2dhist directory. In the filename, ``summarized`` is replaced with ``beta_{beta}``.

3. Create a 2D histogram for the x1 and x3 fields from the input data file.txt, and output in png and pdf formats.

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -c x1,x3 -o 2dhist -f png,pdf file.txt

   2dhist/2Dhistogram_file_x1_vs_x3.png and 2dhist/2Dhistogram_file_x1_vs_x3.pdf are output.

4. Set the value range to 3.0-6.0. The same range is set for all axes.

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -r 3.0,6.0 -o 2dhist file.txt

5. Use a config file to describe the options. Prepare conf.toml as follows:

   .. code-block:: toml

      field_list = ["beta", "fx", "z1", "z2", "z3", "weight"]
      columns = ["z1", "z2"]
      bins = 120
      range = [[3.0, 6.0], [-3.0, 3.0], [0.0, 3.0]]
      data_dir = "./summarized"
      output_dir = "2dhist"

   The axis labels are z1, z2, z3, with value ranges of 3.0-6.0, -3.0-3.0, and 0.0-3.0 respectively.
   Among these, histograms will be drawn for z1 vs z2.

   Run with the config file specified.

   .. code-block:: bash

      $ python3 plt_2D_histogram.py --config conf.toml

   Histograms are created for each result_T*_summarized.txt in the summarized/ directory and output to 2dhist/2Dhistogram_result_T*.png.

NOTES
-----

Data File Format
~~~~~~~~~~~~~~~~

Data files must be in the following format:

.. code-block:: text

   # Comment line (optional)
   beta_value fx_value x1_value x2_value ... xN_value weight_value
   beta_value fx_value x1_value x2_value ... xN_value weight_value
   ...

Each line contains numerical data separated by whitespace. In the standard format, each column has the following meaning:

* Column 1: beta value (inverse temperature)
* Column 2: fx value (function value)
* Columns 3 to (N+2): Parameter values x1, x2, ..., xN
* Last column: weight

2D Histogram Display Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2D histograms generated by this script have the following features:

* Uses the "Reds" color scale (density represented by shades of red)
* Color mapping is on a logarithmic scale (LogNorm), allowing visualization of low-density regions
* Color bar is displayed as "Normalized Density (Log Scale)"
* Grid lines are displayed in light gray, making it easier to grasp data positions
* Zero-density regions are replaced with a very small value (1e-10) to enable display on a logarithmic scale

Histogram Creation Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script creates histograms through the following steps:

1. Load data from input files
2. Normalize weights (so they sum to 1)
3. Generate all combinations (pairs) of specified variables (columns)
4. Create a 2D histogram for each pair
5. Save each histogram in the specified format

Output file naming convention:

* Normal files:

  ``2Dhistogram_{input_filename}_{parameter1}_vs_{parameter2}.{format}``

* Files containing _summarized.txt (output from summarize_each_T.py):

  ``2Dhistogram_{filename_with_summarized_replaced_by_beta_{beta_value}}_{parameter1}_vs_{parameter2}.{format}``

Performance
~~~~~~~~~~~~

* Memory requirements increase with data volume
* 2D histograms require more computation and memory than 1D histograms
* With many variables, the number of combinations increases rapidly (N*(N-1)/2 histograms for N variables)
* For processing many files or combinations, use the ``--progress`` option to monitor progress

Error Handling and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If data files are not found: Error message is displayed
* If data format is invalid (non-numeric, mismatched columns): That file is skipped and an error message is displayed
* If field names don't exist: A key error occurs
* If unable to write to the output directory: Permission error is displayed
* Memory shortage: May occur especially with large datasets

If an error occurs during processing, the creation of that file or that specific histogram is skipped and processing continues.
A summary of successes and failures is displayed at the end.
