plt_1D_histogram.py
====================

NAME
----
Create 1D Marginalized Histograms

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_1D_histogram.py [OPTION]... [FILE]...

DESCRIPTION
-----------

Creates 1D marginalized histograms from data files specified in FILE.

The data files should be in text format, containing numerical data in multiple columns.
In the standard format, each line contains space-separated values for beta, fx, x1, ..., xN, weight.
Beta is the inverse temperature, x1, ... xN are parameter values (N is the dimension of parameters), fx is the function value at that point, and weight is the weight value.
Field names can be specified with the field_list option, or you can use the parameter labels (label_list) from the input file used in PAMC calculations.

If FILE is not specified, the script reads files named result_*_summarized.txt from the directory specified by the data_dir option.

The axes for creating histograms can be specified with the columns option. If not specified, all axes x1, ..., xN will be used. Specify field names as a comma-separated list. For example, ``--column x1,x3`` will create histograms marginalized along the ``x1`` and ``x3`` axes.

The histogram range can be specified with the range option. In that case, the same range will be used for all displayed axes. To specify ranges for each axis individually, provide a list of ``[xmin, xmax]`` pairs in the config file, or use the ``min_list`` and ``max_list`` from the input parameter file.

.. note::
   * Python 3.6 or higher is required (due to the use of f-strings).
   * The tqdm library is required for progress bar display. If not installed, regular messages will be displayed.
   * Be mindful of memory usage when processing large datasets.

The following command line options are available.
These options can also be provided collectively in a config file. The config file uses TOML format, with options specified in the format option_name = value.

**-b BINS, --bins BINS**
    Specifies the number of bins. Default value is 60.
    
**-c COLUMNS, --columns COLUMNS**
    Specifies the field names for which to create histograms. Multiple field names can be specified as a comma-separated list. If omitted, all axes will be used.
            
**-d DATA_DIR, --data_dir DATA_DIR**
    Specifies the directory from which to retrieve data files (when ``file`` is not specified). If not specified, the current directory is used.
            
**-f FORMAT, --format FORMAT**
    Specifies the format of the output histogram files. Any format supported by matplotlib can be specified. Multiple formats can be specified as a comma-separated list. Default value is ``png``.

**-o OUTPUT_DIR, --output_dir OUTPUT_DIR**
    Specifies the directory to which histogram files are output. If not specified, files are written to the current directory. If the directory does not exist, it is automatically created.

**-r RANGE, --range RANGE**
    Specifies the histogram range in the format xmin,xmax. If specified via the range command line option, it applies to all axes. To vary by axis, specify in the parameter file or config file. If not specified in any of these, the range is automatically set for each axis.
    
**-w WEIGHT_COLUMN, --weight_column WEIGHT_COLUMN**
    Specifies the column number (0-based) of the weight value. Default value is -1 (last column).

**--config CONFIG**
    Specifies a config file. The config file is in TOML format and specifies options equivalent to command line options. Option priority is: parameter file < config file < command line options.
    
**--params PARAMS**
    Specifies the input parameter file used when running PAMC. Range information (min_list, max_list) and field_list information (label_list) are obtained from the parameter file.
    
**--field_list FIELD_LIST**
    Specifies field names. If not specified, the standard format is assumed: beta, fx, x1, .. xN, weight (where N is the parameter dimension). If obtained from a parameter file, the values from label_list are used for x1 .. xN.
    Used for field name specification in columns.
    
**--progress**
    Displays a progress bar during execution. The tqdm library is required for display. If tqdm is not installed, messages about the processing status of each file are displayed instead.
    
**--xlabel XLABEL**
    Specifies the label string for the x-axis.
    
**-h, --help**
    Displays a help message and exits the program.

USAGE
-----

1. Run with a specified input data file file.txt. Output to the 1dhist directory.

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -o 1dhist file.txt

   1dhist/1Dhistogram_file.png is output.

2. When input data files are prepared in the data directory as result_T0_summarized.txt to result_T10_summarized.txt. Set the output destination to the 1dhist directory.

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -d data -o 1dhist

   1Dhistogram_result_T0_beta_NNNN.png to 1Dhistogram_result_T10_beta_MMMM.png are output to the 1dhist directory. In the filename, ``summarized`` is replaced with ``beta_{beta}``.

3. Create histograms for the x1 and x3 fields from the input data file.txt, and output in png and pdf formats.

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -c x1,x3 -o 1dhist -f png,pdf file.txt

   1dhist/1Dhistogram_file.png and 1dhist/1Dhistogram_file.pdf are output.

4. Set the value range to 3.0-6.0. The same range is set for all axes.

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -r 3.0,6.0 -o 1dhist file.txt

5. Use a config file to describe the options. Prepare conf.toml as follows:

   .. code-block:: toml

      field_list = ["beta", "fx", "z1", "z2", "z3", "weight"]
      columns = ["z1", "z2"]
      bins = 120
      range = [[3.0, 6.0], [-3.0, 3.0], [0.0, 3.0]]
      data_dir = "./summarized"
      output_dir = "1dhist"

   The axis labels are z1, z2, z3, and their value ranges are 3.0-6.0, -3.0-3.0, and 0.0-3.0, respectively.
   Histograms are drawn for z1 and z2.

   Run with the config file specified.

   .. code-block:: bash

      $ python3 plt_1D_histogram.py --config conf.toml

   Histograms are created for each result_T*_summarized.txt in the summarized/ directory and output to 1dhist/1Dhistogram_result_T*.png.

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

Each line consists of numerical data separated by whitespace. In the standard format, each column has the following meaning:

* Column 1: beta value (inverse temperature)
* Column 2: fx value (function value)
* Columns 3 to (N+2): Parameter values x1, x2, ..., xN
* Last column: weight

Histogram Creation Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script creates histograms using the following procedure:

1. Load data from input files
2. Normalize weights (so that they sum to 1)
3. Create a 1D histogram for each specified variable (column)
4. Save each histogram in the specified format

Output file naming convention:

* Normal files:
  
  ``1Dhistogram_{input_filename}.{format}``

* Files containing _summarized.txt (output from summarize_each_T.py):
  
  ``1Dhistogram_{input_filename_with_summarized_replaced_by_beta_{beta_value}}.{format}``

Performance
~~~~~~~~~~~~

* When processing large data files, the required memory is roughly proportional to the file size
* Processing speed is relatively fast due to the use of NumPy
* When processing many files, progress can be monitored with the ``--progress`` option

Error Handling and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If a data file is not found: An error message is displayed
* If the data format is invalid (non-numeric, mismatched column count): That file is skipped and an error message is displayed
* If a field name does not exist: A key error occurs
* If the output directory cannot be written to: A permission error is displayed

If an error occurs during processing, that file is skipped and processing continues with the next file.
A summary of successes and failures is displayed at the end.
