summarize_each_T.py
====================

NAME
----
Extract annealed data from each temperature point in PAMC output files


SYNOPSIS
--------

.. code-block:: bash

   python3 summarize_each_T.py [OPTION]...


DESCRIPTION
-----------

Extracts replica data at the point where annealing is completed from MCMC output files (result_T*.txt) at each temperature point for each process in PAMC calculations. The data is stored as files for each temperature point in the specified directory.

PAMC calculation data is assumed to be arranged in the format DATA_DIRECTORY/[process_number]/result_T[temperature_index].txt.
Each file format consists of space-separated numerical data: MCMC step number (step), replica number (walker), temperature (T), fx, coordinate values (x1 .. xN, where N is the dimension), weight, and ancestor.

Output data is arranged in the format EXPORT_DIRECTORY/result_T[temperature_index]_summarized.txt.
Each file format consists of: inverse temperature (beta), fx, coordinate values (x1 .. xN), and weight.

If an input parameter file used in PAMC calculations is specified as INPUT_FILE, the number of replicas (nreplica) and the directory storing calculation data (data_directory) are obtained from the input file. However, command line arguments take precedence.

.. note::
   * Python 3.6 or higher is required (due to the use of type hints and f-strings).
   * Inverse temperature (beta) is calculated as the reciprocal of temperature (T) (beta = 1/T). When T = 0, beta is set to 0.
   * By default, the last nreplica lines from each file are extracted. This number of lines corresponds to the number of replicas.
   * If nreplica is not specified, data from the last MCMC step is automatically determined and extracted.
   * The tqdm library is required for progress bar display. If not installed, processing will be executed without a progress bar.
   * If the output directory does not exist, it will be created automatically.

The following command line options are available:

**-i INPUT_FILE, \-\-input_file INPUT_FILE**
    Specifies the TOML format input parameter file used for PAMC calculations. If specified, the number of replicas and output directory are read from this file.

**-n NREPLICA, \-\-nreplica NREPLICA**
    Specifies the number of replicas per process. If not specified and no input file is specified, only data from the last step of each file is extracted.

**-d DATA_DIRECTORY, \-\-data_directory DATA_DIRECTORY**
    Directory storing PAMC calculation data. This option takes precedence even if an input file is specified.

**-o EXPORT_DIRECTORY, \-\-export_directory EXPORT_DIRECTORY**
    Directory to write extracted data. Default is "summarized".

**\-\-progress**
    Displays a progress bar during execution. The tqdm library is required for display.

**-h, \-\-help**
    Displays help message and exits the program.

USAGE
-----

1. Basic usage

   .. code-block:: bash

      python3 summarize_each_T.py -d output -o summarized

   Processes result_T*.txt files from all process folders in the output directory and saves them to the summarized directory.
   Data from the last MC step of each file is extracted.

2. Using a TOML configuration file

   .. code-block:: bash

      python3 summarize_each_T.py -i input.toml -o summarized

   Loads settings from input.toml (number of replicas, data directory), processes the data, and saves it to the summarized directory.

3. Explicitly specifying the number of replicas

   .. code-block:: bash

      python3 summarize_each_T.py -d output -n 16 -o summarized

   Extracts the last 16 lines from each file (for 16 replicas).

4. Displaying a progress bar

   .. code-block:: bash

      python3 summarize_each_T.py -d output -o summarized --progress

   Displays a progress bar during processing (requires the tqdm library).

NOTES
-----

Data Conversion Details
~~~~~~~~~~~~~~~~~~~~~~~

This script performs the following data conversions:

1. Input data format:

   If the input parameter has been given by the temperature Tmin and Tmax, 
   
   .. code-block:: text

      step walker_id T fx x1 ... xN weight ancestor

   If the input parameter has been given by the inverse temperature bmin and bmax, 
   
   .. code-block:: text

      step walker_id beta fx x1 ... xN weight ancestor

   The item types of columns are shown in the header part as comments.

2. Output data format:

   .. code-block:: text

      beta fx x1 ... xN weight

Key conversion points:
   * Extraction of data from the last MC step
   * Conversion from temperature (T) to inverse temperature (beta = 1/T)
   * Removal of unnecessary columns (step, walker_id, ancestor)

Whether the data file contains T or beta is identified from the header of the file.
If it is not identified, T is assumed and a warning message wiill be shown.
When temperature (T) is 0, inverse temperature (beta) is also set to 0.

TOML Configuration File Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TOML configuration file is expected to have the following format:

.. code-block:: toml

   [base]
   output_dir = "output"  # Data directory

   [algorithm.pamc]
   nreplica_per_proc = 16  # Number of replicas per process

Errors may occur if the required sections and parameters are not in the configuration file.

Processing Mechanism
~~~~~~~~~~~~~~~~~~~~

This script processes data in the following steps:

1. Parse command line arguments (or load from TOML configuration file)
2. Create output directory (if it doesn't exist)
3. Pattern matching of input files (DATA_DIRECTORY/\*/result_T*.txt)
4. Process each file:
   
   a. Read file line by line
   b. Identify T or beta from the comment lines
   c. Extract the last n lines if the number of replicas is specified
   d. Extract lines from the last step if the number of replicas is not specified
   e. Process data conversion (temperature → inverse temperature, remove unnecessary columns)
   f. Write results to output file

Performance and Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The `\-\-progress` option can be used to visualize progress when processing many files at once.
* Be mindful of memory usage when processing very large files.
* Since data is written to output files in append mode (`a`), results may be duplicated if the same process is executed multiple times. If re-executing, empty the output directory or specify a new directory.
* If loading settings from a TOML file, an additional library (tomli) is required for Python versions below 3.11.

Error Handling
~~~~~~~~~~~~~~

* If an input file is not found: The file processing is skipped and an error message is displayed.
* If there are no write permissions for the output directory: A permission error occurs.
* If the data line format differs from expected (e.g., insufficient columns): Errors may occur during processing of the relevant line.
* If the TOML configuration file format is incorrect: Errors occur during parsing.

The script processes each file in a try-except block, so even if an error occurs in one file, processing of other files continues.
