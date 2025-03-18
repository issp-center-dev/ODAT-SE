separateT.py
============

NAME
----
Separate MCMC log files by temperature points

SYNOPSIS
--------

.. code-block:: bash

   python3 separateT.py [OPTION]... [FILE]...


DESCRIPTION
-----------

Separates MCMC log files (result.txt, trial.txt) into individual files for each temperature point.
Files are created in the same directory as the input, and filenames follow the format of the original filename with ``_T{index}`` appended. ``{index}`` is the index of different temperature points, assigned from 0 in the order they appear in the log file.

.. note::
   * Python 3.6 or higher is required (due to the use of type hints and f-strings).
   * MCMC log files are expected to have temperature values in the third column (index 2).
   * Comment lines (lines starting with #) in the input file are preserved in all output files.
   * The tqdm library is required for progress bar display. If not installed, regular messages will be displayed.

The following command line options are available.
If FILE is specified, that file will be processed. If no file is explicitly specified, ``DATA_DIR/*/FILE_TYPE`` will be processed.

**FILE**
    Specifies the MCMC log file. Multiple files can be specified.

**-d DATA_DIR, \-\-data_dir DATA_DIR**
    Specifies the directory from which to retrieve data files (when ``FILE`` is not specified).
			
**-t FILE_TYPE, \-\-file_type FILE_TYPE**
    Specifies the target filename when running with a directory specified. Default is result.txt.

**\-\-progress**
    Displays a progress bar during execution. The tqdm library is required for display. If tqdm is not installed, the name of the file being processed will be displayed as a message instead.

**-h, \-\-help**
    Displays help message and exits the program.

USAGE
-----

1. Run with a specified filename

   .. code-block:: bash

      python3 separateT.py output/0/result.txt

   output/0/result_T0.txt, output/0/result_T1.txt, ... are created.

2. Split files under a specified directory

   .. code-block:: bash

      python3 separateT.py -d output

   output/0/result.txt, output/1/result.txt, ... will be processed.

3. Split files other than result.txt under a specified directory

   .. code-block:: bash

      python3 separateT.py -t trial.txt -d output

   Splits output/0/trial.txt, output/1/trial.txt, ...

4. Process multiple files at once and display progress

   .. code-block:: bash

      python3 separateT.py --progress file1.txt file2.txt file3.txt

   Each file is split by temperature points, and progress is displayed with a progress bar.

NOTES
-----

File Format
~~~~~~~~~~~

The input file (MCMC log file) is expected to have the following format:

.. code-block:: text

   # Comment line (optional)
   step replica_id T fx x1 ... xN ...
   step replica_id T fx x1 ... xN ...
   ...

Each line contains space-separated data, with the third column (index 2) being the temperature value T.
Consecutive lines with the same temperature value are grouped into a single file.

Processing Mechanism
~~~~~~~~~~~~~~~~~~~~

This script processes data in the following steps:

1. Reads the input file line by line
2. Records comment lines (lines starting with #) as headers
3. Obtains the temperature value from the third column (index 2) of each data line
4. Whenever the temperature value changes, writes the accumulated data to a separate file
5. Data for each temperature value is saved to a file with the original filename plus "_T{index}"

Output File Format
~~~~~~~~~~~~~~~~~~

The output files have the following format:

* Filename: Original filename with "_T{index}" added (e.g., result.txt → result_T0.txt, result_T1.txt, ...)
* File content: Headers (comment lines) from the input file, followed by data lines with the same temperature value

Performance
~~~~~~~~~~~~

* Files are processed line by line, so memory usage is kept low even for very large files
* Data for each temperature point is buffered in memory, so memory usage may increase if there is a large amount of data for a single temperature point
* Processing time increases with the size of the input file, but is relatively fast due to line-by-line processing
* When processing multiple files, you can use the `\-\-progress` option to monitor progress

Error Handling
~~~~~~~~~~~~~~

* If the input file is not found: A file open error occurs and a message is displayed
* If the output file cannot be written: A permission error or similar occurs and a message is displayed
* If the data line has insufficient columns: An index error may occur (if the third column does not exist)
