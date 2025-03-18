extract_combined.py
===================

NAME
----
Extract specific items from log files in combined format

SYNOPSIS
--------

.. code-block:: bash

   python3 extract_combined.py [OPTION]... -t tag [FILE]...


DESCRIPTION
-----------

Extracts lines tagged with a specific tag from MCMC log files in combined format.

In the combined format, multiple data types are stored with a tag at the beginning of each line in the format ``<tag>``. This script extracts lines with a specific ``tag`` and outputs them to a file. The output file will be named ``tag`` and placed in the same directory as the input file.

.. note::
   * Python 3.5 or higher is required (due to the use of type hints).
   * The tag format must be ``<tag>`` followed by a space.

The following command line options are available.
If FILE is specified, that file will be processed. If no file is explicitly specified, DATA_DIR/\*/combined.txt will be processed.

**FILE**
    Specifies the MCMC log file(s) (combined.txt). Multiple files can be specified.
    
**-t TAG, \-\-tag TAG**
    Specifies the tag to extract. This is a required parameter. The tag is a string.
    
**-d DATA_DIR, \-\-data_dir DATA_DIR**
    Specifies the directory to get data files from (when ``FILE`` is not specified).
			
**\-\-progress**
    Displays a progress bar during execution. Requires the tqdm library. If tqdm is not installed, the name of the file being processed will be displayed instead.
    
**-h, \-\-help**
    Displays help message and exits the program.

USAGE
-----

Basic usage examples are shown below.

.. code-block:: bash

   # Extract lines with the "energy" tag from a specific file
   python3 extract_combined.py -t energy path/to/combined.txt

   # Extract lines with the "acc" tag from all combined.txt files in a specific directory
   python3 extract_combined.py -t acc -d ./mcmc_results/

   # Process multiple files with a progress bar
   python3 extract_combined.py -t energy --progress file1.txt file2.txt file3.txt

NOTES
-----

Usage Notes
~~~~~~~~~~~

1. **Exact Tag Matching**: 
   Tags must match exactly. If you specify the "energy" tag, only lines starting with "``<energy>``" will be extracted.
   Lines with "``<Energy>``" or "``<energy_value>``" will not be matched. A space must follow the tag.

2. **Output File Overwriting**: 
   If a file with the same name already exists, it will be overwritten without warning. It is recommended to back up important files beforehand.

3. **Processing Large Files**: 
   The script processes files line by line, so memory consumption is kept low even for very large files.
   When processing large files, you can use the ``--progress`` option to monitor progress.

4. **Handling Filenames and Paths**:
   The output filename is simply ``tag``. If the tag contains invalid characters (characters that cannot be used in filenames),
   an error may occur when creating the file.

5. **Empty File Processing**:
   If the input file is empty or if no lines with the specified tag are found, an empty file will be generated.

Combined Format File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Files in combined format have the following structure:

.. code-block:: text

   <tag1> value1 value2 ...
   <tag2> value3 value4 ...
   <tag1> value5 value6 ...
   ...

Each line begins with a tag in the format ``<tag>``, and ``extract_combined.py`` searches for and extracts lines with matching tags.
The tag portion is removed from the output file.

General Workflow
~~~~~~~~~~~~~~~~

This script is used when analyzing MCMC simulation results:

1. When MCMC simulation is run with the export_combined_files option enabled, ``combined.txt`` files are generated.
2. Use ``extract_combined.py`` to extract necessary files.
3. Analyze or plot the extracted data files with other tools.

When processing multiple simulation results at once, combining the ``-d`` option with the ``--progress`` option is efficient.

Examples of commonly used tags in MCMC simulations:
 * ``trial.txt``: Monte Carlo trial step logs
 * ``result.txt``: Monte Carlo step logs
 * ``weight.txt``: Weight values in PAMC calculations
 * ``time.txt``: Calculation times


Error Handling and Output
~~~~~~~~~~~~~~~~~~~~~~~~~

* If the specified tag is not found: An empty file is generated
* If the input file cannot be read: An error message is displayed on standard error output
* If the output file cannot be written: A permission error is displayed on standard error output

The script displays progress information on standard output. If the ``--progress`` option is specified and the tqdm library is installed, a progress bar will be displayed. Otherwise, the name of the file being processed will be displayed.
