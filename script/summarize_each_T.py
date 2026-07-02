# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import glob
import argparse

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def read_toml(input_filename):
    """
    Reads the TOML configuration file and extracts required parameters.

    Parameters
    ----------
    input_filename : str
        Path to the TOML configuration file.

    Returns
    -------
    dict
        Contains 'replica_per_proc' and 'output_dir' extracted from the config.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
        # Numeric comparison: a string compare is wrong (e.g. "1.10.0" < "1.2.0").
        major_minor = tuple(int(x) for x in tomllib.__version__.split(".")[:2])
        if major_minor < (1, 2):
            raise ImportError("tomli 1.2.0 or later required")

    with open(input_filename, "rb") as fp:
        dict_toml = tomllib.load(fp)

    # This tool targets PAMC output; give a clear error (not a bare KeyError)
    # if the config is for another algorithm or lacks the field. Pass -n to
    # override the replica count without a config.
    try:
        replica_per_proc = dict_toml["algorithm"]["pamc"]["nreplica_per_proc"]
    except (KeyError, TypeError):
        raise KeyError(
            f"{input_filename}: [algorithm.pamc].nreplica_per_proc not found "
            "(expected a PAMC config; use -n to set the replica count instead)"
        )
    output_dir = dict_toml.get("base", {}).get("output_dir", ".")

    return {
        "replica_per_proc": replica_per_proc,
        "output_dir": output_dir
    }

def extract_columns(file_path, export_dir, replica_per_proc, written):
    """
    Extract the final population (one row per walker) from a result file.

    Selects the final sample of each walker, drops the step / walker_id /
    ancestor columns, and writes the remaining columns to a per-temperature
    output file.

    When ``replica_per_proc > 0`` the last ``replica_per_proc`` rows are taken
    (they are the final population). Otherwise the final sample of each walker
    is selected as the row with the largest step per walker id (column 1), so
    walkers that ended at different steps are handled correctly.

    Output files are aggregated across all input files of a run: the first time
    a given output file is written in this run it is truncated (``written``
    tracks that), so re-running does not duplicate data.

    Parameters
    ----------
    file_path : str
        Path to the input data file.
    export_dir : str
        Directory where the output file will be saved.
    replica_per_proc : int
        Number of final rows to extract (<= 0 selects the last step per walker).
    written : set
        Set of output paths already written in this run (mutated here).

    Note
    ----
    format of input data file:
        step walker_id T(or beta) fx x1 ... xN weight ancestor
    format of output file:
        T(or beta) fx x1 ... xN weight
    """
    # Prepare output file path
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))
    output_filename = file_base + "_summarized" + file_ext
    output_file_path = os.path.join(export_dir, output_filename)

    # Read input data file (skip comment and blank lines)
    with open(file_path, "r") as f:
        lines_all = f.readlines()
    headers = [line.strip() for line in lines_all if line.startswith("#")]
    lines = [line for line in lines_all
             if line.strip() and not line.startswith("#")]

    # Build the output header (column labels minus step/walker_id/ancestor)
    header_line = None
    for line in headers:
        if "walker" in line:
            header_line = "# " + " ".join(line.replace("#", "").split()[2:-1]) + "\n"
            break
    if header_line is None:
        print(f"Warning: no header with 'walker' found in {file_path}; "
              "output header omitted")

    # Select the final sample(s)
    if replica_per_proc > 0:
        extracted_data = lines[-replica_per_proc:]
    else:
        # Final sample of each walker: the row with the largest step per
        # walker id (column 1). Output ordered by walker id.
        last = {}
        for line in lines:
            cols = line.split()
            step, walker = int(cols[0]), cols[1]
            if walker not in last or step > last[walker][0]:
                last[walker] = (step, line)
        extracted_data = [last[w][1] for w in sorted(last, key=int)]

    # Omit step, walker_id, ancestor fields
    new_lines = [" ".join(line.split()[2:-1]) + "\n" for line in extracted_data]

    # Truncate on the first write to this output file in the current run, then
    # append for the remaining input files. This aggregates per-temperature
    # data across directories without duplicating it on re-runs.
    if output_file_path not in written:
        mode = "w"
        written.add(output_file_path)
    else:
        mode = "a"
    with open(output_file_path, mode) as f:
        if mode == "w" and header_line is not None:
            f.write(header_line)
        f.writelines(new_lines)


def main():
    """
    Main function to parse arguments and execute data extraction.
    """
    parser = argparse.ArgumentParser(description="Extract data from data files based on TOML config.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the TOML configuration file.")
    parser.add_argument("-n", "--nreplica", type=int, help="Number of replica per process.")
    parser.add_argument("-d", "--data_directory", type=str, help="Path to the data directory.")
    parser.add_argument("-o", "--export_directory", type=str, default="summarized", help="Path to the export directory.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")

    args = parser.parse_args()

    replica_per_proc = 0
    output_dir = "."

    # Read TOML configuration
    if args.input_file:
        toml_config = read_toml(args.input_file)
        replica_per_proc = toml_config["replica_per_proc"]
        output_dir = toml_config["output_dir"]

    # Overwrite options by commandline args (use "is not None" so -n 0 is honored)
    if args.nreplica is not None:
        replica_per_proc = args.nreplica
    if args.data_directory:
        output_dir = args.data_directory

    # Prepare output directory
    os.makedirs(args.export_directory, exist_ok=True)

    # Find input files
    file_pattern = os.path.join(output_dir, "*", "result_T*.txt")
    input_files = sorted(glob.glob(file_pattern))
    if not input_files:
        parser.error(f"no input files found matching {file_pattern}")

    iterator = tqdm(input_files) if (tqdm and args.progress) else input_files

    # Extract data from each file. `written` makes the per-temperature output
    # files truncate-then-append within one run (idempotent on re-run).
    written = set()
    err = 0
    for file_path in iterator:
        try:
            extract_columns(file_path, args.export_directory, replica_per_proc, written)
        except Exception as e:
            print(f"Error processing file {file_path}: {type(e).__name__}: {e}")
            err += 1

    if err:
        print(f"ERROR: {err} file(s) failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
