# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Input file handling for AkaiKKR solver.

This module provides functions for loading and modifying AkaiKKR input files,
including applying KKR parameters from TOML configuration.
"""

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class KKRInputData:
    """
    Container for AkaiKKR input data.

    This class holds the parsed content of an AkaiKKR input file
    and provides methods for modifying parameters.

    Parameters
    ----------
    content : str
        Raw content of the input file.
    parameters : Dict[str, Any]
        Parsed parameters from the input file.

    Attributes
    ----------
    content : str
        Raw content of the input file.
    parameters : Dict[str, Any]
        Dictionary of parsed parameters.
    """

    def __init__(self, content: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize KKRInputData.

        Parameters
        ----------
        content : str
            Raw content of the input file.
        parameters : Optional[Dict[str, Any]]
            Parsed parameters from the input file.
        """
        self.content = content
        self.parameters = parameters or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value.

        Parameters
        ----------
        key : str
            Parameter name.
        default : Any
            Default value if parameter is not found.

        Returns
        -------
        Any
            Parameter value or default.
        """
        return self.parameters.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a parameter value.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value.
        """
        self.parameters[key] = value

    def copy(self) -> "KKRInputData":
        """
        Create a deep copy of this input data.

        Returns
        -------
        KKRInputData
            A deep copy of this instance.
        """
        return KKRInputData(
            content=self.content,
            parameters=copy.deepcopy(self.parameters),
        )


def load_input_file(file_path: Union[str, Path]) -> KKRInputData:
    """
    Load an AkaiKKR input file.

    Reads the template input file and parses its parameters into
    a structured format that can be modified.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the AkaiKKR input file (template.in).

    Returns
    -------
    KKRInputData
        Parsed input data containing raw content and parameters.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the input file cannot be parsed.

    Examples
    --------
    >>> input_data = load_input_file("template.in")
    >>> print(input_data.get("ewidth"))
    2.0
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read input file {file_path}: {e}") from e

    parameters = _parse_kkr_input(content)
    logger.debug(f"Loaded KKR input file: {file_path}")

    return KKRInputData(content=content, parameters=parameters)


def _parse_kkr_input(content: str) -> Dict[str, Any]:
    """
    Parse AkaiKKR input file content into a dictionary.

    Parameters
    ----------
    content : str
        Raw content of the AkaiKKR input file.

    Returns
    -------
    Dict[str, Any]
        Dictionary of parsed parameters.
    """
    parameters: Dict[str, Any] = {}
    lines = content.strip().split("\n")

    # Parse go command and pot file
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse "go" command line
        if line.startswith("go"):
            parts = line.split()
            if len(parts) >= 1:
                parameters["command"] = parts[0]
            if len(parts) >= 2:
                parameters["pot_file"] = parts[1]
            continue

        # Parse key=value pairs
        if "=" in line:
            # Handle lines like "brvtyp=fcc a=7.5"
            pairs = re.findall(r"(\w+)\s*=\s*([^\s,]+)", line)
            for key, value in pairs:
                parameters[key] = _parse_value(value)
            continue

        # Parse lines with values only (positional parameters)
        parts = line.split()
        if len(parts) >= 1:
            # Try to identify the line type based on content
            pass

    return parameters


def _parse_value(value_str: str) -> Any:
    """
    Parse a string value into the appropriate Python type.

    Parameters
    ----------
    value_str : str
        String representation of the value.

    Returns
    -------
    Any
        Parsed value (int, float, or str).
    """
    value_str = value_str.strip()

    # Try integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def apply_kkr_parameters_from_config(
    input_data: KKRInputData,
    config: Dict[str, Any],
) -> KKRInputData:
    """
    Apply KKR parameters from TOML configuration to input data.

    This function reads the [kkr] section from the configuration and
    applies the specified parameters to the input data, overwriting
    template values.

    Parameters
    ----------
    input_data : KKRInputData
        Base input data loaded from template file.
    config : Dict[str, Any]
        Configuration dictionary containing [kkr] section.

    Returns
    -------
    KKRInputData
        Modified input data with applied parameters.

    Notes
    -----
    The configuration supports the following sections:

    [kkr.go]
        - command: go command (default: "go")
        - pot_file: potential file name (default: "pot.dat")

    [kkr.lattice]
        - brvtyp: Bravais lattice type
        - a: lattice constant a
        - c_a: c/a ratio
        - b_a: b/a ratio
        - alpha, beta, gamma: lattice angles

    [kkr.calculation]
        - edelt: energy mesh width
        - ewidth: energy window width
        - reltyp: relativistic type (sra, etc.)
        - sdftyp: SDF type (mjw, etc.)
        - magtyp: magnetic type (mag, etc.)
        - record: record type (init, 2nd, etc.)

    [kkr.output]
        - outtyp: output type
        - bzqlty: BZ quality
        - maxitr: maximum iterations
        - pmix: mixing parameter

    Examples
    --------
    >>> input_data = load_input_file("template.in")
    >>> config = {
    ...     "kkr": {
    ...         "calculation": {"ewidth": 2.5, "edelt": 0.001},
    ...         "output": {"bzqlty": 2},
    ...     }
    ... }
    >>> modified_data = apply_kkr_parameters_from_config(input_data, config)
    """
    result = input_data.copy()
    kkr_config = config.get("kkr", {})

    if not kkr_config:
        logger.debug("No [kkr] section in config, returning input data unchanged")
        return result

    # Define parameter mappings for each section
    section_mappings = {
        "go": ["command", "pot_file"],
        "lattice": ["brvtyp", "a", "c_a", "b_a", "alpha", "beta", "gamma"],
        "calculation": [
            "edelt",
            "ewidth",
            "reltyp",
            "sdftyp",
            "magtyp",
            "record",
        ],
        "output": ["outtyp", "bzqlty", "maxitr", "pmix"],
    }

    # Apply parameters from each section
    for section_name, param_names in section_mappings.items():
        section_config = kkr_config.get(section_name, {})
        if not section_config:
            continue

        for param_name in param_names:
            if param_name in section_config:
                old_value = result.get(param_name)
                new_value = section_config[param_name]
                result.set(param_name, new_value)
                logger.info(
                    f"Applied KKR parameter: {param_name} = {new_value}"
                    f" (was: {old_value})"
                )

    return result


def generate_input_content(
    input_data: KKRInputData,
    variables: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate AkaiKKR input file content from input data.

    Parameters
    ----------
    input_data : KKRInputData
        Input data containing parameters.
    variables : Optional[Dict[str, float]]
        Variable substitutions for template placeholders.

    Returns
    -------
    str
        Generated input file content.
    """
    content = input_data.content

    # Apply variable substitutions
    if variables:
        for name, value in variables.items():
            # Replace placeholders like ${name} or @name@
            content = content.replace(f"${{{name}}}", str(value))
            content = content.replace(f"@{name}@", str(value))

    # Apply parameter updates
    for param_name, param_value in input_data.parameters.items():
        # Update parameter values in content
        pattern = rf"({param_name}\s*=\s*)([^\s,]+)"
        replacement = rf"\g<1>{param_value}"
        content = re.sub(pattern, replacement, content)

    return content


def write_input_file(
    input_data: KKRInputData,
    output_path: Union[str, Path],
    variables: Optional[Dict[str, float]] = None,
) -> None:
    """
    Write input data to an AkaiKKR input file.

    Parameters
    ----------
    input_data : KKRInputData
        Input data to write.
    output_path : Union[str, Path]
        Path to the output file.
    variables : Optional[Dict[str, float]]
        Variable substitutions for template placeholders.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    output_path = Path(output_path)
    content = generate_input_content(input_data, variables)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Wrote KKR input file: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to write input file {output_path}: {e}") from e

