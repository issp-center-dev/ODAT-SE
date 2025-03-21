# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Template engine for text-based file generation with value substitution.

This module provides a flexible template system that supports both Python and
Fortran-style formatting, with customizable format patterns for different keywords.
"""

import os
import sys

#-- delay import
# import fortranformat

class Template:
    """
    A template engine for generating formatted text with keyword substitution.
    
    This class provides functionality to generate text content by replacing keywords
    with formatted values. It supports multiple formatting styles including Python
    formatting, Fortran-style formatting, and custom formatters.
    
    Attributes
    ----------
    content : str or None
        The template content with keywords to be replaced.
    keywords : list
        List of keywords that will be replaced with formatted values.
    style : str
        The formatting style ("default", "fortran", or "custom").
    formatter : callable
        Function used to format values.
    default_format : str
        Default format string for values.
    format_patterns : dict or None
        Mapping of keyword prefixes to format strings.
    
    Examples
    --------
    >>> # Basic usage with default formatting
    >>> template = Template(
    ...     content="Temperature: KEY_TEMP",
    ...     keywords=["KEY_TEMP"],
    ...     format="{:.2f}"
    ... )
    >>> template.generate([298.15])
    'Temperature: 298.15'
    
    >>> # Using Fortran-style formatting
    >>> template = Template(
    ...     content="Pressure: KEY_PRESS",
    ...     keywords=["KEY_PRESS"],
    ...     style="fortran",
    ...     format="F10.3"
    ... )
    >>> template.generate([101.325])
    'Pressure:    101.325'
    """

    def __init__(self, *, file=None, content=None, keywords=[], 
                 format=None, style="default", formatter=None):
        """
        Initialize the Template instance.
        
        Parameters
        ----------
        file : str, optional
            Path to template file to read content from.
        content : str, optional
            Direct template content string.
        keywords : list, optional
            List of keywords to be replaced in the template.
        format : str or dict, optional
            Format specification for values. Can be:
            - A string: Used as default format for all values
            - A dict: Maps keyword prefixes to format strings
            The special key "*" in dict provides default format.
        style : str, optional
            Formatting style to use. Options are:
            - "default": Python string formatting
            - "fortran": Fortran-style formatting
            - "custom": Custom formatter function
            Default is "default".
        formatter : callable, optional
            Custom formatter function when style="custom".
            Should accept (key, value, format) parameters.
        
        Raises
        ------
        ValueError
            If format parameter is neither str nor dict.
        IOError
            If template file cannot be read.
        """
        if content:
            self.content = content
        elif file:
            self.content = self._read_file(file)
        else:
            self.content = None

        self.keywords = keywords

        self.style = style
        if self.style.lower() == "custom":
            self.formatter = formatter
            default_format = None
        elif self.style.lower() == "fortran":
            self.formatter = self._format_value_fortran
            default_format = "F12.6"
        else:
            self.formatter = self._format_value_default
            default_format = "{:12.6f}"

        if format:
            if isinstance(format, str):
                self.default_format = format
                self.format_patterns = None
            elif isinstance(format, dict):
                if "*" in format.keys():
                    self.default_format = format["*"]
                else:
                    self.default_format = default_format
                self.format_patterns = format
            else:
                raise ValueError("unsupported type for format: {}".format(type(format)))
        else:
            self.default_format = default_format
            self.format_patterns = None

    def generate(self, values, *, output=None):
        """
        Generate formatted content by replacing keywords with values.
        
        Parameters
        ----------
        values : list
            List of values to substitute for keywords.
        output : str, optional
            Path to output file. If provided, writes result to file.
            
        Returns
        -------
        str or None
            Generated content string, or None if template content is None.
            
        Raises
        ------
        ValueError
            If number of values doesn't match number of keywords.
        IOError
            If output file cannot be written.
            
        Examples
        --------
        >>> template = Template(
        ...     content="X: KEY_X Y: KEY_Y",
        ...     keywords=["KEY_X", "KEY_Y"],
        ...     format="{:.2f}"
        ... )
        >>> template.generate([1.234, 5.678])
        'X: 1.23 Y: 5.68'
        """
        if self.content is None:
            content = None
        else:
            if len(values) != len(self.keywords):
                raise ValueError("numbers of keywords and values do not match")

            content = self.content
            for k, v in zip(self.keywords, values):
                format = self._find_format(k)
                w = self.formatter(k, v, format)
                content = content.replace(k, w)
        if output:
            try:
                with open(output, "w") as fp:
                    fp.write(content)
            except Exception as e:
                print("ERROR: {}".format(e))
                raise e
        return content

    def _format_value_default(self, key, value, fmt):
        """
        Format value using Python's string formatting.
        
        Parameters
        ----------
        key : str
            Keyword being replaced.
        value : Any
            Value to format.
        fmt : str
            Format string.
            
        Returns
        -------
        str
            Formatted value string.
        """
        s = fmt.format(value)
        return s

    def _format_value_fortran(self, key, value, fmt):
        """
        Format value using Fortran-style formatting.
        
        Parameters
        ----------
        key : str
            Keyword being replaced.
        value : Any
            Value to format.
        fmt : str
            Fortran format descriptor.
            
        Returns
        -------
        str
            Formatted value string.
        """
        import fortranformat
        writer = fortranformat.FortranRecordWriter(fmt)
        s = writer.write([value])
        return s

    def _find_format(self, keyword):
        """
        Find the appropriate format string for a keyword.
        
        Parameters
        ----------
        keyword : str
            Keyword to find format for.
            
        Returns
        -------
        str
            Format string to use for the keyword.
        """
        if self.format_patterns is not None:
            for k, f in self.format_patterns.items():
                if keyword.startswith(k):
                    return f
        return self.default_format

    def _read_file(self, file):
        """
        Read template content from a file.
        
        Parameters
        ----------
        file : str
            Path to template file.
            
        Returns
        -------
        str
            File contents as string.
            
        Raises
        ------
        IOError
            If file cannot be read.
        """
        try:
            with open(file, "r") as f:
                content = f.read()
        except Exception as e:
            print("ERROR: {}".format(e))
            content = None
            raise e
        return content
