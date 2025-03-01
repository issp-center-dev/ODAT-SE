# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import sys

#-- delay import
# import fortranformat

class Template:
    def __init__(self, *, file = None, content = None, keywords = [], format = None, style = "default", formatter = None):
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

    def generate(self, values, *, output = None):
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
        s = fmt.format(value)
        return s

    def _format_value_fortran(self, key, value, fmt):
        import fortranformat
        writer = fortranformat.FortranRecordWriter(fmt)
        s = writer.write([value])
        return s

    def _find_format(self, keyword):
        if self.format_patterns is not None:
            for k, f in self.format_patterns.items():
                if keyword.startswith(k):
                    return f
        return self.default_format

    def _read_file(self, file):
        try:
            with open(file, "r") as f:
                content = f.read()
        except Exception as e:
            print("ERROR: {}".format(e))
            content = None
            raise e
        return content
