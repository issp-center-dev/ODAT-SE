# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import logging

def basicConfig(combined_filename=None, combined_mode=None):
    return BaseWriter.basicConfig(combined_filename, combined_mode)

class BaseWriter:
    _fp = None
    _fp_count = 0
    _logger = logging.getLogger("BaseWriter")

    _combined_filename = "combined.txt"
    _combined_filemode = "w"

    @classmethod
    def basicConfig(cls, combined_filename=None, combined_mode=None):
        if combined_filename:
            cls._combined_filename = combined_filename
        if combined_mode:
            cls._combined_filemode = combined_mode

    def __init__(self, filename=None, mode="w", combined=False):
        self._logger = logging.getLogger(__class__.__name__)
        self.filename = filename
        self.mode = mode
        self.combined = combined
        self.fp = None
        self.tag = ""

        self._logger.debug(f"initialize: file={self.filename}, mode=\"{self.mode}\", combined={self.combined}")

        self._open()

    def __del__(self):
        self._logger.debug(f"finalize: destructor called, file={self.filename}")
        self._close()

    def close(self):
        self._close()

    def _write(self, *args):
        #self._logger.debug(f"write: {args}")
        if self.fp:
            if self.combined:
                for arg in args:
                    for s in arg.split("\n"):
                        self.fp.write(self.tag + s + "\n")
            else:
                self.fp.write("\n".join(args) + "\n")

    def _open(self):
        if self.combined:
            self._logger.debug(f"open: combined file")
            self.fp = __class__._open_combined()
            self.tag = f"<{self.filename}> "
        else:
            self._logger.debug(f"open: local file, file={self.filename}")
            if self.filename is not None:
                self.fp = open(self.filename, self.mode)
            self.tag = ""
        self._logger.debug(f"open: tag=\"{self.tag}\"")

    def _close(self):
        if self.fp:
            if self.combined:
                __class__._close_combined()
            else:
                self.fp.close()
                self._logger.debug(f"close: closed, file={self.filename}")
            self.fp = None

    @classmethod
    def _open_combined(cls):
        if cls._fp is None:
            cls._logger.debug(f"open_combined: open file {cls._combined_filename} with mode=\"{cls._combined_filemode}\"")
            cls._fp = open(cls._combined_filename, cls._combined_filemode)
        else:
            cls._logger.debug(f"open_combined: already opened")
        cls._fp_count += 1
        cls._logger.debug(f"open_combined: increment counter to {cls._fp_count}")
        return cls._fp

    @classmethod
    def _close_combined(cls):
        if cls._fp:
            cls._fp_count -= 1
            cls._logger.debug(f"close_combined: decrement counter to {cls._fp_count}")
            if cls._fp_count <= 0:
                cls._fp.close()
                cls._fp = None
                cls._logger.debug("close_combined: closed")
        

class TextWriter(BaseWriter):
    def __init__(self, filename=None, mode="w", *, combined=False):
        self._logger = logging.getLogger(__class__.__name__)
        self._logger.debug(f"initialize: file={filename}, mode=\"{mode}\", combined={combined}")
        super().__init__(filename=filename, mode=mode, combined=combined)

    def write(self, *args):
        self._write(*args)


class DataWriter(BaseWriter):
    def __init__(self, filename=None, mode="w", item_list=[], *, description=None, long_format=False, combined=False):
        self._logger = logging.getLogger(__class__.__name__)
        self._logger.debug(f"initialize: file={filename}, mode=\"{mode}\", item_list={item_list}, combined={combined}, description={description}, long_format={long_format}")
        super().__init__(filename=filename, mode=mode, combined=combined)
        self.header = self._find_item_list(item_list)
        if mode == "w":
            self._write_header(description, long_format)

    def write(self, *args):
        self._write_items(args)

    def _write_header(self, description, long_format):
        if description:
            for s in description.split("\n"):
                self._write("# " + s)
        if long_format:
            for idx, (s,_,w) in enumerate(self.header, 1):
                msg = f"# {idx}: " + (w if w else s)
                self._write(msg)
        else:
            self._write("# " + " ".join([s for s,_,_ in self.header]))
        self._logger.debug("write_header: written")

    def _write_items(self, items):
        assert len(items) == len(self.header)
        msg = []
        for (_, fmt, _), v in zip(self.header, items):
            msg.append(fmt.format(v) if fmt else str(v))
        self._write(" ".join(msg))
        #self._logger.debug(f"write_items: {items}")
        
    def _find_item_list(self, item_list):
        items = []
        for v in item_list:
            if isinstance(v, str):
                items.append((v, None, None))
            elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) == 3:
                items.append(tuple(v))
            else:
                raise ValueError(f"unknown item {v}")
        return items
