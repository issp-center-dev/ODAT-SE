# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import time
import odatse
import numpy as np

# type hints
from pathlib import Path
from typing import List, Dict, Any, Optional

# Parameters
# ----------
# [runner.log]
#   interval
#   filename
#   write_input
#   write_result

class Logger:
    logfile: Path
    buffer_size: int
    buffer: List[str]
    num_calls: int
    time_start: float
    time_previous: float
    to_write_result: bool
    to_write_input: bool

    def __init__(self, info: Optional[odatse.Info] = None,
                 *,
                 buffer_size: int = 0,
                 filename: str = "runner.log",
                 write_input: bool = False,
                 write_result: bool = False,
                 params: Optional[Dict[str,Any]] = None,
                 **rest) -> None:

        if info is not None:
            info_log = info.runner.get("log", {})
        else:
            info_log = params

        self.buffer_size = info_log.get("interval", buffer_size)
        self.filename = info_log.get("filename", filename)
        self.to_write_input = info_log.get("write_input", write_input)
        self.to_write_result = info_log.get("write_result", write_result)

        self.time_start = time.perf_counter()
        self.time_previous = self.time_start
        self.num_calls = 0
        self.buffer = []

    def is_active(self) -> bool:
        return self.buffer_size > 0

    def prepare(self, proc_dir: Path) -> None:
        if not self.is_active():
            return

        self.logfile = proc_dir / self.filename
        if self.logfile.exists():
            self.logfile.unlink()

        with open(self.logfile, "w") as f:
            f.write("# $1: num_calls\n")
            f.write("# $2: elapsed time from last call\n")
            f.write("# $3: elapsed time from start\n")
            if self.to_write_result:
                f.write("# $4: result\n")
            if self.to_write_input:
                f.write("# ${}-: input\n".format(5 if self.to_write_result else 4))
            f.write("\n")

    def count(self, x: np.ndarray, args, result: float) -> None:
        if not self.is_active():
            return

        self.num_calls += 1

        t = time.perf_counter()

        fields = []
        fields.append(str(self.num_calls).ljust(6))
        fields.append("{:.6f}".format(t - self.time_previous))
        fields.append("{:.6f}".format(t - self.time_start))
        if self.to_write_result:
            fields.append(result)
        if self.to_write_input:
            for val in x:
                fields.append(val)
        self.buffer.append(" ".join(map(str, fields)) + "\n")

        self.time_previous = t

        if len(self.buffer) >= self.buffer_size:
            self.write()

    def write(self) -> None:
        if not self.is_active():
            return
        with open(self.logfile, "a") as f:
            for w in self.buffer:
                f.write(w)
        self.buffer.clear()
