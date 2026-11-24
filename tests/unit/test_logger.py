import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.util.logger import Logger


def test_logger_without_info_or_params():
    """Constructing with neither info nor params must not raise (regression:
    info_log was None and the .get() lookups hit AttributeError). The keyword
    defaults are used."""
    lg = Logger()
    assert lg.buffer_size == 0
    assert lg.filename == "runner.log"
    assert lg.to_write_input is False
    assert lg.to_write_result is False


def test_logger_keyword_defaults_used_without_info():
    lg = Logger(buffer_size=7, filename="my.log",
                write_input=True, write_result=True)
    assert lg.buffer_size == 7
    assert lg.filename == "my.log"
    assert lg.to_write_input is True
    assert lg.to_write_result is True


def test_logger_params_take_precedence():
    lg = Logger(params={"interval": 3, "filename": "p.log", "write_input": True},
                buffer_size=99, filename="ignored.log")
    assert lg.buffer_size == 3
    assert lg.filename == "p.log"
    assert lg.to_write_input is True
    # not provided in params -> keyword default
    assert lg.to_write_result is False
