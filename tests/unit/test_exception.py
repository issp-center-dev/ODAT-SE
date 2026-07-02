import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.insert(0, SOURCE_PATH)

import pytest

from odatse.exception import Error, InputError


def test_input_error_str_carries_message():
    """str(exc) must return the message (regression: __init__ used to skip
    super().__init__, leaving args empty and str(exc) == '')."""
    msg = "ERROR: section base does not appear in input"
    exc = InputError(msg)
    assert str(exc) == msg
    assert exc.args == (msg,)
    # the explicit attribute is still available
    assert exc.message == msg


def test_input_error_is_error_subclass():
    assert issubclass(InputError, Error)
    assert issubclass(Error, Exception)


def test_input_error_message_survives_raise():
    msg = "bad input"
    with pytest.raises(InputError) as excinfo:
        raise InputError(msg)
    assert str(excinfo.value) == msg
    assert excinfo.value.message == msg
