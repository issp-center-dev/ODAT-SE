import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

from odatse.solver.template import Template

from tempfile import TemporaryDirectory

class switch_dir:
    def __init__(self, d):
        self.d = d
    def __enter__(self):
        self.owd = os.getcwd()
        os.chdir(self.d)
        return self
    def __exit__(self, ex_type, ex_value, tb):
        os.chdir(self.owd)
        return ex_type is None

def test_base():
    # default format = {:12.6f}

    s = "  opt001  "
    #   "  12345.789012  "
    v = "      0.100000  "

    tmpl = Template(content=s, keywords=["opt001"])
    r = tmpl.generate([0.1])

    assert r == v

def test_base_fortran():
    # default format in fortran mode = F12.6

    s = "  opt001  "
    #   "  12345.789012  "
    v = "      0.100000  "

    tmpl = Template(content=s, keywords=["opt001"], style="fortran")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_string():
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format="{:16.8e}")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_string_fortran():
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    0.10000000E+00  "

    tmpl = Template(content=s, keywords=["opt001"], format="E16.8", style="fortran")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern():
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format={"opt": "{:16.8e}"})
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern_default():
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format={"*": "{:16.8e}"})
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern2():
    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "{:16.8e}", "var": "{:8.4f}"})
    r = tmpl.generate([0.1, -1.2])

    assert r == v

def test_format_pattern2_output():
    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "{:16.8e}", "var": "{:8.4f}"})

    with TemporaryDirectory() as work_dir:
        with switch_dir(work_dir):

            r = tmpl.generate([0.1, -1.2], output="result.dat")

            with open("result.dat", "r") as fp:
                r2 = fp.read()

    assert r == v
    assert r2 == v

def test_read_file():
    s = """  1  2  2  2 
  2
  2
  opt001   0.0000   0.0000   1
  var02   1.2781   1.2781   2
   3.6050   0.0000   0.0000   3
   5.4125   1.2781   1.2781   4
   1.8075   0.0000   0.0000
"""

    v = """  1  2  2  2 
  2
  2
   0.1000   0.0000   0.0000   1
   -1.200   1.2781   1.2781   2
   3.6050   0.0000   0.0000   3
   5.4125   1.2781   1.2781   4
   1.8075   0.0000   0.0000
"""

    with TemporaryDirectory() as work_dir:
        with switch_dir(work_dir):

            with open("templ.dat", "w") as fp:
                fp.write(s)

            tmpl = Template(file="templ.dat", keywords=["opt001", "var02"], format={"opt": "F7.4", "var": "F7.3"}, style="fortran")
            r = tmpl.generate([0.1, -1.2], output="result.dat")

            with open("result.dat", "r") as fp:
                r2 = fp.read()

    assert r == v
    assert r2 == v

def test_none():
    tmpl = Template()
    r = tmpl.generate([])

    assert r is None

def test_length_mismatch():
    import pytest

    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "{:16.8e}", "var": "{:8.4f}"})

    with TemporaryDirectory() as work_dir:
        with switch_dir(work_dir):

            with pytest.raises(ValueError) as e:
                r = tmpl.generate([0.1, -1.2, 4.0], output="result.dat")

            assert str(e.value) == "numbers of keywords and values do not match"

def test_error_format():
    import pytest

    with pytest.raises(ValueError) as e:
        tmpl = Template(format=["{:12.6f}"])

    assert str(e.value) == "unsupported type for format: <class 'list'>"

def test_error_file_not_found():
    import pytest

    with pytest.raises(FileNotFoundError) as e:
        tmpl = Template(file="notfound.dat")

    assert str(e.value) == "[Errno 2] No such file or directory: 'notfound.dat'"

def test_error_file_output():
    import pytest

    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "F7.4", "var": "F7.3"}, style="fortran")

    with pytest.raises(FileNotFoundError) as e:
        r = tmpl.generate([0.1, -1.2], output="/sample/path/to/result.dat")

    assert str(e.value) == "[Errno 2] No such file or directory: '/sample/path/to/result.dat'"
