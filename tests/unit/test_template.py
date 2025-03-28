import os
import sys

SOURCE_PATH = os.path.join(os.path.dirname(__file__), '../../src')
sys.path.append(SOURCE_PATH)

from odatse.solver.template import Template

from tempfile import TemporaryDirectory

class switch_dir:
    """
    Context manager for temporarily changing the working directory.
    
    Parameters
    ----------
    d : str
        The directory path to switch to.
        
    Returns
    -------
    self : switch_dir
        The context manager instance.
    """
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
    """
    Test Template class with default format string ({:12.6f}).
    
    Tests basic template substitution using default formatting options
    for a single keyword replacement.
    """
    # Example format: {:12.6f} will format as "  12345.789012  "
    # default format = {:12.6f}

    s = "  opt001  "
    #   "  12345.789012  "
    v = "      0.100000  "

    tmpl = Template(content=s, keywords=["opt001"])
    r = tmpl.generate([0.1])

    assert r == v

def test_base_fortran():
    """
    Test Template class with default Fortran format (F12.6).
    
    Tests basic template substitution using Fortran-style formatting
    for a single keyword replacement.
    """
    # Example Fortran format F12.6 will format as "  12345.789012  "
    # default format in fortran mode = F12.6

    s = "  opt001  "
    #   "  12345.789012  "
    v = "      0.100000  "

    tmpl = Template(content=s, keywords=["opt001"], style="fortran")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_string():
    """
    Test Template class with custom Python format string.
    
    Tests template substitution using explicit format string {:16.8e}
    for scientific notation output.
    """
    # Format pattern shows width and expected scientific notation
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format="{:16.8e}")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_string_fortran():
    """
    Test Template class with custom Fortran format string.
    
    Tests template substitution using Fortran format 'E16.8'
    for scientific notation output.
    """
    # ... existing code ...
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    0.10000000E+00  "

    tmpl = Template(content=s, keywords=["opt001"], format="E16.8", style="fortran")
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern():
    """
    Test Template class with format pattern for specific keyword prefix.
    
    Tests template substitution using a format dictionary with 'opt' prefix
    pattern matching.
    """
    # ... existing code ...
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format={"opt": "{:16.8e}"})
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern_default():
    """
    Test Template class with default format pattern.
    
    Tests template substitution using a format dictionary with '*' as
    default pattern for all keywords.
    """
    # ... existing code ...
    s = "  opt001  "
    #   "  1234567890123456  "
    #   "     .12345678e-nn  "
    v = "    1.00000000e-01  "

    tmpl = Template(content=s, keywords=["opt001"], format={"*": "{:16.8e}"})
    r = tmpl.generate([0.1])

    assert r == v

def test_format_pattern2():
    """
    Test Template class with multiple format patterns.
    
    Tests template substitution using different format patterns for
    'opt' and 'var' keyword prefixes.
    """
    # ... existing code ...
    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "{:16.8e}", "var": "{:8.4f}"})
    r = tmpl.generate([0.1, -1.2])

    assert r == v

def test_format_pattern2_output():
    """
    Test Template class with file output.
    
    Tests template substitution and writing results to a file,
    using multiple format patterns.
    """
    # ... existing code ...
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
    """
    Test Template class with input file reading.
    
    Tests reading template from a file and generating output with
    Fortran-style formatting.
    """
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
    """
    Test Template class with empty initialization.
    
    Tests that generate() returns None when Template is initialized
    without content or file.
    """
    tmpl = Template()
    r = tmpl.generate([])

    assert r is None

def test_length_mismatch():
    """
    Test Template class with mismatched keywords and values.
    
    Tests that appropriate ValueError is raised when the number of
    values doesn't match the number of keywords.
    """
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
    """
    Test Template class with invalid format type.
    
    Tests that appropriate ValueError is raised when format
    parameter is of unsupported type.
    """
    import pytest

    with pytest.raises(ValueError) as e:
        tmpl = Template(format=["{:12.6f}"])

    assert str(e.value) == "unsupported type for format: <class 'list'>"

def test_error_file_not_found():
    """
    Test Template class with nonexistent input file.
    
    Tests that appropriate FileNotFoundError is raised when
    specified template file doesn't exist.
    """
    import pytest

    with pytest.raises(FileNotFoundError) as e:
        tmpl = Template(file="notfound.dat")

    assert str(e.value) == "[Errno 2] No such file or directory: 'notfound.dat'"

def test_error_file_output():
    """
    Test Template class with invalid output path.
    
    Tests that appropriate FileNotFoundError is raised when
    trying to write to a nonexistent directory path.
    """
    import pytest

    s = "  opt001  var02  "
    #   "  1234567890123456  12345678  "
    #   "     .12345678e-nn     .1234  "
    v = "    1.00000000e-01   -1.2000  "

    tmpl = Template(content=s, keywords=["opt001", "var02"], format={"opt": "F7.4", "var": "F7.3"}, style="fortran")

    with pytest.raises(FileNotFoundError) as e:
        r = tmpl.generate([0.1, -1.2], output="/sample/path/to/result.dat")

    assert str(e.value) == "[Errno 2] No such file or directory: '/sample/path/to/result.dat'"
