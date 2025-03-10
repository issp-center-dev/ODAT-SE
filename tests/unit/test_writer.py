import pytest
from pathlib import Path
from odatse.util.data_writer import DataWriter, TextWriter, basicConfig

import logging
logging.basicConfig(level=logging.DEBUG)

def test_write():
    """
    Test basic DataWriter functionality with long format headers.
    
    Tests:
    - Header with description
    - Long format column descriptions
    - Basic data writing
    - File cleanup
    """
    dw = DataWriter("sample.txt", item_list=["A", "B", "C"], description="sample_output", long_format=True)
    dw.write(1,2,3)
    dw.write(4,5,6)
    dw.close()
    
    ref = """# sample_output
# 1: A
# 2: B
# 3: C
1 2 3
4 5 6
"""

    with open("sample.txt", "r") as fp:
        result = fp.read()
    assert result == ref
    Path("sample.txt").unlink()

def test_write_short():
    """
    Test DataWriter with compact header format.
    
    Tests:
    - Simple header without descriptions
    - Basic data writing
    - File cleanup
    """
    dw = DataWriter("sample.txt", item_list=["A", "B", "C"], long_format=False)
    dw.write(1,2,3)
    dw.write(4,5,6)
    dw.close()
    
    ref = """# A B C
1 2 3
4 5 6
"""

    with open("sample.txt", "r") as fp:
        result = fp.read()
    assert result == ref
    Path("sample.txt").unlink()
    
def test_write_format():
    """
    Test DataWriter with custom format specifications.
    
    Tests:
    - Float formatting with 6 decimal places
    - Integer formatting with fixed width
    - Column descriptions in long format
    - File cleanup
    """
    dw = DataWriter("sample.txt", item_list=[("A", "{:.6f}", "Alpha"), ("B", "{:4d}", "Bravo")], long_format=True)
    dw.write(1.5, 4)
    dw.write(2.7, -4)
    dw.close()
    
    ref = """# 1: Alpha
# 2: Bravo
1.500000    4
2.700000   -4
"""

    with open("sample.txt", "r") as fp:
        result = fp.read()
    assert result == ref
    Path("sample.txt").unlink()
    
def test_write_format_failure():
    """
    Test error handling for invalid format specifications.
    
    Tests:
    - Validation of item list format
    - Error message for incorrect tuple length
    """
    with pytest.raises(ValueError) as e:
        dw = DataWriter("sample.txt", item_list=[("A", "{:.6f}"), ("B", "{:4d}", "Bravo")], long_format=True)
    assert str(e.value).startswith("unknown item")
    
def test_write_combined():
    """
    Test combined output mode with multiple writers.
    
    Tests:
    - Global configuration for combined output
    - Multiple writers writing to single file
    - File tagging with source identifiers
    - Interleaved writes from different sources
    - File cleanup
    """
    basicConfig(combined_filename="combined_output.txt", combined_mode="w")

    dw1 = DataWriter("data1.txt", item_list=["A", "B", "C"], long_format=False, combined=True)
    dw2 = DataWriter("data2.txt", item_list=["D", "E"], long_format=False, combined=True)
    dw1.write(1,2,3)
    dw2.write(0.1, 0.2)
    dw1.write(4,5,6)
    dw2.write(0.4, 0.8)
    dw1.close()
    dw2.close()
    
    ref = """<data1.txt> # A B C
<data2.txt> # D E
<data1.txt> 1 2 3
<data2.txt> 0.1 0.2
<data1.txt> 4 5 6
<data2.txt> 0.4 0.8
"""

    with open("combined_output.txt", "r") as fp:
        result = fp.read()
    assert result == ref
    Path("combined_output.txt").unlink()
    
def test_write_text():
    """
    Test basic TextWriter functionality.
    
    Tests:
    - Simple text writing without formatting
    - Multi-line text handling
    - File cleanup
    """
    dw = TextWriter("sample.txt")
    dw.write("government of the people,", "by the people", "for the people,")
    dw.write("shall not perish from the earth.")
    dw.close()
    
    ref = """government of the people,
by the people
for the people,
shall not perish from the earth.
"""

    with open("sample.txt", "r") as fp:
        result = fp.read()
    assert result == ref
    Path("sample.txt").unlink()

def test_write_text_none():
    """
    Test TextWriter with no file specified.
    
    Tests:
    - Handling of null file configuration
    - No-op write operations
    """
    basicConfig(None, None)
    dw = TextWriter()
    dw.write("nothing")
    assert True

def test_write_text_combined_none():
    """
    Test combined mode cleanup with TextWriter.
    
    Tests:
    - Combined file mode initialization
    - Proper cleanup of shared resources
    - File cleanup
    """
    basicConfig("combined.txt", "a")
    dw = TextWriter(combined=True)
    dw.close()
    #XXX: Force cleanup of combined file
    dw._close_combined()
    assert True
    Path("combined.txt").unlink()
