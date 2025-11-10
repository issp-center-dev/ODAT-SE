import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python diff.py <resfile> <reffile>")
    sys.exit(1)

resfile = sys.argv[1]
reffile = sys.argv[2]

def load_result(filename):
    res = {}
    for line in open(filename):
        line = line.strip()
        if line.startswith("#"):
            continue
        words = line.split()
        res[words[0]] = float(words[2]) # words[1] is '='
    return res

res = load_result(resfile)
ref = load_result(reffile)

if len(res) != len(ref):
    print(f"Number of lines in result file {resfile} and reference file {reffile} differ")
    sys.exit(1)

for key in res:
    if key not in ref:
        print(f"Key {key} not found in reference file")
        print(f"res: {res[key]}")
        print(f"ref: {ref[key]}")
        sys.exit(1)
    if abs(res[key] - ref[key]) > 1e-6:
        print(f"Key {key} differs: {res[key]} != {ref[key]}")
        sys.exit(1)

sys.exit(0)
