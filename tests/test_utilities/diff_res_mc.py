import sys

if len(sys.argv) != 3:
    print("Usage: python diff.py <resfile> <reffile>")
    sys.exit(1)

resfile = sys.argv[1]
reffile = sys.argv[2]

def load_result(filename):
    res = {}
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                words = line.split()
                if len(words) != 3:
                    print(f"Invalid line: {line}")
                    sys.exit(1)
                res[words[0]] = float(words[2]) # words[1] is '='
    except Exception as e:
        print(f"Error loading result file {filename}: {e}")
        sys.exit(1)
    return res

res = load_result(resfile)
ref = load_result(reffile)

if len(res) != len(ref):
    print(f"Number of lines in result file {resfile} and reference file {reffile} differ")
    sys.exit(1)

for key in res:
    if key not in ref:
        print(f"Key {key} not found in reference file")
        sys.exit(1)
    if abs(res[key] - ref[key]) > 1e-6:
        print(f"Key {key} differs: {res[key]} != {ref[key]}")
        sys.exit(1)

sys.exit(0)
