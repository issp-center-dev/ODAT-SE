import math
import sys

# Allow small platform-dependent floating-point drift (Linux vs macOS/Clang).
#EPS = 0.0
EPS = 1.0e-8

if len(sys.argv) < 3:
    print("usage: {} file1 file2".format(sys.argv[0]))
    sys.exit(0)

file_a = sys.argv[1]
file_b = sys.argv[2]

with open(file_a, "r") as fp:
    lines_a = fp.readlines()
with open(file_b, "r") as fp:
    lines_b = fp.readlines()

err = 0
for idx, (ta, tb) in enumerate(zip(lines_a, lines_b)):
    vas = ta.split()
    vbs = tb.split()
    if len(vas) != len(vbs):
        print("DIFF: line {}: number of keywords differ: {} {}".format(idx, len(vas), len(vbs)))
        continue
    for va, vb in zip(vas, vbs):
        # literally equal
        if va == vb:
            continue
        # try compare as intergers
        try:
            iva = int(va)
            ivb = int(vb)
            if iva == ivb:
                continue
        except ValueError:
            pass
        # try compare as floating point numbers
        try:
            fva = float(va)
            fvb = float(vb)
            # NaN on both sides is acceptable (nan vs -nan across platforms)
            if math.isnan(fva) and math.isnan(fvb):
                continue
            if abs(fva) < EPS and abs(fvb) < EPS:
                r = abs(fva - fvb)
            else:
                r = abs(fva - fvb) / (abs(fva) + abs(fvb))
            if r < EPS:
                continue
        except ValueError:
            pass
        # different anyway
        err += 1
        print("DIFF: line {}: \"{}\" \"{}\"".format(idx, va, vb))

if err > 0:
    print("compare failed (err={})".format(err))
    sys.exit(1)

sys.exit(0)
