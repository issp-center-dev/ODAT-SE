#!/bin/sh

export PYTHONUNBUFFERED=1

# Remove the output directory if it exists
rm -rf output

# Run the Python script with the input file and measure the time taken
time ${PYTHON:-python3} ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/res.txt
gradfile=output/0/History_Gradient.txt

# Initialize the result variable
res=0

# The gradient history must have been written by use_gradient = true
if [ ! -f $gradfile ]; then
  echo TEST FAILED: $gradfile was not created
  res=1
fi

# Compare fx and x with the reference values within the tolerances in ref.txt
# (a text diff would depend on the last digits of the optimizer's result)
echo compare $resfile ref.txt
${PYTHON:-python3} - $resfile ref.txt <<'PYEOF' || res=1
import sys

def read(path):
    d = {}
    with open(path) as fp:
        for line in fp:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = float(v)
    return d

res = read(sys.argv[1])
ref = read(sys.argv[2])
ok = True
for key in ("fx", "x1", "x2"):
    if key not in res:
        print("missing {} in {}".format(key, sys.argv[1]))
        ok = False
        continue
    tol = ref["tol_fx"] if key == "fx" else ref["tol_x"]
    diff = abs(res[key] - ref[key])
    print("{}: {} (ref {}, |diff| = {:.3e}, tol {:.1e})".format(key, res[key], ref[key], diff, tol))
    if not diff <= tol:
        ok = False
sys.exit(0 if ok else 1)
PYEOF

# Check the result of the comparison
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
