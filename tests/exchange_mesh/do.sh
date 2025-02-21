#!/bin/sh

export PYTHONUNBUFFERED=1

# Remove the output directory if it exists
rm -rf output

# Generate MeshData.txt using makemesh.py
echo generate MeshData.txt
time python3 ./makemesh.py > MeshData.txt

echo
# Generate neighborlist.txt using odatse_neighborlist.py with a radius of 0.11
echo generate neighborlist.txt
time python3 ../../src/odatse_neighborlist.py -r 0.11 MeshData.txt

echo
# Perform exchange Monte Carlo simulation using odatse_main.py with input.toml
echo perform exchange mc
time python3 ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/best_result.txt

# Compare the result file with the reference file
echo diff $resfile ref.txt
res=0
diff $resfile ref.txt || res=$?
if [ $res -eq 0 ]; then
  # Output TEST PASS if files are identical
  echo TEST PASS
  true
else
  # Output TEST FAILED if files differ
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi
