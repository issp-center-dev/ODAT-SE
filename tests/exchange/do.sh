#!/bin/sh

# Remove the output directory if it exists
rm -rf output

# Run the Python script using mpiexec with 4 processes and measure the time taken
time mpiexec --oversubscribe -np 4 python3 ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/best_result.txt

# Compare the result file with the reference file
echo diff $resfile ref.txt
res=0
diff $resfile ref.txt || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  # If the files are the same, print TEST PASS
  echo TEST PASS
  true
else
  # If the files differ, print TEST FAILED with the result file path
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi