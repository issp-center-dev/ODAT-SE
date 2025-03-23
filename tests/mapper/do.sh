#!/bin/sh

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Remove the output directory if it exists
rm -rf output

# Run the Python script with MPI, using 2 processes
time mpiexec -np 2 python3 ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/ColorMap.txt

# Compare the result file with the reference file
echo diff $resfile ref.txt
res=0
diff $resfile ref.txt || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi
