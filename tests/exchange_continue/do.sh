#!/bin/sh

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Command to run the Python script with MPI
CMD="mpiexec -np 4 python3 ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with the first input file and measure the time taken
time $CMD input1a.toml
# Run the command with the continuation input file and measure the time taken
time $CMD --cont input1b.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the command with the second input file and measure the time taken
time $CMD input2.toml

# Define the result files to compare
# resfile=output1/best_result.txt
# reffile=output2/best_result.txt
resfile=output1/result_T0.txt
reffile=output2/result_T0.txt

# Compare the result files and store the result of the comparison
echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?

# Check the result of the comparison and print the appropriate message
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
