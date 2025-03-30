#!/bin/sh

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Command to run the Python script with MPI
CMD="mpiexec -np 2 python3 ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with input1a.toml
time $CMD input1a.toml

# Run the command with input1b.toml in continuation mode
time $CMD --cont input1b.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the command with input2.toml
time $CMD input2.toml

# Define the result and reference files for comparison
# resfile=output1/best_result.txt
# reffile=output2/best_result.txt
resfile=output1/fx.txt
reffile=output2/fx.txt

# Print the diff command to be executed
echo diff $resfile $reffile

# Initialize the result variable
res=0

# Compare the result and reference files, update the result variable if they differ
diff $resfile $reffile || res=$?

# Check the result of the diff command and print the appropriate message
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
