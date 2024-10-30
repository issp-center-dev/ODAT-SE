#!/bin/sh

# Command to run the Python script with MPI
CMD="mpiexec --oversubscribe -np 2 python3 -u ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with input1a.toml and log the output
time $CMD input1a.toml 2>&1 | tee run.log.1a

# Run the command with input1b.toml in continuation mode and log the output
time $CMD --cont input1b.toml 2>&1 | tee run.log.1b

# Remove the output2 directory if it exists
rm -rf output2

# Run the command with input2.toml and log the output
time $CMD input2.toml 2>&1 | tee run.log.2

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