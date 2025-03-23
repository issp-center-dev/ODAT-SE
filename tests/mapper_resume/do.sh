#!/bin/sh

export PYTHONUNBUFFERED=1

# Command to run the main Python script
CMD="python3 ../../src/odatse_main.py"
# Uncomment the following line to run with MPI
# CMD="mpiexec -np 2 python3 ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with a timeout of 12 seconds using input1.toml
time timeout 12s $CMD input1.toml

# Run the command with the --resume option using input1.toml
time $CMD --resume input1.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the command using input2.toml
time $CMD input2.toml

# Define the result and reference files
resfile=output1/ColorMap.txt
reffile=output2/ColorMap.txt

# Print the diff command to be executed
echo diff $resfile $reffile

# Initialize the result variable
res=0

# Compare the result and reference files, update res if they differ
diff $resfile $reffile || res=$?

# Check if the files are identical
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
