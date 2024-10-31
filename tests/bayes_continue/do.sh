#!/bin/sh

# Command to run the main Python script
CMD="python3 -u ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the main script with input1a.toml and input1b.toml
time $CMD input1a.toml
time $CMD --cont input1b.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the main script with input2.toml
time $CMD input2.toml

# Define the result and reference files
resfile=output1/BayesData.txt
reffile=output2/BayesData.txt

# Compare the result and reference files
echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?

# Check the result of the comparison
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi

