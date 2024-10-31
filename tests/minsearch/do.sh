#!/bin/sh

# Remove the output directory if it exists
rm -rf output

# Run the Python script with the input file and measure the time taken
time python3 ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/res.txt

# Display the diff command being executed
echo diff $resfile ref.txt

# Initialize the result variable
res=0

# Compare the result file with the reference file, update the result variable if they differ
diff $resfile ref.txt || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  # If the files are the same, print TEST PASS
  echo TEST PASS
  true
else
  # If the files differ, print TEST FAILED with the file names
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi