#!/bin/sh

# Remove the existing ColorMap.txt file from the output_transform directory
rm -f output_transform/ColorMap.txt

# Run the odatse_main.py script with the input_transform.toml configuration file
python3 ../../src/odatse_main.py input_transform.toml

# Remove the existing ColorMap.txt file from the output_meshlist directory
rm -f output_meshlist/ColorMap.txt

# Run the odatse_main.py script with the input_meshlist.toml configuration file
python3 ../../src/odatse_main.py input_meshlist.toml

# Calculate the difference between the ColorMap.txt files from both outputs
res=$(
paste output_transform/ColorMap.txt output_meshlist/ColorMap.txt \
  | awk 'BEGIN {diff = 0.0} {diff += ($2 - $(NF))**2} END {print diff/NR}'
)

# Check if the difference is zero
if [ $res = 0 ]; then
  echo TEST PASS
  true
else
  echo "TEST FAILED (diff = $res)"
  false
fi