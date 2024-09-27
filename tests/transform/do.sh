#!/bin/sh

rm -f output_transform/ColorMap.txt
python3 ../../src/odatse_main.py input_transform.toml

rm -f output_meshlist/ColorMap.txt
python3 ../../src/odatse_main.py input_meshlist.toml

res=$(
paste output_transform/ColorMap.txt output_meshlist/ColorMap.txt \
  | awk 'BEGIN {diff = 0.0} {diff += ($2 - $(NF))**2} END {print diff/NR}'
)
if [ $res = 0 ]; then
  echo TEST PASS
  true
else
  echo "TEST FAILED (diff = $res)"
  false
fi
