#!/bin/sh

rm -rf output

time python3 ../../src/odatse_main.py input.toml

resfile=output/BayesData.txt

echo diff $resfile ref.txt
res=0
diff $resfile ref.txt || res=$?
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi

