#!/bin/sh

CMD="python3 -u ../../src/odatse_main.py"

rm -rf output1

time $CMD input1a.toml
time $CMD --cont input1b.toml

rm -rf output2

time $CMD input2.toml


resfile=output1/BayesData.txt
reffile=output2/BayesData.txt

echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi

