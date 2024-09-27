#!/bin/sh

CMD="python3 -u ../../src/odatse_main.py"
#CMD="mpiexec -np 2 python3 -u ../../src/odatse_main.py"

rm -rf output1

time timeout 12s $CMD input1.toml

time $CMD --resume input1.toml

rm -rf output2

time $CMD input2.toml


resfile=output1/ColorMap.txt
reffile=output2/ColorMap.txt

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

