#!/bin/sh

CMD="mpiexec --oversubscribe -np 4 python3 -u ../../src/odatse_main.py"

rm -rf output1

time $CMD input1a.toml
time $CMD --cont input1b.toml

rm -rf output2

time $CMD input2.toml


#resfile=output1/best_result.txt
#reffile=output2/best_result.txt
resfile=output1/result_T0.txt
reffile=output2/result_T0.txt

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

