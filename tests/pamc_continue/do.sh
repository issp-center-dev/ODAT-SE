#!/bin/sh

CMD="mpiexec --oversubscribe -np 2 python3 -u ../../src/odatse_main.py"

rm -rf output1

time $CMD input1a.toml 2>&1 | tee run.log.1a
time $CMD --cont input1b.toml 2>&1 | tee run.log.1b

rm -rf output2

time $CMD input2.toml 2>&1 | tee run.log.2


#resfile=output1/best_result.txt
#reffile=output2/best_result.txt
resfile=output1/fx.txt
reffile=output2/fx.txt

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

