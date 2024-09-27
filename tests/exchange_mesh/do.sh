#!/bin/sh

rm -rf output

echo generate MeshData.txt
time python3 ./makemesh.py > MeshData.txt

echo
echo generate neighborlist.txt
time python3 ../../src/odatse_neighborlist.py -r 0.11 MeshData.txt

echo
echo perform exchange mc
time python3 ../../src/odatse_main.py input.toml

resfile=output/best_result.txt

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

