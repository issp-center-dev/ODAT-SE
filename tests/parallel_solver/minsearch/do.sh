#!/bin/sh


export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Remove the output directory if it exists
rm -rf output

# Run the Python script with MPI
/usr/bin/time mpirun -np 8 python3 ../parallel_solver.py --nalg 4 --nsolve 2 input.toml

# Define the result file path
resfile=output/0/res.txt
reffile=ref_res.txt

# Compare the result file with the reference file
echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
