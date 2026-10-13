#!/bin/sh

# Construction consensus under solver parallelism: a constructor failure
# injected on one worker or one controller (check.py) must make every
# process of the job leave Algorithm(...), without a hang.

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# A hang is the failure mode under test: timeout.py kills the run after
# LIMIT seconds and exits with status 124.
LIMIT=60

rm -rf output
log=log.txt
${PYTHON:-python3} ../../test_utilities/timeout.py $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} check.py > "$log" 2>&1
status=$?
cat "$log"
rm -rf output

if [ $status -eq 124 ]; then
  echo "FAILED: job hung (killed after ${LIMIT}s)"
  echo TEST FAILED
  false
elif [ $status -ne 0 ] || ! grep -q '^ALL RANKS OK$' "$log"; then
  echo "FAILED: exit status $status"
  echo TEST FAILED
  false
else
  echo TEST PASS
  true
fi
