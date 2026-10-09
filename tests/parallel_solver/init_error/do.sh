#!/bin/sh

# An error while the algorithm is constructed on the algorithm ranks only
# (here: a mesh_path that does not exist, detected by the mapper's mesh
# reader on the controllers and not on the solver workers) must terminate
# the whole job with a non-zero exit status under solver parallelism. The
# workers used to enter their control loop in main() and wait forever for
# a controller that had already exited (issue #101).

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is the failure mode under test, so the
# run is killed (and the test fails) when this expires; timeout.py exits
# with status 124 in that case (SIGTERM first, then SIGKILL).
LIMIT=60
TIMEOUT="${PYTHON:-python3} ../../test_utilities/timeout.py"

res=0

# --- the mesh file does not exist: every process must leave, status != 0
echo "=== missing mesh file ==="
rm -rf output
log=log_missing.txt
$TIMEOUT $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} ../../../src/odatse_main.py --nalg 2 --nsolve 2 input_missing.toml > "$log" 2>&1
status=$?

if [ $status -eq 124 ]; then
  echo "FAILED: job hung (killed after ${LIMIT}s)"
  res=1
elif [ $status -eq 0 ]; then
  echo "FAILED: job exited with status 0 despite the missing mesh file"
  res=1
elif ! grep -q "ERROR: .*mesh_path not found" "$log"; then
  echo "FAILED: the error was not reported (exit status $status)"
  res=1
elif grep -q "Traceback" "$log"; then
  echo "FAILED: a traceback was printed instead of the one-line report"
  res=1
else
  echo "ok: terminated with status $status and the error was reported"
fi

# --- control: the same layout with a valid (one-point) mesh file completes
echo "=== valid mesh file ==="
rm -rf output
log=log_valid.txt
$TIMEOUT $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} ../../../src/odatse_main.py --nalg 2 --nsolve 2 input_valid.toml > "$log" 2>&1
status=$?

if [ $status -eq 124 ]; then
  echo "FAILED: job hung (killed after ${LIMIT}s)"
  res=1
elif [ $status -ne 0 ]; then
  echo "FAILED: job exited with status $status"
  res=1
elif [ "$(grep -v '^#' output/ColorMap.txt | wc -l | tr -d ' ')" != "1" ]; then
  echo "FAILED: ColorMap.txt does not hold the single point"
  res=1
else
  echo "ok: completed and evaluated the point"
fi

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
