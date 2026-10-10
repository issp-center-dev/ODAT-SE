#!/bin/sh

# odatse.util.io: a file read on one process for the others. When the read
# fails on that process, every process of the scope must raise instead of
# waiting in the broadcast for it (issues #113, #117).

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is the failure mode under test, so the
# run is killed (and the test fails) when this expires; timeout.py exits
# with status 124 in that case (SIGTERM first, then SIGKILL).
LIMIT=60
TIMEOUT="${PYTHON:-python3} ../../test_utilities/timeout.py"

res=0

# --- the solver reads inside evaluate() with scope "solver": only the
# controller of each group reads, so a worker (rank 1) looking for a missing
# file changes nothing, while a controller (rank 2) doing so ends the job
echo "=== evaluate: missing file on a worker (rank 1): completes ==="
rm -rf output
log=log_worker.txt
FAIL_RANK=1 $TIMEOUT $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} shared_load.py --nalg 2 --nsolve 2 input_solver.toml > "$log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "FAILED: job exited with status $status (timeout = 124)"
  res=1
elif [ "$(grep -v '^#' output/ColorMap.txt | wc -l | tr -d ' ')" != "9" ]; then
  echo "FAILED: ColorMap.txt does not hold the 9 grid points"
  res=1
else
  echo "ok: completed; the worker never reads"
fi

echo "=== evaluate: missing file on a controller (rank 2): every process leaves ==="
rm -rf output
log=log_controller.txt
FAIL_RANK=2 $TIMEOUT $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} shared_load.py --nalg 2 --nsolve 2 input_solver.toml > "$log" 2>&1
status=$?
if [ $status -eq 124 ]; then
  echo "FAILED: job hung (killed after ${LIMIT}s)"
  res=1
elif [ $status -eq 0 ]; then
  echo "FAILED: job exited with status 0 despite the missing file"
  res=1
elif ! grep -q "cannot read file .*missing.txt (failed on rank 2)" "$log"; then
  echo "FAILED: the failed read was not reported (exit status $status)"
  res=1
else
  echo "ok: terminated with status $status and the failed read was reported"
fi

# --- exchange with a neighbor list that does not exist (issue #113): read on
# algorithm rank 0 only; the other algorithm rank used to wait in the
# broadcast forever
echo "=== exchange: missing neighbor list on 2 ranks ==="
rm -rf output
log=log_neighborlist.txt
$TIMEOUT $LIMIT \
  mpirun -np 2 ${PYTHON:-python3} ../../../src/odatse_main.py input_exchange.toml > "$log" 2>&1
status=$?
if [ $status -eq 124 ]; then
  echo "FAILED: job hung (killed after ${LIMIT}s)"
  res=1
elif [ $status -eq 0 ]; then
  echo "FAILED: job exited with status 0 despite the missing neighbor list"
  res=1
elif ! grep -q "ERROR: cannot read neighbor list missing_neighborlist.txt" "$log"; then
  echo "FAILED: the error was not reported (exit status $status)"
  res=1
elif grep -q "Traceback" "$log"; then
  echo "FAILED: a traceback was printed instead of the one-line report"
  res=1
else
  echo "ok: terminated with status $status and the error was reported"
fi

rm -rf output

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
