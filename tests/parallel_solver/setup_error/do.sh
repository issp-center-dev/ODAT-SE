#!/bin/sh

# A failure while the solver or the runner is created on some processes only
# (in practice, e.g. a file that is missing on one node) must terminate the
# whole job with a non-zero exit status and be reported by the process that
# failed. The other processes used to go on into the construction of the
# algorithm and wait there forever (issue #112). run.py injects the failure
# on one global rank.

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is the failure mode under test, so the
# run is killed (and the test fails) when this expires; timeout.py exits
# with status 124 in that case (SIGTERM first, then SIGKILL).
LIMIT=60
TIMEOUT="${PYTHON:-python3} ../../test_utilities/timeout.py"

res=0

# run NAME NP WHERE RANK [ODATSE-ARGS...]
run() {
  name=$1; np=$2; where=$3; rank=$4; shift 4
  echo "=== $name ==="
  rm -rf output
  log=log_$name.txt
  FAIL_WHERE=$where FAIL_RANK=$rank $TIMEOUT $LIMIT \
    mpirun -np $np ${PYTHON:-python3} run.py "$@" input.toml > "$log" 2>&1
  status=$?

  if [ $status -eq 124 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"
    res=1
  elif [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the failure"
    res=1
  elif ! grep -q "^\[rank $rank\] ERROR: injected $where failure" "$log"; then
    echo "FAILED: rank $rank did not report the error (exit status $status)"
    res=1
  # (only odatse's reports: the MPI runtime may print lines with "ERROR")
  elif [ "$(grep -c -E '^(\[rank [0-9]+\] )?ERROR: ' "$log")" != "1" ]; then
    echo "FAILED: the error was reported more than once"
    res=1
  elif grep -q "Traceback" "$log"; then
    echo "FAILED: a traceback was printed instead of the one-line report"
    res=1
  else
    echo "ok: terminated with status $status and rank $rank reported the error"
  fi
}

# global ranks with --nalg 2 --nsolve 2: 0 controller, 1 worker (group 0),
# 2 controller, 3 worker (group 1)
run solver_worker      4 solver 1 --nalg 2 --nsolve 2
run solver_controller  4 solver 2 --nalg 2 --nsolve 2
run runner_rank0       4 runner 0 --nalg 2 --nsolve 2
run runner_worker      4 runner 3 --nalg 2 --nsolve 2
# without solver parallelism
run solver_no_nsolve   2 solver 1

# --- a script that builds the objects itself (tests/parallel_solver/
# parallel_solver.py, which follows the tutorial): the output directory
# cannot be created on one process. There is no one-line report outside the
# odatse command: the failing process prints its traceback, the others leave
# quietly, and the job ends with a non-zero status.
for rank in 1 2; do
  echo "=== driver_makedirs_rank$rank ==="
  rm -rf output
  log=log_driver_makedirs_rank$rank.txt
  DRIVER=../parallel_solver.py FAIL_WHERE=makedirs FAIL_RANK=$rank $TIMEOUT $LIMIT \
    mpirun -np 4 ${PYTHON:-python3} run.py --nalg 2 --nsolve 2 input.toml > "$log" 2>&1
  status=$?

  if [ $status -eq 124 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"
    res=1
  elif [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the failure"
    res=1
  elif ! grep -q "PermissionError: injected makedirs failure" "$log"; then
    echo "FAILED: the injected failure was not reported (exit status $status)"
    res=1
  elif grep -q "OtherAlgorithmProcessError" "$log"; then
    echo "FAILED: the processes that did not fail printed a traceback"
    res=1
  else
    echo "ok: terminated with status $status"
  fi
done

# --- a script that builds the solver and the runner itself *without*
# fail_together() (plain_driver.py): the constructors carry the agreement
# (odatse.mpi.FailTogetherMeta), so the job must still end on every process.
# There is no one-line report outside the odatse command; the injected
# error must appear and the job must end with a non-zero status.
for spec in "solver 1" "solver 2" "runner 3"; do
  where=${spec% *}; rank=${spec#* }
  echo "=== plain_driver_${where}_rank$rank ==="
  rm -rf output
  log=log_plain_driver_${where}_rank$rank.txt
  DRIVER=plain_driver.py FAIL_WHERE=$where FAIL_RANK=$rank $TIMEOUT $LIMIT \
    mpirun -np 4 ${PYTHON:-python3} run.py --nalg 2 --nsolve 2 input.toml > "$log" 2>&1
  status=$?

  if [ $status -eq 124 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"
    res=1
  elif [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the failure"
    res=1
  elif ! grep -q "InputError: injected $where failure" "$log"; then
    echo "FAILED: the injected failure was not reported (exit status $status)"
    res=1
  elif grep -q "OtherAlgorithmProcessError" "$log"; then
    echo "FAILED: the processes that did not fail printed a traceback"
    res=1
  else
    echo "ok: terminated with status $status"
  fi
done

# --- control: the same command without a failure completes
echo "=== no failure ==="
rm -rf output
log=log_valid.txt
$TIMEOUT $LIMIT \
  mpirun -np 4 ${PYTHON:-python3} run.py --nalg 2 --nsolve 2 input.toml > "$log" 2>&1
status=$?

if [ $status -ne 0 ]; then
  echo "FAILED: job exited with status $status"
  res=1
elif [ "$(grep -v '^#' output/ColorMap.txt | wc -l | tr -d ' ')" != "9" ]; then
  echo "FAILED: ColorMap.txt does not hold the 9 grid points"
  res=1
else
  echo "ok: completed and evaluated the grid"
fi

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
