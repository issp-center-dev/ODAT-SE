#!/bin/sh

# An exception inside solver.evaluate() on any rank of a solver group
# (--nsolve > 1) must be reported to the controller and handled like a
# controller-side failure: ignored as NaN when ignore_error = true and the
# exception is a RuntimeError, propagated (non-zero exit, no hang) otherwise.

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is one failure mode under test, so the
# run is killed (and the test fails) when this expires.
LIMIT=60

# run_with_timeout LIMIT CMD...: run CMD, kill it after LIMIT seconds.
# Sets $status to the exit status of CMD and $timed_out to 1 if it was killed.
run_with_timeout() {
  limit=$1; shift
  timed_out=0
  "$@" &
  pid=$!
  ( sleep "$limit"; kill "$pid" 2>/dev/null && touch timed_out.flag ) &
  wd=$!
  wait "$pid"
  status=$?
  kill "$wd" 2>/dev/null
  wait "$wd" 2>/dev/null
  if [ -f timed_out.flag ]; then
    timed_out=1
    rm -f timed_out.flag
  fi
}

# run_case NAME FAILMODE FAILTYPE INPUT: run one job; leaves $status, $timed_out, $log.
run_case() {
  name=$1; mode=$2; type=$3; input=$4
  echo "=== $name (FAILMODE=$mode FAILTYPE=$type $input) ==="
  rm -rf output timed_out.flag
  log=log_$name.txt
  FAILMODE=$mode FAILTYPE=$type run_with_timeout $LIMIT \
    mpirun -np 4 ${PYTHON:-python3} worker_error.py --nalg 2 --nsolve 2 "$input" > "$log" 2>&1
}

res=0

# expect_failure NAME FAILMODE FAILTYPE INPUT PATTERN...: the job must exit
# non-zero without hanging, and the log must contain every PATTERN.
expect_failure() {
  run_case "$1" "$2" "$3" "$4"; shift 4
  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"; res=1; return
  fi
  if [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the error"; res=1; return
  fi
  for pat in "$@"; do
    if ! grep -q -- "$pat" "$log"; then
      echo "FAILED: '$pat' not found in $log (exit status $status)"; res=1; return
    fi
  done
  echo "ok: failed with status $status and the error was reported"
}

# expect_ignored NAME FAILMODE FAILTYPE INPUT: the job must complete with
# status 0 and the failed evaluations must appear as NaN in the colormap.
expect_ignored() {
  run_case "$1" "$2" "$3" "$4"
  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"; res=1; return
  fi
  if [ $status -ne 0 ]; then
    echo "FAILED: job exited with status $status (see $log)"; res=1; return
  fi
  if ! grep -qi nan output/ColorMap.txt; then
    echo "FAILED: no NaN in output/ColorMap.txt"; res=1; return
  fi
  if grep -q "ERROR" "$log"; then
    echo "FAILED: an ignored error was still reported in $log"; res=1; return
  fi
  echo "ok: completed with NaN for the failed evaluations"
}

# A RuntimeError on a worker is propagated by the controller when ignore_error
# is not set, and reported with the failing rank; the worker prints its
# traceback.
expect_failure worker_runtime worker runtime input.toml \
  "solver.evaluate() failed on 1 rank(s)" \
  "RuntimeError: worker failed at evaluation 3" \
  "ERROR: solver worker failed"

# ... and ignored (NaN) when ignore_error = true, exactly like a controller
# failure. This is the case MPI_Abort on the worker used to kill.
expect_ignored worker_ignored worker runtime input_ignore.toml
expect_ignored controller_ignored controller runtime input_ignore.toml
expect_ignored all_ignored all runtime input_ignore.toml

# Every rank raising without ignore_error still terminates cleanly.
expect_failure all_runtime all runtime input.toml \
  "RuntimeError: all failed at evaluation 3"

# ignore_error covers RuntimeError only: a ValueError on a worker surfaces as
# odatse.exception.SolverError on the controller and fails the job. (The
# script calls Algorithm.main() directly, so the error is an uncaught
# traceback here; odatse.main() would print it as "ERROR: ..." instead.)
expect_failure worker_value_ignore worker value input_ignore.toml \
  "SolverError: solver.evaluate() failed on 1 rank(s)" \
  "ValueError: worker failed at evaluation 3"

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
