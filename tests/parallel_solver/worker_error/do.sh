#!/bin/sh

# An exception on a solver worker (solrank > 0) must abort the whole job
# with a non-zero exit status instead of hanging the controller process.

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is the failure mode under test, so the
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

res=0
for mode in before after; do
  echo "=== FAILMODE=$mode ==="
  rm -rf output timed_out.flag
  log=log_$mode.txt

  FAILMODE=$mode run_with_timeout $LIMIT \
    mpirun -np 4 ${PYTHON:-python3} worker_error.py --nalg 2 --nsolve 2 input.toml > "$log" 2>&1

  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"
    res=1
  elif [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the worker error"
    res=1
  elif ! grep -q "ERROR: solver worker failed: worker failed $mode collective" "$log"; then
    echo "FAILED: worker error was not reported (exit status $status)"
    res=1
  else
    echo "ok: aborted with status $status and the error was reported"
  fi
done

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
