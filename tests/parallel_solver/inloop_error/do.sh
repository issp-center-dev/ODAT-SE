#!/bin/sh

# A solver exception on one algorithm rank inside the main loop of a lockstep
# algorithm (exchange, pamc, ttopt) must end the job with the error reported,
# not leave the other ranks waiting in their next collective (replica
# exchange, resampling, the allgather of a sweep). See inloop_error.py.

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
rm -f log_*.txt

# expect_failure NAME INPUT FAIL_AT: rank 1 raises at its FAIL_AT-th
# evaluation; the job must exit non-zero without hanging, and the log must
# hold the error from rank 1, while rank 0 leaves quietly (main() turns the
# OtherAlgorithmProcessError of the consensus into sys.exit(0)).
expect_failure() {
  name=$1; input=$2; at=$3
  echo "=== $name ($input, failure on rank 1 at evaluation $at) ==="
  rm -rf output timed_out.flag
  log=log_$name.txt
  run_with_timeout $LIMIT env FAIL_RANK=1 FAIL_AT=$at \
    mpirun -np 2 ${PYTHON:-python3} inloop_error.py "$input" > "$log" 2>&1
  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"; res=1; return
  fi
  if [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the error"; res=1; return
  fi
  if ! grep -q "\[rank 1\] main() raised ValueError: injected failure on rank 1 at evaluation $at" "$log"; then
    echo "FAILED: the error of rank 1 was not reported (exit status $status)"; res=1; return
  fi
  if ! grep -q "\[rank 0\] main() raised SystemExit: 0" "$log"; then
    echo "FAILED: rank 0 did not leave quietly through the consensus (exit status $status)"; res=1; return
  fi
  echo "ok: failed with status $status and the error was reported"
}

# exchange: 200 steps, an exchange every 50. Evaluation 1 is the initial
# configuration, evaluation k + 1 is step k.
expect_failure exchange_initial     exchange.toml 1     # before the first exchange
expect_failure exchange_before      exchange.toml 5     # between exchanges
expect_failure exchange_after_last  exchange.toml 170   # after the last exchange: no collective
                                                       # follows, so this case passed before the
                                                       # fix too; it guards the raise after the loop
# pamc: 20 walkers per rank; evaluations 1-20 are the initial population
expect_failure pamc_initial         pamc.toml 5
expect_failure pamc_sampling        pamc.toml 50
# ttopt: the evaluations of a sweep are split over the ranks and gathered
expect_failure ttopt_sweep          ttopt.toml 100

rm -rf output
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
