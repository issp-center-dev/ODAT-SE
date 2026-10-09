#!/bin/sh

export PYTHONUNBUFFERED=1

# physbo is an optional dependency of the bayes algorithm: when it is not
# installed the test is skipped (exit status 77, SKIP_RETURN_CODE in
# CMakeLists.txt). Only its absence skips: an installed physbo that fails to
# import makes the test fail as before.
${PYTHON:-python3} -c 'import importlib.util, sys; sys.exit(77 if importlib.util.find_spec("physbo") is None else 0)'
status=$?
if [ $status -eq 77 ]; then
  echo "physbo is not installed: test skipped"
  exit 77
elif [ $status -ne 0 ]; then
  echo "ERROR: could not check for physbo (status $status)"
  exit 1
fi

# Remove the output directory if it exists
rm -rf output

# Run the Python script with the input file and measure the time taken
time ${PYTHON:-python3} ../../src/odatse_main.py input.toml

# Define the result file path
resfile=output/BayesData.txt

# Compare the result file with the reference file
echo diff $resfile ref.txt
res=0
diff $resfile ref.txt || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi
