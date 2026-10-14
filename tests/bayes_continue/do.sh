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

# Command to run the main Python script
CMD="${PYTHON:-python3} ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the main script with input1a.toml and input1b.toml
time $CMD input1a.toml
time $CMD --cont input1b.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the main script with input2.toml
time $CMD input2.toml

# Define the result and reference files
resfile=output1/BayesData.txt
reffile=output2/BayesData.txt

# Compare the result and reference files
echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?

# Check the result of the comparison
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi

