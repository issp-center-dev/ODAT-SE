#!/bin/sh
#
# random_search --cont test
#
# Cases
# -----
#  sobol  : a run continued from 8 to 16 points must evaluate exactly the
#           same point set as a single 16-point run with the same seed
#           (sobol/halton sequences are nested).  Row order differs
#           (restored points come first, and the rank distribution changes),
#           so the ColorMap bodies are compared after sorting.
#  random : the continued run keeps the 8 previously evaluated points and
#           appends 8 new ones.  The gathered row order interleaves old and
#           new points per rank, so containment is checked on sorted rows.
#  halton : continuation without an explicit seed; possible because the
#           sequence generator state is saved in the checkpoint.
#  latin  : --cont must be rejected (the design is not nested).

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

res=0

# --- sobol: continued run must reproduce the one-shot longer run ---
rm -rf output_sobol output_sobol_ref
time $CMD sobol1.toml
time $CMD --cont sobol2.toml
time $CMD sobol_ref.toml

grep -v '^#' output_sobol/ColorMap.txt | sort > sobol_cont.txt
grep -v '^#' output_sobol_ref/ColorMap.txt | sort > sobol_ref_out.txt
echo diff sobol_cont.txt sobol_ref_out.txt
diff sobol_cont.txt sobol_ref_out.txt || { echo "sobol: continued point set differs from one-shot run"; res=1; }

# --- random: continuation keeps the old points and adds new ones ---
rm -rf output_random
time $CMD random1.toml
grep -v '^#' output_random/ColorMap.txt > random_first.txt
time $CMD --cont random2.toml
grep -v '^#' output_random/ColorMap.txt > random_cont.txt

n1=$(wc -l < random_first.txt)
n2=$(wc -l < random_cont.txt)
[ "$n1" -eq 8 ] || { echo "random: unexpected number of points in first run: $n1"; res=1; }
[ "$n2" -eq 16 ] || { echo "random: unexpected number of points in continued run: $n2"; res=1; }
sort random_first.txt > random_first_s.txt
sort random_cont.txt > random_cont_s.txt
missing=$(comm -23 random_first_s.txt random_cont_s.txt)
[ -z "$missing" ] || { echo "random: previously evaluated points were not preserved:"; echo "$missing"; res=1; }

# --- halton without seed: continuation via the saved generator state ---
rm -rf output_halton
time $CMD halton1.toml
grep -v '^#' output_halton/ColorMap.txt > halton_first.txt
time $CMD --cont halton2.toml
grep -v '^#' output_halton/ColorMap.txt > halton_cont.txt

n1=$(wc -l < halton_first.txt)
n2=$(wc -l < halton_cont.txt)
[ "$n1" -eq 8 ] || { echo "halton: unexpected number of points in first run: $n1"; res=1; }
[ "$n2" -eq 16 ] || { echo "halton: unexpected number of points in continued run: $n2"; res=1; }
sort halton_first.txt > halton_first_s.txt
sort halton_cont.txt > halton_cont_s.txt
missing=$(comm -23 halton_first_s.txt halton_cont_s.txt)
[ -z "$missing" ] || { echo "halton: previously evaluated points were not preserved:"; echo "$missing"; res=1; }

# --- latin: --cont must be rejected with an explanation ---
rm -rf output_latin
time $CMD latin1.toml
if $CMD --cont latin2.toml > latin_cont.log 2>&1; then
  echo "latin: continue unexpectedly succeeded"
  res=1
else
  grep -q "not supported for the latin sequence" latin_cont.log || {
    echo "latin: continue failed with an unexpected error:"
    tail -5 latin_cont.log
    res=1
  }
fi

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
