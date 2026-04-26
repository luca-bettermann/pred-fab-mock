#!/bin/bash
# Helper: run one full mock pipeline test, archive plots to labeled subdir.
# Usage: ./run_test.sh <label> <weights_json> <n_baseline> <n_explore> <kappa> [extra-explore-args]
set -e
LABEL=$1
WEIGHTS=$2
N_BASE=$3
N_EXPLORE=$4
KAPPA=$5
EXTRA=${6:-}

PY=/home/claude-user/projects/pfab/pred-fab/.venv/bin/python
ARCHIVE=test_results/$LABEL
mkdir -p "$ARCHIVE"

echo "=== Test: $LABEL ==="
echo "  weights=$WEIGHTS  n_base=$N_BASE  n_explore=$N_EXPLORE  kappa=$KAPPA  extra=$EXTRA"

START=$(date +%s)

$PY cli.py reset > /dev/null 2>&1
$PY cli.py init-schema > /dev/null 2>&1
$PY cli.py init-agent > /dev/null 2>&1
$PY cli.py init-physics --seed 42 > "$ARCHIVE/00_init_physics.log" 2>&1
$PY cli.py configure --weights "$WEIGHTS" > "$ARCHIVE/configure.log" 2>&1

echo "  baseline n=$N_BASE ..."
timeout 480 $PY cli.py baseline --n $N_BASE > "$ARCHIVE/01_baseline.log" 2>&1 || echo "  baseline TIMEOUT/FAIL"

echo "  explore n=$N_EXPLORE k=$KAPPA ..."
timeout 600 $PY cli.py explore --n $N_EXPLORE --kappa $KAPPA $EXTRA > "$ARCHIVE/02_explore.log" 2>&1 || echo "  explore TIMEOUT/FAIL"

echo "  inference ..."
timeout 180 $PY cli.py inference --design-intent '{"n_layers":5}' > "$ARCHIVE/03_inference.log" 2>&1 || echo "  inference TIMEOUT/FAIL"

echo "  analyse ..."
timeout 300 $PY cli.py analyse --test-set "${TEST_SET:-0}" > "$ARCHIVE/04_analyse.log" 2>&1 || echo "  analyse TIMEOUT/FAIL"

echo "  summary ..."
$PY cli.py summary > "$ARCHIVE/05_summary.log" 2>&1

END=$(date +%s)
ELAPSED=$((END - START))
echo "  elapsed: ${ELAPSED}s"

# Copy plots for archival
cp -r plots "$ARCHIVE/plots" 2>/dev/null || true

echo "  archived to $ARCHIVE"
echo
