#!/bin/bash
# Image-level baselines (VINE, Stable Signature): generate -> style attacks -> detect (no inversion)
# -> null calibration on the main experiment's unwatermarked images -> evaluate.
# usage: bash scripts/baseline_posthoc.sh <EXP> <N> "<METHODS comma>" [SEEDS]
#   e.g. bash scripts/baseline_posthoc.sh base50 50 "stable_signature,vine"
source "$(dirname "$0")/env.sh"
EXP=${1:-base50}; N=${2:-50}; METHODS=${3:-stable_signature,vine}; SEEDS=${4:-97,101,202,303,404}
LOG=$CODE/logs/posthoc_${EXP}_$(date +%Y%m%d_%H%M).log
IFS=',' read -ra MS <<< "$METHODS"
for M in "${MS[@]}"; do
  run_stage "GEN_$M" $PY -u src/generate.py --exp "$EXP" --method "$M" --n "$N"
  run_stage "ATK_$M" $PY -u src/attack.py --exp "$EXP" --method "$M" --blocks style --seeds "$SEEDS"
  run_stage "DET_$M" $PY -u src/detect.py --exp "$EXP" --method "$M" --kind both
  run_stage "NULL_$M" $PY -u src/detect_null.py --exp "$EXP" --method "$M" --none_exp main
  run_stage "EVAL_$M" $PY -u src/evaluate.py --exp "$EXP" --methods "$M"
done
mark "POSTHOC ALL DONE"
