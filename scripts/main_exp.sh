#!/bin/bash
# Main experiment pipeline for a list of methods: generate -> attack -> detect (+ null calibration).
# usage: bash scripts/main_exp.sh <EXP> <N> "<METHODS comma>" "<BLOCKS comma>"
#   e.g. bash scripts/main_exp.sh main 200 "none,stylering,gaussian_shading,ringid,gaussmarker" "style"
source "$(dirname "$0")/env.sh"
EXP=${1:-main}; N=${2:-200}; METHODS=${3:-none,stylering}; BLOCKS=${4:-style}; SEEDS=${5:-97,101,202,303,404}
LOG=$CODE/logs/main_${EXP}_$(date +%Y%m%d_%H%M).log
IFS=',' read -ra MS <<< "$METHODS"
# 1) generation for every method (incl. 'none' control)
for M in "${MS[@]}"; do run_stage "GEN_$M" $PY -u src/generate.py --exp "$EXP" --method "$M" --n "$N"; done
# 2) attacks + detection per watermark method
for M in "${MS[@]}"; do
  [ "$M" = "none" ] && continue
  run_stage "ATK_$M" $PY -u src/attack.py --exp "$EXP" --method "$M" --blocks "$BLOCKS" --seeds "$SEEDS"
  run_stage "DET_$M" $PY -u src/detect.py --exp "$EXP" --method "$M" --kind both
  # null calibration: decode the unwatermarked 'none' images with this method's key
  run_stage "NULL_$M" $PY -u src/detect_null.py --exp "$EXP" --method "$M"
done
run_stage EVAL $PY -u src/evaluate.py --exp "$EXP" --methods "$METHODS"
mark "MAIN ALL DONE"
