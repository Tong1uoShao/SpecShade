#!/bin/bash
# Sector-coherence ablation (# OURS artifact fix): keyed +-1 chips per ring sector, 63 cells.
#   sectors=0 -> per-pair chips (shortest coherence length), K -> K angular sectors, 1 -> legacy.
# Phase 1 generates every variant first (early corner-sheet inspection), phase 2 attacks/detects.
# Null calibration reuses the main experiment's unwatermarked images (same as ablation.sh).
# usage: bash scripts/abl_sectors.sh <N> "<sectors comma>" [config]
#   e.g. bash scripts/abl_sectors.sh 50 "0,8"
source "$(dirname "$0")/env.sh"
N=${1:-50}; SECS=${2:-0,8}; CFG=${3:-configs/default.yaml}
LOG=$CODE/logs/abl_sectors_$(date +%Y%m%d_%H%M).log
IFS=',' read -ra KS <<< "$SECS"
for K in "${KS[@]}"; do
  EXP=abl_sec$K
  run_stage "GEN_$EXP" $PY -u src/generate.py --exp "$EXP" --method stylering --n "$N" --config "$CFG" --override sectors=$K
done
for K in "${KS[@]}"; do
  EXP=abl_sec$K
  run_stage "ATK_$EXP" $PY -u src/attack.py --exp "$EXP" --method stylering --blocks style --config "$CFG"
  run_stage "DET_$EXP" $PY -u src/detect.py --exp "$EXP" --method stylering --kind both --config "$CFG" --override sectors=$K
  run_stage "NULL_$EXP" $PY -u src/detect_null.py --exp "$EXP" --method stylering --none_exp main --config "$CFG" --override sectors=$K
  run_stage "EVAL_$EXP" $PY -u src/evaluate.py --exp "$EXP" --methods stylering --config "$CFG"
done
mark "ABL_SECTORS ALL DONE"
