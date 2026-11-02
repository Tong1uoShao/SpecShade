#!/bin/bash
# Read-out controls for the full-budget check (re-detect only, images reused from the source exps):
#   full_spatial_soft : spatial-sign carrier read with the magnitude-weighted soft vote
#   full_chip_hard    : full-budget chip carrier read with plain majority (hard sign) votes
# usage: bash scripts/full_budget_ctl.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/fullbudget_ctl_$(date +%Y%m%d_%H%M).log
redetect(){  # redetect <new_exp> <src_exp> <config>
  local E=$1 S=$2 C=$3 M=stylering
  mkdir -p results/$E/$M && cp results/$S/$M/gen.jsonl results/$S/$M/atk.jsonl results/$E/$M/
  run_stage "DET_$E"  $PY -u src/detect.py --exp "$E" --method "$M" --kind both --config "$C"
  run_stage "NULL_$E" $PY -u src/detect_null.py --exp "$E" --method "$M" --none_exp final --config "$C"
  run_stage "EVAL_$E" $PY -u src/evaluate.py --exp "$E" --methods "$M" --config "$C"
}
redetect full_spatial_soft full_spatial configs/full_spatial_soft.yaml
redetect full_chip_hard    full_chip    configs/full_chip_hard.yaml
mark "FULL_BUDGET_CTL ALL DONE"
