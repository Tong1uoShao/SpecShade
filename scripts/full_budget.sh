#!/bin/bash
# Full-budget check: does StyleRing with the WHOLE sign budget (all 44 rings, real+imaginary
# carrier, per-pair chips, balanced codeword repetition, LLR soft-combining = a frequency-domain
# analogue of Gaussian Shading) reach GS-16? Plus the spatial-sign control (GS-style carrier with
# the same BCH/Chase pipeline). Same protocol subset as capacity_axis.sh: style + regeneration
# blocks, N prompts x 2 seeds, null from the main table's 'none' images.
# usage: bash scripts/full_budget.sh [N=50]
source "$(dirname "$0")/env.sh"
N=${1:-50}; SEEDS=97,101; BLOCKS=style,regeneration
LOG=$CODE/logs/fullbudget_$(date +%Y%m%d_%H%M).log
run_one(){  # run_one <exp> <config>
  local E=$1 C=$2 M=stylering
  run_stage "GEN_$E"  $PY -u src/generate.py --exp "$E" --method "$M" --n "$N" --config "$C"
  run_stage "ATK_$E"  $PY -u src/attack.py --exp "$E" --method "$M" --blocks "$BLOCKS" --seeds "$SEEDS" --config "$C"
  run_stage "DET_$E"  $PY -u src/detect.py --exp "$E" --method "$M" --kind both --config "$C"
  run_stage "NULL_$E" $PY -u src/detect_null.py --exp "$E" --method "$M" --none_exp final --config "$C"
  run_stage "EVAL_$E" $PY -u src/evaluate.py --exp "$E" --methods "$M" --config "$C"
}
run_one full_chip    configs/full_chip.yaml
run_one full_spatial configs/full_spatial.yaml
mark "FULL_BUDGET ALL DONE"
