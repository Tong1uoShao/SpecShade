#!/bin/bash
# Capacity axis: StyleRing at 30 / 61 bit vs Gaussian Shading at 32 / 64 bit, same protocol subset
# (style + regeneration blocks, N prompts x 2 seeds), null from the main table's 'none' images.
# usage: bash scripts/capacity_axis.sh [N=50]
source "$(dirname "$0")/env.sh"
N=${1:-50}; SEEDS=97,101; BLOCKS=style,regeneration
LOG=$CODE/logs/capacity_$(date +%Y%m%d_%H%M).log
run_one(){  # run_one <exp> <config> <method>
  local E=$1 C=$2 M=$3
  run_stage "GEN_$E"  $PY -u src/generate.py --exp "$E" --method "$M" --n "$N" --config "$C"
  run_stage "ATK_$E"  $PY -u src/attack.py --exp "$E" --method "$M" --blocks "$BLOCKS" --seeds "$SEEDS" --config "$C"
  run_stage "DET_$E"  $PY -u src/detect.py --exp "$E" --method "$M" --kind both --config "$C"
  run_stage "NULL_$E" $PY -u src/detect_null.py --exp "$E" --method "$M" --none_exp final --config "$C"
  run_stage "EVAL_$E" $PY -u src/evaluate.py --exp "$E" --methods "$M" --config "$C"
}
run_one final_ours30 configs/final_ours30.yaml stylering
run_one final_gs32   configs/final_gs32.yaml   gaussian_shading
run_one final_ours61 configs/final_ours61.yaml stylering
run_one final_gs64   configs/final_gs64.yaml   gaussian_shading
mark "CAPACITY ALL DONE"
