#!/bin/bash
# One full-budget variant end to end (style + regeneration blocks, N x 2 seeds, null from final/none).
# usage: bash scripts/full_budget_one.sh <exp> <config> [N=50]
source "$(dirname "$0")/env.sh"
E=$1; C=$2; N=${3:-50}; M=stylering; SEEDS=97,101; BLOCKS=${BLOCKS:-style,regeneration}
LOG=$CODE/logs/${E}_$(date +%Y%m%d_%H%M).log
run_stage "GEN_$E"  $PY -u src/generate.py --exp "$E" --method "$M" --n "$N" --config "$C"
run_stage "ATK_$E"  $PY -u src/attack.py --exp "$E" --method "$M" --blocks "$BLOCKS" --seeds "$SEEDS" --config "$C"
run_stage "DET_$E"  $PY -u src/detect.py --exp "$E" --method "$M" --kind both --config "$C"
run_stage "NULL_$E" $PY -u src/detect_null.py --exp "$E" --method "$M" --none_exp final --config "$C"
run_stage "EVAL_$E" $PY -u src/evaluate.py --exp "$E" --methods "$M" --config "$C"
mark "$E ALL DONE"
