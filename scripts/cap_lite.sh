#!/bin/bash
# Reduced capacity sweep (attacks only; generation and null detection were done by extra_queue.sh):
# first 20 prompts x seed 97, img2img (anime) s=0.5; SDXL runs with CPU offload because the GPU is shared.
# usage: setsid nohup bash scripts/cap_lite.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
export SDXL_OFFLOAD=1
LOG=$CODE/logs/cap_lite_$(date +%Y%m%d_%H%M).log
one(){  # one <exp> <config> <method> <variant>
  local E=$1 CF=$2 M=$3 V=$4
  rm -rf results/$E/$M/atk results/$E/$M/atk.jsonl
  mark "ATK_$E START"; $PY -u src/attack.py --exp "$E" --method "$M" --blocks capacity_lite --seeds 97 --limit 20 --config "$CF" >> "$LOG" 2>&1 && mark "ATK_$E OK" || mark "ATK_$E FAIL"
  mark "DET_$E START"; $PY -u src/detect.py --exp "$E" --method "$M" --kind atk --config "$CF" >> "$LOG" 2>&1 && mark "DET_$E OK" || mark "DET_$E FAIL"
  $PY -u src/evaluate.py --exp "$E" --methods "$M" --config "$CF" --variant "$V" >> "$LOG" 2>&1 && mark "EVAL_$E OK" || mark "EVAL_$E FAIL"
}
mark "CAP LITE START"
for L in 128 64 32; do
  one cap_ours$L configs/cap_ours$L.yaml stylering no_ecc
  one cap_gs$L   configs/cap_gs$L.yaml   gaussian_shading soft
done
mark "CAP LITE ALL DONE"
