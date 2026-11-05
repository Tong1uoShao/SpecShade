#!/bin/bash
# Pilot for the compound-attack improvements (top-k candidate retention; ring-weighted search statistic), small samples:
# the two failing compound attacks (50 prompts x 2 seeds), three single attacks (50 prompts x 1 seed) and 200 nulls,
# for the unified detector (baseline, reused from final256u/combo256u where available) and the two pilot variants.
# usage: setsid nohup bash scripts/pilot_sync_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/pilot_sync_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.pilot_done; mkdir -p "$DONE"
stage(){
  local name="$1"; shift
  if [ -e "$DONE/$name" ]; then mark "$name SKIP (done)"; return 0; fi
  for try in 1 2; do
    mark "$name START (try $try)"
    if "$@" >> "$LOG" 2>&1; then touch "$DONE/$name"; mark "$name OK"; return 0; fi
    mark "$name FAIL (try $try)"; sleep 60
  done
  return 1
}
mark "PILOT START"
M=stylering
for V in topk topkw; do
  C=configs/pilot_$V.yaml
  stage prep_c_$V $PY scripts/filter_atk.py --src combo256 --dst pilot_c_$V --cells combo_style_crop:1 combo_regen_rotate:1 --n_prompts 50
  stage prep_k_$V $PY scripts/filter_atk.py --src final256 --dst pilot_k_$V --cells regen_sd:2 img2img_anime:0.5 blur:8 --n_prompts 50
  stage det_c_$V  $PY -u src/detect.py --exp pilot_c_$V --method $M --kind atk --config $C
  stage det_k_$V  $PY -u src/detect.py --exp pilot_k_$V --method $M --kind atk --config $C
  stage null_$V   $PY -u src/detect_null.py --exp pilot_k_$V --method $M --none_exp finalq200 --config $C
done
mark "PILOT ALL DONE"
