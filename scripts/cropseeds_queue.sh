#!/bin/bash
# Two further crop positions (attack seeds 101, 202) for the deep-crop settings of the crop sweep, for SpecShade-256 (unified
# detector) and GS-256 soft vote + the same synchronisation; evaluate.py then averages the three positions per setting.
# usage: setsid nohup bash scripts/cropseeds_queue.sh > /dev/null 2>&1 < /dev/null &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/cropseeds_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.cropseeds_done; mkdir -p "$DONE"
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
mark "CROPSEEDS START"
GS=gaussian_shading; CS=configs/final_gs256_sync_gss.yaml; CU=configs/final256_unified.yaml
stage atk_gs   $PY -u src/attack.py --exp gsync_crop --method $GS --blocks crop_sweep_deep --seeds 101,202 --config $CS
stage atk_ours $PY -u src/attack.py --exp ours_crop --method stylering --blocks crop_sweep_deep --seeds 101,202 --config $CU
stage det_gs   $PY -u src/detect.py --exp gsync_crop --method $GS --kind atk --config $CS
stage det_ours $PY -u src/detect.py --exp ours_crop --method stylering --kind atk --config $CU
stage eval_gs   $PY -u src/evaluate.py --exp gsync_crop --methods $GS --config $CS --variant gss
stage eval_ours $PY -u src/evaluate.py --exp ours_crop --methods stylering --config $CU --variant no_ecc
mark "CROPSEEDS ALL DONE"
