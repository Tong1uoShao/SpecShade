#!/bin/bash
# Deep-crop comparison with ONE synchronisation: GS-256 soft vote + the SpecShade searches (gss) vs SpecShade-256 (unified
# detector) on the crop_sweep block (area 0.5 ... 0.05, 100 images x 1 seed) of their own images. Null files are copied
# from the four-cell runs. Markers in results/.cropsweep_done.   usage: setsid nohup bash scripts/cropsweep_queue.sh > /dev/null 2>&1 < /dev/null &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/cropsweep_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.cropsweep_done; mkdir -p "$DONE"
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
mark "CROPSWEEP START"
GS=gaussian_shading; CS=configs/final_gs256_sync_gss.yaml; CU=configs/final256_unified.yaml
mkdir -p results/gsync_crop/$GS results/ours_crop/stylering
[ -e results/gsync_crop/$GS/gen.jsonl ] || cp results/final_gs256/$GS/gen.jsonl results/gsync_crop/$GS/
[ -e results/ours_crop/stylering/gen.jsonl ] || cp results/final256/stylering/gen.jsonl results/ours_crop/stylering/
stage atk_gs   $PY -u src/attack.py --exp gsync_crop --method $GS --blocks crop_sweep --seeds 97 --config $CS
stage atk_ours $PY -u src/attack.py --exp ours_crop --method stylering --blocks crop_sweep --seeds 97 --config $CU
stage det_gs   $PY -u src/detect.py --exp gsync_crop --method $GS --kind atk --config $CS
stage det_ours $PY -u src/detect.py --exp ours_crop --method stylering --kind atk --config $CU
cp results/gsync_geo/$GS/detect_null_gss.jsonl results/gsync_crop/$GS/ 2>/dev/null
cp results/final256u/stylering/detect_null_no_ecc.jsonl results/ours_crop/stylering/ 2>/dev/null
stage eval_gs   $PY -u src/evaluate.py --exp gsync_crop --methods $GS --config $CS --variant gss
stage eval_ours $PY -u src/evaluate.py --exp ours_crop --methods stylering --config $CU --variant no_ecc
mark "CROPSWEEP ALL DONE"
