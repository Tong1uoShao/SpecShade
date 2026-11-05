#!/bin/bash
# Runs after fourcell_queue.sh (waits for its last marker): (1) the remaining 26 cells of the 41-attack table under the
# unified detector (final256ua = final256 attacked images; the 15 key cells and the clean images are seeded from
# final256u, so only the other cells are detected), (2) the fixed-payload diversity experiment (src/proto_diversity.py).
# usage: setsid nohup bash scripts/after_fourcell.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/after_fourcell_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.after_done; mkdir -p "$DONE"
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
mark "AFTER-FOURCELL WAITING for results/.fourcell_done/eval_cmat"
while [ ! -e "$CODE/results/.fourcell_done/eval_cmat" ]; do sleep 300; done
mark "AFTER-FOURCELL START"
CU=configs/final256_unified.yaml; M=stylering
mkdir -p results/final256ua/$M
[ -e results/final256ua/$M/gen.jsonl ] || cp results/final256/$M/gen.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/atk.jsonl ] || cp results/final256/$M/atk.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/detect_no_ecc.jsonl ] || cp results/final256u/$M/detect_no_ecc.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/detect_null_no_ecc.jsonl ] || cp results/final256u/$M/detect_null_no_ecc.jsonl results/final256ua/$M/
stage det_ua  $PY -u src/detect.py --exp final256ua --method $M --kind both --config $CU
stage eval_ua $PY -u src/evaluate.py --exp final256ua --methods $M --config $CU --variant no_ecc
stage diversity env TAB_OUT=$TAB_OUT $PY -u src/proto_diversity.py --config $CU --n_prompts 5 --n_seeds 20
mark "AFTER-FOURCELL ALL DONE"
