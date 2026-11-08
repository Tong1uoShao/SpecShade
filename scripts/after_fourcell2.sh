#!/bin/bash
# Replacement for after_fourcell.sh: the remaining 26 cells of the 41-attack table (final256ua) are detected NOW, in
# parallel with the tail of fourcell_queue.sh (48 cores, GPU mostly idle), then the script waits for the queue and runs
# the evaluation and the fixed-payload diversity experiment. Same markers (results/.after_done) as after_fourcell.sh.
# usage: setsid nohup bash scripts/after_fourcell2.sh > /dev/null 2>&1 < /dev/null &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/after_fourcell2_$(date +%Y%m%d_%H%M).log
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
mark "AFTER-FOURCELL2 START (final256ua in parallel with the queue)"
CU=configs/final256_unified.yaml; M=stylering
mkdir -p results/final256ua/$M
[ -e results/final256ua/$M/gen.jsonl ] || cp results/final256/$M/gen.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/atk.jsonl ] || cp results/final256/$M/atk.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/detect_no_ecc.jsonl ] || cp results/final256u/$M/detect_no_ecc.jsonl results/final256ua/$M/
[ -e results/final256ua/$M/detect_null_no_ecc.jsonl ] || cp results/final256u/$M/detect_null_no_ecc.jsonl results/final256ua/$M/
stage det_ua  $PY -u src/detect.py --exp final256ua --method $M --kind both --config $CU
stage eval_ua $PY -u src/evaluate.py --exp final256ua --methods $M --config $CU --variant no_ecc
mark "AFTER-FOURCELL2 WAITING for results/.fourcell_done/eval_cmat"
while [ ! -e "$CODE/results/.fourcell_done/eval_cmat" ]; do sleep 300; done
stage diversity env TAB_OUT=$TAB_OUT $PY -u src/proto_diversity.py --config $CU --n_prompts 5 --n_seeds 20
mark "AFTER-FOURCELL2 ALL DONE"
