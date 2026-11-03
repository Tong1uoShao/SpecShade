#!/bin/bash
# SFW baselines (HSTR, HSQR) on the 15 key attacks: generate 100 prompts -> attack (key15, seeds 97/101) -> detect ->
# null (finalq/none, 1,000 images) -> evaluate. Resumable stage markers in results/.sfw_done; a failed stage is retried once.
# usage: setsid nohup bash scripts/sfw_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/sfw_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.sfw_done; mkdir -p "$DONE"
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
mark "SFW QUEUE START"
E=final_sfw; C=configs/final_sfw.yaml
export SDXL_OFFLOAD=${SDXL_OFFLOAD:-0}      # the GPU is shared; keep the SDXL attack engines within ~7 GB
for M in sfw_hstr sfw_hsqr; do
  stage gen_$M  $PY -u src/generate.py --exp $E --method $M --n 100 --config $C
  stage atk_$M  $PY -u src/attack.py --exp $E --method $M --blocks key15 --seeds 97,101 --config $C
  stage det_$M  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
  stage null_$M $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
  stage eval_$M $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant soft
done
mark "SFW QUEUE ALL DONE"
