#!/bin/bash
# T2SMark baseline (official defaults: 256-bit message in channels 1-3, 16-bit session key in channel 0, tau 0.674) on the
# 15 key attacks: generate 100 prompts -> attack (key15, seeds 97/101) -> detect -> null (finalq/none, 1,000 images, cached
# latents) -> evaluate. Resumable markers in results/.t2s_done.   usage: setsid nohup bash scripts/t2s_queue.sh > /dev/null 2>&1 < /dev/null &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/t2s_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.t2s_done; mkdir -p "$DONE"
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
mark "T2S QUEUE START"
E=final_t2s; M=t2smark; C=configs/final_t2s.yaml
export SDXL_OFFLOAD=${SDXL_OFFLOAD:-0}
stage gen  $PY -u src/generate.py --exp $E --method $M --n 100 --config $C
stage atk  $PY -u src/attack.py --exp $E --method $M --blocks key15 --seeds 97,101 --config $C
stage det  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
stage null $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C --cache_latents
stage eval $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant soft
mark "T2S QUEUE ALL DONE"
