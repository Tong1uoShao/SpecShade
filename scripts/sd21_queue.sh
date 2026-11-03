#!/bin/bash
# Second base model: SpecShade-256 and GS-256 on SD2.1-base (ungated mirror LanguageMachines/stable-diffusion-2-1-base),
# 100 prompts x 2 attack seeds, the 15 key attacks (key15), null = 1,000 unwatermarked SD2.1 images (finalq_sd21/none).
# Resumable stage markers in results/.sd21_done; a failed stage is retried once.
# usage: setsid nohup bash scripts/sd21_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/sd21_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.sd21_done; mkdir -p "$DONE"
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
mark "SD21 QUEUE START"
export SDXL_OFFLOAD=${SDXL_OFFLOAD:-0}
CO=configs/final256_sd21.yaml; CG=configs/final_gs256_sd21.yaml
stage gen_none $PY -u src/generate.py --exp finalq_sd21 --method none --n 1000 --config $CO
stage gen_ours $PY -u src/generate.py --exp final256_sd21 --method stylering --n 100 --config $CO
stage gen_gs   $PY -u src/generate.py --exp final_gs256_sd21 --method gaussian_shading --n 100 --config $CG
stage atk_ours $PY -u src/attack.py --exp final256_sd21 --method stylering --blocks key15 --seeds 97,101 --config $CO
stage atk_gs   $PY -u src/attack.py --exp final_gs256_sd21 --method gaussian_shading --blocks key15 --seeds 97,101 --config $CG
stage det_ours $PY -u src/detect.py --exp final256_sd21 --method stylering --kind both --config $CO
stage det_gs   $PY -u src/detect.py --exp final_gs256_sd21 --method gaussian_shading --kind both --config $CG
stage null_ours $PY -u src/detect_null.py --exp final256_sd21 --method stylering --none_exp finalq_sd21 --config $CO
stage null_gs   $PY -u src/detect_null.py --exp final_gs256_sd21 --method gaussian_shading --none_exp finalq_sd21 --config $CG
stage eval_ours $PY -u src/evaluate.py --exp final256_sd21 --methods stylering --config $CO --variant no_ecc
stage eval_gs   $PY -u src/evaluate.py --exp final_gs256_sd21 --methods gaussian_shading --config $CG --variant soft
mark "SD21 QUEUE ALL DONE"
