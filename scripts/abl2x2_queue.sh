#!/bin/bash
# Real-embedding 2x2 ablation on the 15 key attacks: GS-256 images decoded with the magnitude-weighted vote
# (configs/final_gs256_soft.yaml) and SpecShade-256 images decoded with hard cell reads (configs/final256_hard.yaml);
# the other two cells are the paper's GS-256 (hard) and SpecShade-256 (soft). Null = finalq/none (1,000 images).
# usage: setsid nohup bash scripts/abl2x2_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/abl2x2_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.abl2x2_done; mkdir -p "$DONE"
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
mark "ABL2X2 QUEUE START"
KEY15="img2img_anime:0.3 img2img_anime:0.5 adain_anime:0.6 adain_anime:1.0 ip2p_anime:1.0 regen_sd:1 regen_sd:2 regen_sd:4 vae_cheng:3 jpeg:25 blur:8 rotate:75 crop_scale:0.75 mirror:0"
CG=configs/final_gs256_soft.yaml; CO=configs/final256_hard.yaml
stage prep_gs  $PY scripts/filter_atk.py --src final_gs256q --dst final_gs256s --method gaussian_shading --cells $KEY15 --n_prompts 100
stage det_gs   $PY -u src/detect.py --exp final_gs256s --method gaussian_shading --kind both --config $CG
stage null_gs  $PY -u src/detect_null.py --exp final_gs256s --method gaussian_shading --none_exp finalq --config $CG
stage eval_gs  $PY -u src/evaluate.py --exp final_gs256s --methods gaussian_shading --config $CG --variant soft
stage prep_ours $PY scripts/filter_atk.py --src final256 --dst final256h --cells $KEY15 --n_prompts 100
stage det_ours  $PY -u src/detect.py --exp final256h --method stylering --kind both --config $CO
stage null_ours $PY -u src/detect_null.py --exp final256h --method stylering --none_exp finalq --config $CO
stage eval_ours $PY -u src/evaluate.py --exp final256h --methods stylering --config $CO --variant no_ecc
mark "ABL2X2 QUEUE ALL DONE"
