#!/bin/bash
# Unified blind detector (configs/final256_unified.yaml: the joint search is armed as an automatic fallback on EVERY input,
# no attack-type switch) on the 15 key attacks, the four compound attacks, the 1,000-image null set and a held-out null set
# of 500 new unwatermarked images (prompts disjoint from the calibration set). Resumable markers in results/.unified_done.
# usage: setsid nohup bash scripts/unified_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/unified_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.unified_done; mkdir -p "$DONE"
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
mark "UNIFIED QUEUE START"
C=configs/final256_unified.yaml; M=stylering
KEY15="img2img_anime:0.3 img2img_anime:0.5 adain_anime:0.6 adain_anime:1.0 ip2p_anime:1.0 regen_sd:1 regen_sd:2 regen_sd:4 vae_cheng:3 jpeg:25 blur:8 rotate:75 crop_scale:0.75 mirror:0"
stage prep_key15 $PY scripts/filter_atk.py --src final256 --dst final256u --cells $KEY15 --n_prompts 100
stage prep_combo $PY scripts/filter_atk.py --src combo256 --dst combo256u --cells combo_style_crop:1 combo_style_mirror:1 combo_regen_rotate:1 combo_crop_rotate:1 --n_prompts 100
stage det_key15 $PY -u src/detect.py --exp final256u --method $M --kind both --config $C
stage det_combo $PY -u src/detect.py --exp combo256u --method $M --kind atk --config $C
stage null_cal  $PY -u src/detect_null.py --exp final256u --method $M --none_exp finalq --config $C
stage gen_heldout $PY -u src/generate.py --exp finalq2 --method none --n 500 --config configs/final256_heldout.yaml
mkdir -p results/final256u2/$M && cp results/final256u/$M/gen.jsonl results/final256u2/$M/gen.jsonl
stage null_heldout $PY -u src/detect_null.py --exp final256u2 --method $M --none_exp finalq2 --config $C
stage eval_key15 $PY -u src/evaluate.py --exp final256u --methods $M --config $C --variant no_ecc
cp results/final256u/$M/detect_null_no_ecc.jsonl results/combo256u/$M/detect_null_no_ecc.jsonl
stage eval_combo $PY -u src/evaluate.py --exp combo256u --methods $M --config $C --variant no_ecc
mark "UNIFIED QUEUE ALL DONE"
