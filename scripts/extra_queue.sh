#!/bin/bash
# Extra experiments for the paper (run unattended; every stage is resumable and logged, a failed stage never stops the queue):
#   F  frequency analysis : per-ring sign survival of the Fourier coefficients vs. spatial signs (SpecShade-256 images)
#   S  synchronization ablation : the final detector with one component removed, on a subset of the existing attack sets
#   C  capacity sweep : SpecShade and Gaussian Shading at 128 / 64 / 32 bits, 50 prompts x 2 seeds,
#                       img2img (anime) s=0.5, 0.7 and 2 rounds of regeneration; null = finalq/none (1,000 images)
# usage: setsid nohup bash scripts/extra_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/extra_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.extra_done; mkdir -p "$DONE"
stage(){  # stage <name> <cmd...> : skipped when done, retried once (the GPU is shared with other users), never aborts the queue
  local name="$1"; shift
  if [ -e "$DONE/$name" ]; then mark "$name SKIP (done)"; return 0; fi
  for try in 1 2; do
    mark "$name START (try $try)"
    if "$@" >> "$LOG" 2>&1; then touch "$DONE/$name"; mark "$name OK"; return 0; fi
    mark "$name FAIL (try $try)"; sleep 60
  done
  return 1
}
mark "EXTRA QUEUE START"

# ---------------- F: frequency analysis
stage F_freq $PY -u src/proto_freq_survival.py --exp final256 --n 50 --attacks none:0 regen_sd:2.0 img2img_anime:0.5

# ---------------- S: synchronization ablation (final detector minus one component), 50 prompts x 2 seeds
M=stylering; C=configs/final256.yaml; GEO="rotate:75 crop_scale:0.75 crop_scale:0.5 mirror:0"
sync_variant(){  # sync_variant <name> <override...>
  local E=abl_sync_$1; shift
  stage S_prep_$E $PY scripts/filter_atk.py --src geo256 --dst "$E" --cells $GEO --n_prompts 50
  stage S_det_$E  $PY -u src/detect.py --exp "$E" --method $M --kind atk --config $C --override "$@"
  stage S_null_$E $PY -u src/detect_null.py --exp "$E" --method $M --none_exp finalq --config $C --override "$@"
  stage S_eval_$E $PY -u src/evaluate.py --exp "$E" --methods $M --config $C --variant no_ecc
}
sync_variant norot  rot_search=0
sync_variant nocrop crop_search=0
sync_variant noflip flip_search=0
sync_variant intpx  shift_up=1
# compound edit without the joint-search fallback: the final detector (no fallback) on the crop + rotation images, null of the final detector
E=abl_sync_nojoint
stage S_prep_$E $PY scripts/filter_atk.py --src combo256 --dst $E --cells combo_crop_rotate:1 --n_prompts 50
mkdir -p results/$E/$M && cp results/final256/$M/detect_null_no_ecc.jsonl results/$E/$M/
stage S_det_$E  $PY -u src/detect.py --exp $E --method $M --kind atk --config $C
stage S_eval_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc

# ---------------- C: capacity sweep (50 prompts x 2 seeds)
cap_one(){  # cap_one <exp> <config> <method> <variant>
  local E=$1 CF=$2 MM=$3 V=$4
  stage C_gen_$E  $PY -u src/generate.py --exp "$E" --method "$MM" --n 50 --config "$CF"
  stage C_atk_$E  $PY -u src/attack.py --exp "$E" --method "$MM" --blocks capacity --seeds 97,101 --config "$CF"
  stage C_det_$E  $PY -u src/detect.py --exp "$E" --method "$MM" --kind both --config "$CF"
  stage C_null_$E $PY -u src/detect_null.py --exp "$E" --method "$MM" --none_exp finalq --config "$CF"
  stage C_eval_$E $PY -u src/evaluate.py --exp "$E" --methods "$MM" --config "$CF" --variant "$V"
}
for L in 128 64 32; do
  cap_one cap_ours$L configs/cap_ours$L.yaml stylering no_ecc
  cap_one cap_gs$L   configs/cap_gs$L.yaml   gaussian_shading soft
done
mark "EXTRA QUEUE ALL DONE"
