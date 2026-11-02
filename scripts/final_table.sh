#!/bin/bash
# Paper main table: every method under the full attack matrix (style + classical + regeneration blocks),
# N prompts x 2 attack seeds, null calibration on this experiment's own 'none' images; then a quality set
# (N_Q images per method) for FID / CLIP / paired PSNR-SSIM.
# usage: bash scripts/final_table.sh [EXP=final] [N=100] [N_Q=1000] [METHODS]
source "$(dirname "$0")/env.sh"
EXP=${1:-final}; N=${2:-100}; NQ=${3:-1000}
METHODS=${4:-none,stylering,vine,stable_signature,gaussian_shading,ringid,gaussmarker}
SEEDS=97,101; BLOCKS=style,classical,regeneration
LOG=$CODE/logs/final_${EXP}_$(date +%Y%m%d_%H%M).log
IFS=',' read -ra MS <<< "$METHODS"
for M in "${MS[@]}"; do
  run_stage "GEN_$M" $PY -u src/generate.py --exp "$EXP" --method "$M" --n "$N"
  [ "$M" = "none" ] && continue
  run_stage "ATK_$M" $PY -u src/attack.py --exp "$EXP" --method "$M" --blocks "$BLOCKS" --seeds "$SEEDS"
  run_stage "DET_$M" $PY -u src/detect.py --exp "$EXP" --method "$M" --kind both
  run_stage "NULL_$M" $PY -u src/detect_null.py --exp "$EXP" --method "$M"
  run_stage "EVAL_$M" $PY -u src/evaluate.py --exp "$EXP" --methods "$M"
done
run_stage EVAL_ALL $PY -u src/evaluate.py --exp "$EXP" --methods "$METHODS"
mark "ROBUSTNESS DONE"
# ---- quality set ----
for M in "${MS[@]}"; do
  run_stage "GENQ_$M" $PY -u src/generate.py --exp "${EXP}q" --method "$M" --n "$NQ"
done
run_stage QUALITY $PY -u src/evaluate.py --exp "${EXP}q" --methods "$METHODS" --quality
run_stage QUALITY_PAIRS $PY -u src/quality_pairs.py --exp "${EXP}q" --methods "$METHODS"
mark "FINAL ALL DONE"
