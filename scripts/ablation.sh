#!/bin/bash
# Ablations of Part 3 §2.3 on StyleRing (N images, style block):
#   placement : spatial_sign | lowcut | highring | measured   (re-embed + attack + detect)
#   alloc     : measured | uniform | lowcut                    (re-embed + attack + detect)
#   decode    : soft | hard | argmin                            (detect-only, shares inverted latents)
#   no_ecc    : embed differs -> re-embed + attack + detect
# usage: bash scripts/ablation.sh <N> [placement|alloc|decode|all]
source "$(dirname "$0")/env.sh"
N=${1:-200}; WHICH=${2:-all}
LOG=$CODE/logs/ablation_$(date +%Y%m%d_%H%M).log

embed_variant(){  # embed_variant <exp> <alloc_strategy> [decode_variant]
  local EXP=$1 STRAT=$2 DEC=${3:-soft}
  run_stage "GEN_$EXP" $PY -u src/generate.py --exp "$EXP" --method stylering --n "$N" --override alloc_strategy=$STRAT decode_variant=$DEC
  run_stage "ATK_$EXP" $PY -u src/attack.py --exp "$EXP" --method stylering --blocks style
  run_stage "DET_$EXP" $PY -u src/detect.py --exp "$EXP" --method stylering --kind both --override alloc_strategy=$STRAT decode_variant=$DEC
  # null calibration with the SAME variant, on the main experiment's unwatermarked images
  run_stage "NULL_$EXP" $PY -u src/detect_null.py --exp "$EXP" --method stylering --none_exp main --override alloc_strategy=$STRAT decode_variant=$DEC
  run_stage "EVAL_$EXP" $PY -u src/evaluate.py --exp "$EXP" --methods stylering
}

if [ "$WHICH" = placement ] || [ "$WHICH" = all ]; then
  for S in spatial_sign lowcut highring; do embed_variant "abl_place_$S" $S; done   # measured == main exp
fi
if [ "$WHICH" = alloc ] || [ "$WHICH" = all ]; then
  embed_variant abl_alloc_uniform uniform
fi
if [ "$WHICH" = decode ] || [ "$WHICH" = all ]; then
  # decode-only variants on the main experiment's images (cache latents once with soft, then reuse)
  run_stage DET_main_cache $PY -u src/detect.py --exp main --method stylering --kind both --cache_latents
  for D in hard argmin; do
    run_stage "DET_main_$D" $PY -u src/detect.py --exp main --method stylering --kind both --cache_latents --decode_variant $D
    run_stage "NULL_main_$D" $PY -u src/detect_null.py --exp main --method stylering --decode_variant $D
    run_stage "EVAL_main_$D" $PY -u src/evaluate.py --exp main --methods stylering --variant $D
  done
  embed_variant abl_noecc measured no_ecc
fi
mark "ABLATION ALL DONE"
