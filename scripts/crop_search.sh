#!/bin/bash
# Crop synchronisation (translation + scale search) on the geometry sweep, reusing the cached latents of
# exp full_rep_rot (same images: rotate 15..90, random crop 0.9/0.75/0.5). Null calibrated with the search.
# usage: bash scripts/crop_search.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/crop_search_$(date +%Y%m%d_%H%M).log
E=full_rep_crop; M=stylering; C=configs/full_rep_crop.yaml
mkdir -p results/$E/$M && cp results/full_rep_rot/$M/gen.jsonl results/full_rep_rot/$M/atk.jsonl results/$E/$M/ && ln -sfn ../../full_rep_rot/$M/inv results/$E/$M/inv
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C --cache_latents
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "CROP_SEARCH ALL DONE"
