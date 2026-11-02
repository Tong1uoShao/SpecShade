#!/bin/bash
# Gated synchronisation (rotation + crop search, both peak-gated) as the final detector:
#  (1) geometry sweep on the cached full_rep_rot latents (exp full_rep_geo);
#  (2) style + regeneration images of full_rep re-detected with the same detector (exp full_rep_syncq):
#      the price of the always-on searches on non-geometric attacks.
# usage: bash scripts/sync_gated.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/sync_gated_$(date +%Y%m%d_%H%M).log
M=stylering; C=configs/full_rep_sync.yaml
E=full_rep_geo
mkdir -p results/$E/$M && cp results/full_rep_rot/$M/gen.jsonl results/full_rep_rot/$M/atk.jsonl results/$E/$M/ && ln -sfn ../../full_rep_rot/$M/inv results/$E/$M/inv
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C --cache_latents
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
E=full_rep_syncq
mkdir -p results/$E/$M && cp results/full_rep/$M/gen.jsonl results/full_rep/$M/atk.jsonl results/$E/$M/
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "SYNC_GATED ALL DONE"
