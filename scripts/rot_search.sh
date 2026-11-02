#!/bin/bash
# Rotation synchronisation search on the geometry sweep (rotate 15/30/45/75/90 deg, crop 0.9/0.75/0.5):
#  (1) full_rep images (frequency-domain GS), detected WITH the search (exp full_rep_rot) and WITHOUT it on the
#      same cached latents (exp full_rep_rot0);
#  (2) the final-table K=8 images (100 x 2), detected with the search (exp rot_k8).
# usage: bash scripts/rot_search.sh
source "$(dirname "$0")/env.sh"
SEEDS=97,101; M=stylering
LOG=$CODE/logs/rot_search_$(date +%Y%m%d_%H%M).log
E=full_rep_rot; C=configs/full_rep_rot.yaml
mkdir -p results/$E/$M && cp results/full_rep/$M/gen.jsonl results/$E/$M/
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks geometry --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C --cache_latents
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
E0=full_rep_rot0; C0=configs/full_rep.yaml
mkdir -p results/$E0/$M && cp results/$E/$M/gen.jsonl results/$E/$M/atk.jsonl results/$E0/$M/ && ln -sfn ../../$E/$M/inv results/$E0/$M/inv
run_stage DET_$E0  $PY -u src/detect.py --exp $E0 --method $M --kind both --config $C0 --cache_latents
run_stage NULL_$E0 $PY -u src/detect_null.py --exp $E0 --method $M --none_exp final --config $C0
run_stage EVAL_$E0 $PY -u src/evaluate.py --exp $E0 --methods $M --config $C0 --variant no_ecc
E8=rot_k8; C8=configs/default.yaml
mkdir -p results/$E8/$M && cp results/final/$M/gen.jsonl results/$E8/$M/
run_stage ATK_$E8  $PY -u src/attack.py --exp $E8 --method $M --blocks geometry --seeds $SEEDS --config $C8
run_stage DET_$E8  $PY -u src/detect.py --exp $E8 --method $M --kind both --config $C8 --override rot_search=1
run_stage NULL_$E8 $PY -u src/detect_null.py --exp $E8 --method $M --none_exp final --config $C8 --override rot_search=1
run_stage EVAL_$E8 $PY -u src/evaluate.py --exp $E8 --methods $M --config $C8
mark "ROT_SEARCH ALL DONE"
