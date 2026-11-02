#!/bin/bash
# The complete method at the paper protocol: frequency-domain GS (full-budget chip carrier, magnitude-weighted
# votes, repetition) + gated rotation/crop synchronisation at detection (configs/full_rep_sync.yaml).
# Robustness: N prompts x 2 attack seeds, all 41 cells (style + classical + regeneration), null = final/none.
# Quality: N_Q images -> exp finalq_fr (reuses finalq/none) -> FID / CLIP.
# usage: [EXP=final_fr256 CFG=configs/full_rep256_sync.yaml QEXP=finalq_fr256] bash scripts/final_full_rep.sh [N=100] [N_Q=1000]
source "$(dirname "$0")/env.sh"
N=${1:-100}; NQ=${2:-1000}; SEEDS=97,101; BLOCKS=style,classical,regeneration
LOG=$CODE/logs/${EXP:-final_fr}_$(date +%Y%m%d_%H%M).log
E=${EXP:-final_fr}; M=stylering; C=${CFG:-configs/full_rep_sync.yaml}; QE=${QEXP:-finalq_fr}
run_stage GEN_$E  $PY -u src/generate.py --exp $E --method $M --n $N --config $C
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks $BLOCKS --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "ROBUSTNESS_$E DONE"
mkdir -p results/$QE && [ -e results/$QE/none ] || ln -s ../finalq/none results/$QE/none
run_stage GENQ_$E    $PY -u src/generate.py --exp $QE --method $M --n $NQ --config $C
run_stage QUALITY_$E $PY -u src/evaluate.py --exp $QE --methods none,$M --quality --config $C
mark "FINAL_$E ALL DONE"
