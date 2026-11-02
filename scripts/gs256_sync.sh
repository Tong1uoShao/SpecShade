#!/bin/bash
# Control: Gaussian Shading 256 bit (official capacity, 64 copies/bit) + synchronisation searches, geometry sweep.
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/gs256_sync_$(date +%Y%m%d_%H%M).log
E=gs256_sync; M=gaussian_shading; C=configs/final_gs256_sync.yaml; SEEDS=97,101
mkdir -p results/$E/$M && cp results/final_gs256/$M/gen.jsonl results/$E/$M/
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks geometry --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "GS256_SYNC ALL DONE"
