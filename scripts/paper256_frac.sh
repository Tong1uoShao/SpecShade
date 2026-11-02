#!/bin/bash
# Non-integer rotation angles (7.5/22.5/37.5/67.5 deg, half-way between the 1-degree search grid points),
# 100 images x 2 attack seeds, final detector (configs/final256.yaml). The null is copied from geo256
# (same detector, same 1,000 finalq/none images). Detection runs in 3 disjoint shards (CPU-bound search).
# usage: nohup bash scripts/paper256_frac.sh &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_frac_$(date +%Y%m%d_%H%M).log
E=geo256f; M=stylering; C=configs/final256.yaml
mkdir -p results/$E/$M && cp results/final_fr256/$M/gen.jsonl results/$E/$M/
cp results/geo256/$M/detect_null_no_ecc.jsonl results/$E/$M/
run_stage ATK_$E $PY -u src/attack.py --exp $E --method $M --blocks geometry_frac --seeds 97,101 --config $C
mark "DET_$E 3 shards START"
for i in 0 1 2; do
  DETECT_SHARD=$i/3 $PY -u src/detect.py --exp $E --method $M --kind both --config $C >> logs/paper256_frac_shard$i.log 2>&1 &
done
wait
run_stage DET_${E}_REST $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "PAPER256_FRAC ALL DONE"
