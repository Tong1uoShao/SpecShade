#!/bin/bash
# Extra geometry checks on the v2 method (first N images of final_fr2):
#  (1) mirror (horizontal / vertical) with the flip candidate; (2) crop sweep: how small can a random crop get.
# usage: bash scripts/geo_extra.sh [N=50]
source "$(dirname "$0")/env.sh"
N=${1:-50}; M=stylering
LOG=$CODE/logs/geo_extra_$(date +%Y%m%d_%H%M).log
one(){  # one <exp> <block> <seeds> <config>
  local E=$1 B=$2 SD=$3 C=$4
  mkdir -p results/$E/$M && head -n $N results/final_fr2/$M/gen.jsonl > results/$E/$M/gen.jsonl
  run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks $B --seeds $SD --config $C
  run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
  run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
  run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
}
one fr2_mirror   mirror     97     configs/full_rep_sync2_flip.yaml
one fr2_cropmin  crop_sweep 97,101 configs/full_rep_sync2_deepcrop.yaml
mark "GEO_EXTRA ALL DONE"
