#!/bin/bash
# Paper (256-bit main) batch B: full GaussMarker, starts as soon as the GNR is trained (in parallel with A).
#  B  gm256      FULL GaussMarker (official 256-bit recipe + trained GNR, one fixed message): 100 images x 2 seeds,
#                the 15 key cells (attacks.yaml key15), null = finalq/none (1000).
#  C1 final256   ours, the 41-cell main table re-detected with the ONE final detector (configs/final256.yaml),
#                null = finalq/none (1000).
#  C2 final_gs256q  GS-256 main table re-calibrated on the same 1000-image null (detections reused).
# usage: bash scripts/paper256_bc.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_b_$(date +%Y%m%d_%H%M).log
SEEDS=97,101
mark "WAIT GNR"
until grep -q "GNR DONE" results/gnr/run.log 2>/dev/null; do
  grep -q "GNR FAIL" results/gnr/run.log 2>/dev/null && { mark "GNR FAIL"; exit 1; }
  sleep 60
done
mark "GNR OK"
E=gm256; M=gaussmarker_full; C=configs/final_gm256.yaml
run_stage GEN_$E  $PY -u src/generate.py --exp $E --method $M --n 100 --config $C
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks key15 --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "PAPER256_B ALL DONE"
