#!/bin/bash
# Paper (256-bit main) batches B + C, chained after batch A and the GNR training.
#  B  gm256      FULL GaussMarker (official 256-bit recipe + trained GNR, one fixed message): 100 images x 2 seeds,
#                the 15 key cells (attacks.yaml key15), null = finalq/none (1000).
#  C1 final256   ours, the 41-cell main table re-detected with the ONE final detector (configs/final256.yaml),
#                null = finalq/none (1000).
#  C2 final_gs256q  GS-256 main table re-calibrated on the same 1000-image null (detections reused).
# usage: bash scripts/paper256_bc.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_bc_$(date +%Y%m%d_%H%M).log
SEEDS=97,101
mark "WAIT batch A + GNR"
until grep -q "PAPER256_A ALL DONE" logs/paper256_a_*.log 2>/dev/null && grep -q "GNR DONE" results/gnr/run.log 2>/dev/null; do
  if grep -q "STAGE .* FAIL" $(ls -t logs/paper256_a_*.log | head -1) || grep -q "GNR FAIL" results/gnr/run.log 2>/dev/null; then
    mark "UPSTREAM FAIL"; exit 1
  fi
  sleep 120
done
mark "UPSTREAM OK"

# ---- B: full GaussMarker
E=gm256; M=gaussmarker_full; C=configs/final_gm256.yaml
run_stage GEN_$E  $PY -u src/generate.py --exp $E --method $M --n 100 --config $C
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks key15 --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "GM256 DONE"

# ---- C1: ours, main table with the final detector
E=final256; M=stylering; C=configs/final256.yaml
mkdir -p results/$E/$M
cp results/final_fr256/$M/gen.jsonl results/final_fr256/$M/atk.jsonl results/$E/$M/
[ -e results/$E/$M/atk ] || ln -s ../../final_fr256/$M/atk results/$E/$M/atk
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc

# ---- C2: GS-256 main table, 1000-image null (same detections)
E=final_gs256q; M=gaussian_shading; C=configs/final_gs256.yaml
mkdir -p results/$E/$M
cp results/final_gs256/$M/gen.jsonl results/final_gs256/$M/atk.jsonl results/final_gs256/$M/detect.jsonl results/$E/$M/
[ -e results/$E/$M/atk ] || ln -s ../../final_gs256/$M/atk results/$E/$M/atk
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "PAPER256_BC ALL DONE"
