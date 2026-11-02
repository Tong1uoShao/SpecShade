#!/bin/bash
# Paper (256-bit main) batch C, after batches A and B: GS-256 on the 1000-image null, then the 41-cell main
# table re-detected with the final detector (null reused from geo256: same config, same finalq/none images).
#  B  gm256      FULL GaussMarker (official 256-bit recipe + trained GNR, one fixed message): 100 images x 2 seeds,
#                the 15 key cells (attacks.yaml key15), null = finalq/none (1000).
#  C1 final256   ours, the 41-cell main table re-detected with the ONE final detector (configs/final256.yaml),
#                null = finalq/none (1000).
#  C2 final_gs256q  GS-256 main table re-calibrated on the same 1000-image null (detections reused).
# usage: bash scripts/paper256_bc.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_c_$(date +%Y%m%d_%H%M).log
SEEDS=97,101
mark "START (independent of batch A: geo256 null and the reused detections are ready)"
E=final_gs256q; M=gaussian_shading; C=configs/final_gs256.yaml
mkdir -p results/$E/$M
cp results/final_gs256/$M/gen.jsonl results/final_gs256/$M/atk.jsonl results/final_gs256/$M/detect.jsonl results/$E/$M/
[ -e results/$E/$M/atk ] || ln -s ../../final_gs256/$M/atk results/$E/$M/atk
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
E=final256; M=stylering; C=configs/final256.yaml
mkdir -p results/$E/$M
cp results/final_fr256/$M/gen.jsonl results/final_fr256/$M/atk.jsonl results/$E/$M/
[ -e results/$E/$M/atk ] || ln -s ../../final_fr256/$M/atk results/$E/$M/atk
cp results/geo256/$M/detect_null_no_ecc.jsonl results/$E/$M/       # same detector + same null images as geo256
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "PAPER256_C ALL DONE"
