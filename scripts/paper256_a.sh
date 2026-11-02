#!/bin/bash
# Paper (256-bit main) batch A, on the 100 images x 2 attack seeds of the existing 256-bit tables:
#  A1 geo256    ours, final detector (configs/final256.yaml): rotation 15-90, crop 0.9/0.75/0.5, mirror h/v;
#               null = finalq/none (1000 unwatermarked images).
#  A2 combo256  ours, compound edits (style+crop, style+mirror, regen+rotate, crop+rotate) with the
#               compound-geometry mode (configs/final256_combo.yaml: joint search only as a fallback); null = final/none.
#  A3 GS-256 on mirror + compound edits: original decoder and + our sync; plus the original decoder on the
#               geometry sweep (re-detect of the gs256_sync attack images).
# usage: bash scripts/paper256_a.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_a_$(date +%Y%m%d_%H%M).log
SEEDS=97,101
link_atk(){  # link_atk <src_exp> <dst_exp> <method>: reuse attacked images, detect with another config
  mkdir -p results/$2/$3
  cp results/$1/$3/gen.jsonl results/$1/$3/atk.jsonl results/$2/$3/
  [ -e results/$2/$3/atk ] || ln -s ../../$1/$3/atk results/$2/$3/atk
}

# ---- A1: ours, geometry + mirror, final detector
E=geo256; M=stylering; C=configs/final256.yaml
mkdir -p results/$E/$M && cp results/final_fr256/$M/gen.jsonl results/$E/$M/
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks geometry,mirror --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc

# ---- A2: ours, compound edits, compound-geometry mode
E=combo256; C=configs/final256_combo.yaml
mkdir -p results/$E/$M && cp results/final_fr256/$M/gen.jsonl results/$E/$M/
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks combined --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc

# ---- A3: GS-256 (original decoder, then + our sync) on mirror + compound edits; original decoder on geometry
M=gaussian_shading
E=combo_gs256; C=configs/final_gs256.yaml
mkdir -p results/$E/$M && cp results/final_gs256/$M/gen.jsonl results/$E/$M/
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks mirror,combined --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
E=combo_gs256s; C=configs/final_gs256_sync.yaml
link_atk combo_gs256 $E $M
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
E=geo_gs256; C=configs/final_gs256.yaml
link_atk gs256_sync $E $M
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "PAPER256_A ALL DONE"
