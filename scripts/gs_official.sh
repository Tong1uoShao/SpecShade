#!/bin/bash
# Cross-check: official Gaussian Shading code (embed + decode) in our pipeline, 50 prompts x 2 seeds,
# anime style block + regeneration + geometry sweep; our re-implementation decodes the same latents
# with the official key (bit agreement recorded in detect meta).
# usage: [EXP=gs_official256 CFG=configs/final_gs_official256.yaml] bash scripts/gs_official.sh [N=50]
source "$(dirname "$0")/env.sh"
N=${1:-50}; SEEDS=97,101
LOG=$CODE/logs/${EXP:-gs_official}_$(date +%Y%m%d_%H%M).log
E=${EXP:-gs_official}; M=gs_official; C=${CFG:-configs/final_gs_official.yaml}
run_stage GEN_$E  $PY -u src/generate.py --exp $E --method $M --n $N --config $C
run_stage ATK_$E  $PY -u src/attack.py --exp $E --method $M --blocks style,regeneration,geometry --names img2img_anime adain_anime ip2p_anime regen_sd vae_cheng rotate crop_scale --seeds $SEEDS --config $C
run_stage DET_$E  $PY -u src/detect.py --exp $E --method $M --kind both --config $C
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp final --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "GS_OFFICIAL ALL DONE"
