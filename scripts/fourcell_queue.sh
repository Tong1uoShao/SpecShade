#!/bin/bash
# Domain x vote-rule comparison under GEOMETRIC and COMPOUND attacks with ONE synchronisation (reviewer P0):
#   GS-256 hard / soft vote + the same searches as SpecShade (configs/final_gs256_sync_gsh.yaml / _gss.yaml), on the
#   already attacked GS images (geo_gs256: 8 geometric cells; combo_gs256: mirror x2 + 4 compound cells);
#   SpecShade-256 soft (configs/final256_unified.yaml) and hard (configs/final256_hard_unified.yaml) cell reads on
#   the already attacked SpecShade images (geo256: 10 cells; combo256: 4 compound cells).
#   Null = finalq/none (1,000 images), inverted once (--cache_latents) and re-read by every detector.
# Then the compound-attack matrix (configs/attacks.yaml block combo_matrix, 18 cells x 100 images, unified detector).
# Resumable markers in results/.fourcell_done.   usage: setsid nohup bash scripts/fourcell_queue.sh > /dev/null 2>&1 &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/fourcell_queue_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.fourcell_done; mkdir -p "$DONE"
stage(){
  local name="$1"; shift
  if [ -e "$DONE/$name" ]; then mark "$name SKIP (done)"; return 0; fi
  for try in 1 2; do
    mark "$name START (try $try)"
    if "$@" >> "$LOG" 2>&1; then touch "$DONE/$name"; mark "$name OK"; return 0; fi
    mark "$name FAIL (try $try)"; sleep 60
  done
  return 1
}
mark "FOURCELL QUEUE START"
GEO8="rotate:15 rotate:30 rotate:45 rotate:75 rotate:90 crop_scale:0.9 crop_scale:0.75 crop_scale:0.5"
COMBO4="combo_style_crop:1 combo_style_mirror:1 combo_regen_rotate:1 combo_crop_rotate:1"
GS=gaussian_shading; CH=configs/final_gs256_sync_gsh.yaml; CS=configs/final_gs256_sync_gss.yaml
CU=configs/final256_unified.yaml; CHU=configs/final256_hard_unified.yaml

# ---- GS-256 + synchronisation, soft then hard (the hard pass re-uses the cached inverted latents) ----
stage prep_gs_geo   $PY scripts/filter_atk.py --src geo_gs256   --dst gsync_geo   --method $GS --cells $GEO8 --n_prompts 100
stage prep_gs_combo $PY scripts/filter_atk.py --src combo_gs256 --dst gsync_combo --method $GS --cells $COMBO4 mirror:0 mirror:1 --n_prompts 100
stage det_gss_geo   $PY -u src/detect.py --exp gsync_geo   --method $GS --kind both --config $CS --cache_latents
stage det_gss_combo $PY -u src/detect.py --exp gsync_combo --method $GS --kind atk  --config $CS --cache_latents
stage null_gss      $PY -u src/detect_null.py --exp gsync_geo --method $GS --none_exp finalq --config $CS --cache_latents
stage det_gsh_geo   $PY -u src/detect.py --exp gsync_geo   --method $GS --kind both --config $CH --cache_latents
stage det_gsh_combo $PY -u src/detect.py --exp gsync_combo --method $GS --kind atk  --config $CH --cache_latents
stage null_gsh      $PY -u src/detect_null.py --exp gsync_geo --method $GS --none_exp finalq --config $CH --cache_latents
cp results/gsync_geo/$GS/detect_null_gss.jsonl results/gsync_combo/$GS/ 2>/dev/null
cp results/gsync_geo/$GS/detect_null_gsh.jsonl results/gsync_combo/$GS/ 2>/dev/null
stage eval_gss_geo   $PY -u src/evaluate.py --exp gsync_geo   --methods $GS --config $CS --variant gss
stage eval_gss_combo $PY -u src/evaluate.py --exp gsync_combo --methods $GS --config $CS --variant gss
stage eval_gsh_geo   $PY -u src/evaluate.py --exp gsync_geo   --methods $GS --config $CH --variant gsh
stage eval_gsh_combo $PY -u src/evaluate.py --exp gsync_combo --methods $GS --config $CH --variant gsh

# ---- SpecShade-256, unified detector: soft (geometry block) and hard (geometry + compound) ----
GEO10="$GEO8 mirror:0 mirror:1"
stage prep_ours_geo    $PY scripts/filter_atk.py --src geo256   --dst geo256u    --cells $GEO10 --n_prompts 100
stage prep_ours_geoh   $PY scripts/filter_atk.py --src geo256   --dst geo256hu   --cells $GEO10 --n_prompts 100
stage prep_ours_comboh $PY scripts/filter_atk.py --src combo256 --dst combo256hu --cells $COMBO4 --n_prompts 100
stage det_geo256u  $PY -u src/detect.py --exp geo256u --method stylering --kind both --config $CU --cache_latents
[ -e results/geo256hu/stylering/inv ] || ln -s ../../geo256u/stylering/inv results/geo256hu/stylering/inv
stage det_geo256hu   $PY -u src/detect.py --exp geo256hu   --method stylering --kind both --config $CHU --cache_latents
stage det_combo256hu $PY -u src/detect.py --exp combo256hu --method stylering --kind atk  --config $CHU
stage null_hu        $PY -u src/detect_null.py --exp geo256hu --method stylering --none_exp finalq --config $CHU --cache_latents
cp results/final256u/stylering/detect_null_no_ecc.jsonl results/geo256u/stylering/ 2>/dev/null
cp results/geo256hu/stylering/detect_null_no_ecc.jsonl  results/combo256hu/stylering/ 2>/dev/null
stage eval_geo256u    $PY -u src/evaluate.py --exp geo256u    --methods stylering --config $CU  --variant no_ecc
stage eval_geo256hu   $PY -u src/evaluate.py --exp geo256hu   --methods stylering --config $CHU --variant no_ecc
stage eval_combo256hu $PY -u src/evaluate.py --exp combo256hu --methods stylering --config $CHU --variant no_ecc
mark "FOURCELL DONE"

# ---- compound-attack matrix: regeneration strength x geometric strength (SpecShade-256, unified detector) ----
mkdir -p results/cmat256/stylering && [ -e results/cmat256/stylering/gen.jsonl ] || cp results/final256/stylering/gen.jsonl results/cmat256/stylering/
stage atk_cmat  $PY -u src/attack.py --exp cmat256 --method stylering --blocks combo_matrix --seeds 97 --config $CU
stage det_cmat  $PY -u src/detect.py --exp cmat256 --method stylering --kind atk --config $CU --cache_latents
cp results/final256u/stylering/detect_null_no_ecc.jsonl results/cmat256/stylering/ 2>/dev/null
stage eval_cmat $PY -u src/evaluate.py --exp cmat256 --methods stylering --config $CU --variant no_ecc
mark "FOURCELL QUEUE ALL DONE"
