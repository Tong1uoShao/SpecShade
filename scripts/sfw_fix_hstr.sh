#!/bin/bash
# (1) Finish the HSTR attack settings skipped by the batching bug (vae, jpeg, blur, rotate, crop, mirror, combo), re-detect
#     the new images (resume) and re-evaluate the identification-protocol files (detect.jsonl).
# (2) Re-detect both SFW variants under the official verification protocol (score = -distance to the known key):
#     variant 'known' -> detect_known.jsonl / detect_null_known.jsonl / tables/main_known.csv.
# Runs after the SFW queue; marks results/.sfw_done/fix_hstr (the MaXsive queue waits for it).
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/sfw_fix_hstr_$(date +%Y%m%d_%H%M).log
E=final_sfw; C=configs/final_sfw.yaml
mark "FIX HSTR START"
M=sfw_hstr
$PY -u src/attack.py --exp $E --method $M --blocks key15 --seeds 97,101 --config $C >> "$LOG" 2>&1 && mark "atk OK" || mark "atk FAIL"
$PY -u src/detect.py --exp $E --method $M --kind both --config $C >> "$LOG" 2>&1 && mark "det OK" || mark "det FAIL"
$PY -u src/evaluate.py --exp $E --methods sfw_hstr,sfw_hsqr --config $C --variant soft >> "$LOG" 2>&1 && mark "eval OK" || mark "eval FAIL"
for M in sfw_hstr sfw_hsqr; do
  $PY -u src/detect.py --exp $E --method $M --kind both --decode_variant known --config $C >> "$LOG" 2>&1 && mark "det_known_$M OK" || mark "det_known_$M FAIL"
  $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --decode_variant known --config $C >> "$LOG" 2>&1 && mark "null_known_$M OK" || mark "null_known_$M FAIL"
done
$PY -u src/evaluate.py --exp $E --methods sfw_hstr,sfw_hsqr --config $C --variant known >> "$LOG" 2>&1 && mark "eval_known OK" || mark "eval_known FAIL"
touch results/.sfw_done/fix_hstr
mark "FIX HSTR DONE"
