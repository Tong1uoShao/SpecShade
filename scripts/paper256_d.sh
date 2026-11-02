#!/bin/bash
# Paper batch D (after batch A): re-calibrate the reference methods and the 16-bit runs on the same 1,000-image
# null (finalq/none) as the 256-bit methods. Detections are reused; only the null scores and tables are recomputed.
#  final_refq   RingID, Stable Signature, VINE-R, GS-16 (configs/default.yaml, as in exp final)
#  final_fr2q   SpecShade-16 v2 (configs/full_rep_sync2.yaml, as in exp final_fr2)
# usage: bash scripts/paper256_d.sh
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_d_$(date +%Y%m%d_%H%M).log
mark "START (independent of batch A: geo256 null and the reused detections are ready)"
reuse(){  # reuse <src_exp> <dst_exp> <method> <detect file>
  mkdir -p results/$2/$3
  cp results/$1/$3/gen.jsonl results/$1/$3/atk.jsonl results/$1/$3/$4 results/$2/$3/
  [ -e results/$2/$3/atk ] || ln -s ../../$1/$3/atk results/$2/$3/atk
}
E=final_refq; C=configs/default.yaml
for M in gaussian_shading ringid stable_signature vine; do
  reuse final $E $M detect.jsonl
  run_stage NULL_${E}_$M $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
done
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods gaussian_shading,ringid,stable_signature,vine --config $C
E=final_fr2q; M=stylering; C=configs/full_rep_sync2.yaml
reuse final_fr2 $E $M detect_no_ecc.jsonl
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "PAPER256_D ALL DONE"
