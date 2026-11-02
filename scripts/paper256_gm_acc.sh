#!/bin/bash
# GaussMarker-256 with the bit-agreement-only score (fuse_lambda = 0): positives rescored offline from the stored
# acc_known (exactly the lambda = 0 score), null recomputed on the same 1,000 unwatermarked images.
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_gmacc_$(date +%Y%m%d_%H%M).log
E=gm256a; M=gaussmarker_full; C=configs/final_gm256_acc.yaml
mkdir -p results/$E/$M
cp results/gm256/$M/gen.jsonl results/gm256/$M/atk.jsonl results/$E/$M/
[ -e results/$E/$M/atk ] || ln -s ../../gm256/$M/atk results/$E/$M/atk
$PY - <<'PYEOF'
import json
src, dst = "results/gm256/gaussmarker_full/detect.jsonl", "results/gm256a/gaussmarker_full/detect.jsonl"
with open(dst, "w") as f:
    for l in open(src):
        r = json.loads(l); r["score"] = (r.get("meta") or {}).get("acc_known", r["score"]); f.write(json.dumps(r) + "\n")
PYEOF
mark "RESCORE OK"
run_stage NULL_$E $PY -u src/detect_null.py --exp $E --method $M --none_exp finalq --config $C
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C
mark "PAPER256_GMACC ALL DONE"
