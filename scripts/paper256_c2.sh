#!/bin/bash
# Batch C continuation: the final256 re-detection was CPU-bound (synchronization search on CPU, GPU ~5% busy), so the
# remaining images are split over 3 parallel detect processes (DETECT_SHARD=i/3 on the full job list, disjoint), then
# one unsharded pass catches anything missed, duplicates from the earlier overlap are dropped, and the table is
# evaluated. Continues the log of batch C (same end marker "PAPER256_C ALL DONE").
# usage: nohup bash scripts/paper256_c2.sh &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/paper256_c_20261008_1038.log
E=final256; M=stylering; C=configs/final256.yaml
mark "DET_$E RESUMED as 3 shards"
for i in 0 1 2; do
  DETECT_SHARD=$i/3 $PY -u src/detect.py --exp $E --method $M --kind both --config $C >> logs/paper256_c2_shard$i.log 2>&1 &
done
wait
run_stage DET_${E}_REST $PY -u src/detect.py --exp $E --method $M --kind both --config $C
$PY - results/$E/$M/detect_no_ecc.jsonl >> "$LOG" 2>&1 <<"PYEOF"
import json, sys
p = sys.argv[1]; seen = set(); out = []
for l in open(p):
    k = json.loads(l)["image"]
    if k not in seen:
        seen.add(k); out.append(l)
n0 = sum(1 for _ in open(p)); open(p, "w").writelines(out); print("dedup", n0, "->", len(out))
PYEOF
run_stage EVAL_$E $PY -u src/evaluate.py --exp $E --methods $M --config $C --variant no_ecc
mark "PAPER256_C ALL DONE"
mark "DEDUP DONE"
