#!/bin/bash
# False positives on EDITED unwatermarked images: 100 images of finalq/none under 9 key attack settings (regen x1/x2/x4,
# img2img 0.3/0.5, crop 0.75, rotation 75, mirror, img2img+crop), detected with the unified detector; the FPR per attack is
# computed against the unified threshold (results/final256u). usage: setsid nohup bash scripts/editnull_queue.sh &
source "$(dirname "$0")/env.sh"
LOG=$CODE/logs/editnull_$(date +%Y%m%d_%H%M).log
DONE=$CODE/results/.editnull_done; mkdir -p "$DONE"
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
mark "EDITNULL START"
C=configs/final256_unified.yaml
stage atk $PY -u src/attack.py --exp finalq --method none --blocks key15 --names regen_sd img2img_anime crop_scale rotate mirror combo_style_crop --seeds 97 --limit 100 --config $C
stage prep $PY - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in open("results/finalq/none/atk.jsonl")]
out = Path("results/finalq_atk/none"); out.mkdir(parents=True, exist_ok=True)
out.joinpath("gen.jsonl").write_text("".join(json.dumps({"prompt_id": f"{r['src_key']}_{r['attack']}_{r['strength']}", "gen_seed": 0,
                                                           "image": r["image"], "attack": r["attack"], "strength": r["strength"]}) + "\n" for r in rows))
print(len(rows), "attacked unwatermarked images")
PY
mkdir -p results/final256u_an/stylering && cp results/final256u/stylering/gen.jsonl results/final256u_an/stylering/gen.jsonl
stage det $PY -u src/detect_null.py --exp final256u_an --method stylering --none_exp finalq_atk --config $C
mark "EDITNULL ALL DONE"
