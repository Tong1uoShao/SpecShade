"""Cross-machine equivalence smoke: re-detect a few already-detected images with the same config and
compare score / bit accuracy against the recorded detect.jsonl rows (DDIM inversion is deterministic
per machine; expect |Δscore| small and identical decoded bits).
usage: python tests/smoke_migrate.py --exp abl_sec8 --sectors 8 --n 4"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import bits_to_hex  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config, read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="abl_sec8")
    ap.add_argument("--sectors", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    cfg["stylering"]["sectors"] = args.sectors
    rows = [r for r in read_jsonl(ROOT / "results" / args.exp / "stylering" / "detect.jsonl") if r["kind"] == "clean"][:args.n]
    rows += [r for r in read_jsonl(ROOT / "results" / args.exp / "stylering" / "detect.jsonl")
             if r["kind"] == "atk" and r["attack"] == "img2img_anime" and r["strength"] == 0.5][:args.n]
    rows = [r for r in rows if (ROOT / r["image"]).exists()]
    print(f"{len(rows)} images with recorded detections found locally")
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    m = build_method("stylering", cfg, be)
    worst = 0.0
    for r in rows:
        z = be.invert_batch([Image.open(ROOT / r["image"]).convert("RGB")], steps=cfg["backend"]["inv_steps"]).cpu()
        res = m.extract(z)
        s_old = r["score"] if r["score"] is not None else float("-inf")
        s_new = res.score if res.accepted else float("-inf")
        d = abs(s_new - s_old) if (s_old > -1e9 and s_new > -1e9) else (0.0 if s_old == s_new else float("inf"))
        worst = max(worst, d)
        same_bits = (bits_to_hex(res.bits) == r["bits_dec_hex"]) if res.bits is not None else (r["bits_dec_hex"] is None)
        print(f"{Path(r['image']).name:42s} old={s_old:8.3f} new={s_new:8.3f} dscore={d:.4f} bits_same={same_bits}")
    print(f"SMOKE_MIGRATE worst |dscore| = {worst:.4f}  ({'OK' if worst < 0.05 else 'CHECK'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
