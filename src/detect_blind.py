"""Blind detection of externally edited images (user-made crops / rotations / anything) with the complete
method (frequency-domain GS + gated rotation/crop synchronisation). No attack parameters are used: each image
is centre-cropped to a square (if needed), resized to 512, inverted, synchronised and read. The recovered
angle / zoom / shift are printed so they can be compared with what the user actually did; the payload is
identified among the source images by bit agreement (traceability), and the score is compared with the
1% FPR threshold of the full-protocol run (final_fr).
usage: python src/detect_blind.py --images results/realworld/atk_fr [--sources results/full_rep/stylering/gen.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.ecc import hex_to_bits  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402


def to_square_512(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    if w != h:
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((512, 512), Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--config", default="configs/full_rep_blind.yaml")
    ap.add_argument("--sources", default="results/full_rep/stylering/gen.jsonl")
    ap.add_argument("--thr_table", default="results/final_fr/tables/main_no_ecc.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--src_dir", default=None)
    ap.add_argument("--pids", default="p0000,p0001,p0002")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    m = build_method("stylering", cfg, be)
    src = {json.loads(l)["prompt_id"]: hex_to_bits(json.loads(l)["payload_hex"]) for l in open(ROOT / a.sources)}
    if a.pids:                                                      # only the images that were handed out
        keep = set(a.pids.split(","))
        src = {k: v for k, v in src.items() if k in keep}
    elif a.src_dir and Path(a.src_dir).is_dir():
        keep = {q.stem for q in Path(a.src_dir).iterdir()}
        src = {k: v for k, v in src.items() if k in keep}
    thr = None
    p = ROOT / a.thr_table
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["cell"] == "clean" and r["method"] == "stylering":
                thr = float(r["thr_1fpr"])
    files = sorted(q for q in Path(a.images).iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    rows = []
    for q in files:
        im0 = Image.open(q)
        im = to_square_512(im0)
        z = be.invert_batch([im], steps=cfg["backend"]["inv_steps"]).cpu()
        res = m.extract(z)
        accs = {pid: float((res.bits == bits).mean()) for pid, bits in src.items()}
        best = max(accs, key=accs.get)
        meta = res.meta or {}
        row = {"file": q.name, "input_size": list(im0.size), "score": round(res.score, 4), "thr_1fpr": thr,
               "detected": (None if thr is None else bool(res.score >= thr)),
               "theta": meta.get("theta"), "rot_scale": meta.get("rot_scale"), "rot_gated": meta.get("gated"),
               "scale": meta.get("scale"), "ty": meta.get("ty"), "tx": meta.get("tx"), "crop_gated": meta.get("crop_gated"),
               "best_source": best, "bit_acc_best": round(accs[best], 4),
               "bit_acc_all": {k: round(v, 3) for k, v in accs.items()}}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
