"""Detect StyleRing watermarks in externally attacked images (e.g. style transfer done by a
third-party image model in a chat UI). # OURS: real-world attack check.

Input file names must be  <prompt_id>_<variant>[__<service>].<ext>
    variant  : k1 (main, full-ring) | k8 (abl_sec8) | chip (abl_sec0)
    service  : free tag (gpt / grok / gemini / ...), default 'unknown'
e.g. p0003_k8__gpt.png. The payload comes from the variant's gen.jsonl; the decision threshold
(TPR@1%FPR) from the variant's main_soft.csv (null distribution of the main experiment).
Non-square outputs are centre-cropped to a square before the 512x512 resize.

Usage (GPU): python -u src/detect_external.py --images results/realworld/atk --out results/realworld/detect_external.jsonl
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import bits_to_hex, hex_to_bits  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import append_jsonl, done_keys, load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402

VARIANTS = {"k1": ("main", 1), "k8": ("abl_sec8", 8), "chip": ("abl_sec0", 0)}
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_name(p: Path):
    stem = p.stem
    head, _, service = stem.partition("__")
    parts = head.split("_")
    if len(parts) < 2 or parts[1] not in VARIANTS:
        return None
    service = service or "_".join(parts[2:]) or "unknown"     # accepts p0001_k1__gpt and p0001_k1_gpt
    return parts[0], parts[1], service


def threshold_of(exp: str) -> float:
    with open(ROOT / "results" / exp / "tables" / "main_soft.csv") as f:
        for r in csv.DictReader(f):
            if r["method"] == "stylering" and r.get("thr_1fpr"):
                return float(r["thr_1fpr"])
    raise RuntimeError(f"no thr_1fpr for {exp}")


def square(im: Image.Image, tol: float = 0.02) -> tuple[Image.Image, bool]:
    w, h = im.size
    if abs(w - h) <= tol * max(w, h):
        return im, False
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of externally attacked images")
    ap.add_argument("--out", default="results/realworld/detect_external.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    log = get_logger("detect_external", ROOT / "logs")
    out = ROOT / args.out
    files = sorted(p for p in (ROOT / args.images).iterdir() if p.suffix.lower() in EXTS)
    jobs = []
    for p in files:
        meta = parse_name(p)
        if meta is None:
            log.info(f"skip (name not parsable): {p.name}")
            continue
        jobs.append({"image": str(p.relative_to(ROOT)), "pid": meta[0], "variant": meta[1], "service": meta[2]})
    done = done_keys(out, "image")
    todo = [j for j in jobs if j["image"] not in done]
    log.info(f"images={len(jobs)} todo={len(todo)}")
    if todo:
        payload, thr, methods = {}, {}, {}
        be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"],
                       size=cfg["backend"]["size"])
        for v, (exp, K) in VARIANTS.items():
            gen = read_jsonl(ROOT / "results" / exp / "stylering" / "gen.jsonl")
            payload[v] = {g["prompt_id"]: g["payload_hex"] for g in gen}
            thr[v] = threshold_of(exp)
            c = {**cfg, "stylering": {**cfg["stylering"], "sectors": K}}
            methods[v] = build_method("stylering", c, be)
        B = max(1, args.batch)
        for b0 in range(0, len(todo), B):
            batch = todo[b0:b0 + B]
            t0 = time.time()
            ims, crops, sizes = [], [], []
            for j in batch:
                im = Image.open(ROOT / j["image"]).convert("RGB")
                sizes.append(im.size)
                im, cropped = square(im)
                ims.append(im)
                crops.append(cropped)
            Z = be.invert_batch(ims, steps=cfg["backend"]["inv_steps"]).cpu()
            for j, z, sz, cr in zip(batch, Z, sizes, crops):
                v = j["variant"]
                res = methods[v].extract(z.unsqueeze(0))
                ph = payload[v].get(j["pid"])
                truth = hex_to_bits(ph) if ph else None
                acc = float((res.bits == truth).mean()) if (res.bits is not None and truth is not None) else None
                score = res.score if (res.score is not None and np.isfinite(res.score)) else None
                detected = bool(res.accepted and score is not None and score >= thr[v])
                append_jsonl(out, {**j, "orig_size": list(sz), "center_cropped": cr, "payload_hex": ph,
                                   "bits_dec_hex": bits_to_hex(res.bits) if res.bits is not None else None,
                                   "bit_acc": acc, "exact": bool(res.bits is not None and truth is not None and acc == 1.0),
                                   "accepted": res.accepted, "score": score, "thr_1fpr": thr[v], "detected": detected,
                                   "t_inv_s": round((time.time() - t0) / len(batch), 2)})
            log.info(f"[{min(b0 + B, len(todo))}/{len(todo)}]")
    # ---- summary ----
    rows = [r for r in read_jsonl(out) if r["image"] in {j["image"] for j in jobs}]
    g = defaultdict(list)
    for r in rows:
        g[(r["service"], r["variant"])].append(r)
    print(f"{'service':>10} {'variant':>7} {'n':>3} {'detect@1%':>9} {'exact':>6} {'bitacc':>7} {'score':>6} {'thr':>6}")
    for (s, v), rs in sorted(g.items(), key=lambda kv: (kv[0][0], list(VARIANTS).index(kv[0][1]))):
        accs = [r["bit_acc"] for r in rs if r["bit_acc"] is not None]
        scs = [r["score"] for r in rs if r["score"] is not None]
        print(f"{s:>10} {v:>7} {len(rs):>3} {np.mean([r['detected'] for r in rs]):>9.2f} "
              f"{np.mean([r['exact'] for r in rs]):>6.2f} {np.mean(accs) if accs else float('nan'):>7.3f} "
              f"{np.mean(scs) if scs else float('nan'):>6.2f} {rs[0]['thr_1fpr']:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
