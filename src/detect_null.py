"""Null calibration: run a method's extractor on the UNWATERMARKED ('none') images of an
experiment -> results/{exp}/{method}/detect_null.jsonl (scores used for TPR@FPR thresholds).
Usage (GPU): python -u src/detect_null.py --exp main --method stylering [--decode_variant hard]
             [--none_exp main]   (ablation experiments reuse the 'none' images of the main exp)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.sd_backend import SDBackend  # noqa: E402
from models import IMAGE_LEVEL, build_method  # noqa: E402
from utils.io import append_jsonl, done_keys, load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seeds import derive_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--decode_variant", default=None, help="must match the detect.py run being calibrated")
    ap.add_argument("--none_exp", default=None, help="experiment whose 'none' images to decode (default: --exp)")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg["stylering"][k] = type(cfg["stylering"].get(k, v))(v) if k in cfg["stylering"] else v
    if args.decode_variant:
        cfg["stylering"]["decode_variant"] = args.decode_variant
    variant = cfg["stylering"].get("decode_variant", "soft")
    log = get_logger(f"null_{args.exp}_{args.method}_{variant}", ROOT / "logs")
    res = ROOT / cfg["paths"]["results"] / args.exp
    out = res / args.method / ("detect_null.jsonl" if variant == "soft" else f"detect_null_{variant}.jsonl")
    gen = read_jsonl(ROOT / cfg["paths"]["results"] / (args.none_exp or args.exp) / "none" / "gen.jsonl")
    done = done_keys(out, "image")
    todo = [g for g in gen if g["image"] not in done]
    log.info(f"null images={len(gen)} todo={len(todo)}")
    if not todo:
        return 0
    if args.method in IMAGE_LEVEL:                 # OURS: no inversion; score = agreement with the message
        method = build_method(args.method, cfg, None)   # the image *would* carry (fixed key or derived)
        for k, g in enumerate(todo):
            t0 = time.time()
            ref = method.fixed_payload if method.fixed_payload is not None else \
                derive_payload(g["prompt_id"], g["gen_seed"], method.n_bits)
            r = method.extract_image(Image.open(ROOT / g["image"]).convert("RGB"), ref)
            append_jsonl(out, {"image": g["image"], "score": (None if (isinstance(r.score, float) and not np.isfinite(r.score)) else r.score),
                               "accepted": r.accepted, "t_inv_s": round(time.time() - t0, 3)})
            if k % 40 == 0:
                log.info(f"[{k+1}/{len(todo)}] score={r.score:.3f}")
        log.info("NULL DONE")
        return 0
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    method = build_method(args.method, cfg, be)
    B = 8
    for b0 in range(0, len(todo), B):
        batch = todo[b0:b0 + B]
        t0 = time.time()
        Z = be.invert_batch([Image.open(ROOT / g["image"]).convert("RGB") for g in batch],
                            steps=cfg["backend"]["inv_steps"]).cpu()
        for g, z in zip(batch, Z):
            if getattr(method, "needs_ref", False):      # score against the key the image would carry (SFW protocol)
                ref = method.fixed_payload if method.fixed_payload is not None else \
                    derive_payload(g["prompt_id"], g["gen_seed"], method.n_bits)
                r = method.extract(z.unsqueeze(0), ref_payload=ref)
            else:
                r = method.extract(z.unsqueeze(0))
            append_jsonl(out, {"image": g["image"], "score": (None if (isinstance(r.score, float) and not np.isfinite(r.score)) else r.score),
                               "accepted": r.accepted, "t_inv_s": round((time.time() - t0) / len(batch), 2)})
        if (b0 // B) % 5 == 0:
            log.info(f"[{min(b0+B, len(todo))}/{len(todo)}] accepted={r.accepted} score={r.score}")
    log.info("NULL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
