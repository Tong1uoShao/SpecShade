"""Stage 4: invert + extract for clean and attacked images of one method.

Reads results/{exp}/{method}/gen.jsonl (+ atk.jsonl), writes detect.jsonl (resume by image).
Usage (GPU): python -u src/detect.py --exp main --method stylering [--kind both] [--decode_variant soft]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import bits_to_hex, hex_to_bits  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import IMAGE_LEVEL, build_method  # noqa: E402
from utils.io import append_jsonl, done_keys, ensure_dir, load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--kind", choices=["clean", "atk", "both"], default="both")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--decode_variant", default=None)
    ap.add_argument("--cache_latents", action="store_true", help="save inverted latents (.pt) for decode ablations")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=8, help="images per batched DDIM inversion")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg["stylering"][k] = type(cfg["stylering"].get(k, v))(v) if k in cfg["stylering"] else v
    if args.decode_variant:
        cfg["stylering"]["decode_variant"] = args.decode_variant
    variant = cfg["stylering"].get("decode_variant", "soft")
    log = get_logger(f"detect_{args.exp}_{args.method}_{variant}", ROOT / "logs")
    base = ROOT / cfg["paths"]["results"] / args.exp / args.method
    det_path = base / (f"detect.jsonl" if variant == "soft" else f"detect_{variant}.jsonl")
    inv_dir = ensure_dir(base / "inv") if args.cache_latents else None

    gen = read_jsonl(base / "gen.jsonl")
    payload_of = {g["image"]: g["payload_hex"] for g in gen}
    jobs = []
    if args.kind in ("clean", "both"):
        jobs += [{"image": g["image"], "kind": "clean", "attack": "none", "strength": 0.0, "atk_seed": -1,
                  "payload_hex": g["payload_hex"]} for g in gen]
    if args.kind in ("atk", "both"):
        jobs += [{"image": a["image"], "kind": "atk", "attack": a["attack"], "strength": a["strength"],
                  "atk_seed": a["atk_seed"], "payload_hex": payload_of.get(a["src_image"])}
                 for a in read_jsonl(base / "atk.jsonl")]
    done = done_keys(det_path, "image")

    def _cache_of(j):
        return (inv_dir / (Path(j["image"]).stem + ".pt")) if inv_dir else None

    # with --cache_latents an already-detected image still needs its latent cached (decode ablations)
    todo = [j for j in jobs if j["image"] not in done or (inv_dir and not _cache_of(j).exists())]
    shard = os.environ.get("DETECT_SHARD")        # "i/n": this process takes jobs i, i+n, ... of the FULL job list
    if shard:
        i, n = map(int, shard.split("/"))
        pick = {j["image"] for idx, j in enumerate(jobs) if idx % n == i}
        todo = [j for j in todo if j["image"] in pick]
    log.info(f"exp={args.exp} method={args.method} variant={variant} jobs={len(jobs)} todo={len(todo)}")
    if not todo:
        return 0
    if args.method in IMAGE_LEVEL:                 # OURS: post-hoc / decoder methods read the image directly
        method = build_method(args.method, cfg, None)
        for k, j in enumerate(todo):
            t0 = time.time()
            truth = hex_to_bits(j["payload_hex"]) if j["payload_hex"] else None
            res = method.extract_image(Image.open(ROOT / j["image"]).convert("RGB"), truth)
            bit_acc = float((res.bits == truth).mean()) if (res.bits is not None and truth is not None) else None
            append_jsonl(det_path, {**j, "bits_dec_hex": bits_to_hex(res.bits) if res.bits is not None else None,
                                    "bit_acc": bit_acc, "accepted": res.accepted,
                                    "score": (None if (isinstance(res.score, float) and not np.isfinite(res.score)) else res.score),
                                    "n_corrected": res.n_corrected, "t_inv_s": round(time.time() - t0, 3)})
            if k % 200 == 0:
                log.info(f"[{k+1}/{len(todo)}] {j['kind']} {j['attack']} acc={bit_acc} score={res.score:.3f}")
        log.info("DETECT DONE")
        return 0
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"],
                   size=cfg["backend"]["size"])
    method = build_method(args.method, cfg, be)
    B = max(1, args.batch)
    for b0 in range(0, len(todo), B):
        batch = todo[b0:b0 + B]
        t0 = time.time()
        zhats: list[torch.Tensor | None] = [None] * len(batch)
        need = []
        for k, j in enumerate(batch):
            cache = _cache_of(j)
            if cache is not None and cache.exists():
                zhats[k] = torch.load(cache, map_location="cpu", weights_only=True)
            else:
                need.append(k)
        if need:
            ims = [Image.open(ROOT / batch[k]["image"]).convert("RGB") for k in need]
            Z = be.invert_batch(ims, steps=cfg["backend"]["inv_steps"]).cpu()
            for z, k in zip(Z, need):
                zhats[k] = z.unsqueeze(0)
                if inv_dir:
                    torch.save(zhats[k], inv_dir / (Path(batch[k]["image"]).stem + ".pt"))
        t_inv = round((time.time() - t0) / len(batch), 2)
        bit_acc, res = None, None
        for j, zhat in zip(batch, zhats):
            if j["image"] in done:        # cache-only pass: latent saved above, row already written
                continue
            truth = hex_to_bits(j["payload_hex"]) if j["payload_hex"] else None
            # methods whose official protocol scores against the known message get the reference (SFW)
            res = method.extract(zhat, ref_payload=truth) if getattr(method, "needs_ref", False) else method.extract(zhat)
            bit_acc = float((res.bits == truth).mean()) if (res.bits is not None and truth is not None) else None
            append_jsonl(det_path, {**j, "bits_dec_hex": bits_to_hex(res.bits) if res.bits is not None else None,
                                    "bit_acc": bit_acc, "accepted": res.accepted,
                                    "score": (None if (isinstance(res.score, float) and np.isnan(res.score)) else res.score),
                                    "n_corrected": res.n_corrected, "t_inv_s": t_inv,
                                    **({"meta": res.meta} if res.meta else {})})
        if (b0 // B) % 10 == 0:
            log.info(f"[{min(b0+B, len(todo))}/{len(todo)}] {batch[-1]['kind']} {batch[-1]['attack']} "
                     f"acc={bit_acc} accepted={res.accepted if res else 'cache-only'} ({t_inv}s/img)")
    log.info("DETECT DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
