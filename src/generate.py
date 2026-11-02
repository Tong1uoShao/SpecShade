"""Stage 2 of the pipeline: generate watermarked images for one method.

results/{exp}/{method}/gen/{prompt_id}.png + gen.jsonl (resume by prompt_id).
Usage (GPU): python -u src/generate.py --exp main --method stylering --n 200 [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import bits_to_hex  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import append_jsonl, done_keys, ensure_dir, load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seeds import derive_payload, per_image_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gen_seed_base", type=int, default=97)
    ap.add_argument("--override", nargs="*", default=[], help="key=value for stylering.* (ablations)")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg["stylering"][k] = type(cfg["stylering"].get(k, v))(v) if k in cfg["stylering"] else v
    log = get_logger(f"generate_{args.exp}_{args.method}", ROOT / "logs")
    out = ensure_dir(ROOT / cfg["paths"]["results"] / args.exp / args.method / "gen")
    meta_path = out.parent / "gen.jsonl"
    prompts = read_jsonl(ROOT / cfg["paths"]["prompts"])[:args.n]
    done = done_keys(meta_path, "prompt_id")
    todo = [p for p in prompts if p["prompt_id"] not in done]
    log.info(f"exp={args.exp} method={args.method} n={len(prompts)} todo={len(todo)}")
    if not todo:
        return 0
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"],
                   size=cfg["backend"]["size"])
    method = build_method(args.method, cfg, be)
    method.prepare_backend(be)                     # no-op except e.g. Stable Signature (fine-tuned VAE decoder)
    for i, p in enumerate(prompts):
        if p["prompt_id"] in done:
            continue
        seed = per_image_seed(args.gen_seed_base, i)
        if method.fixed_payload is not None:       # single baked-in key
            payload = method.fixed_payload
        else:
            payload = derive_payload(p["prompt_id"], seed, method.n_bits) if method.n_bits else None
        t0 = time.time()
        zT = method.embed(payload, seed)
        img = be.generate(zT, p["prompt"], steps=cfg["backend"]["gen_steps"], cfg=cfg["backend"]["cfg"], seed=seed)
        if method.image_level:
            img = method.embed_image(img, payload)  # post-hoc encoders watermark the generated image
        img_path = out / f"{p['prompt_id']}.png"
        img.save(img_path)
        append_jsonl(meta_path, {"prompt_id": p["prompt_id"], "prompt": p["prompt"], "gen_seed": seed,
                                 "payload_hex": bits_to_hex(payload) if payload is not None else None,
                                 "image": str(img_path.relative_to(ROOT)), "t_gen_s": round(time.time() - t0, 2),
                                 "method": args.method})
        if i % 10 == 0:
            log.info(f"[{i+1}/{len(prompts)}] {p['prompt_id']} {time.time()-t0:.1f}s")
    log.info("GENERATE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
