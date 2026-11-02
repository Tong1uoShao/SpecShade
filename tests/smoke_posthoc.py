"""GPU smoke test for image-level baselines (stable_signature / vine): generate 2 images through
the real pipeline path, read the watermark back (expect bit acc ~1.0), and read an unwatermarked
image (expect ~0.5). usage: python tests/smoke_posthoc.py --method stable_signature [--n 2]"""
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
from models import build_method  # noqa: E402
from utils.io import load_config, read_jsonl  # noqa: E402
from utils.seeds import derive_payload, per_image_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    out = ROOT / "results" / "smoke_posthoc" / args.method
    out.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(ROOT / cfg["paths"]["prompts"])[:args.n]
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    m = build_method(args.method, cfg, be)
    m.prepare_backend(be)
    accs, scores = [], []
    for i, p in enumerate(prompts):
        seed = per_image_seed(97, i)
        payload = m.fixed_payload if m.fixed_payload is not None else derive_payload(p["prompt_id"], seed, m.n_bits)
        t0 = time.time()
        img = be.generate(m.embed(payload, seed), p["prompt"], steps=cfg["backend"]["gen_steps"], cfg=cfg["backend"]["cfg"], seed=seed)
        if m.image_level:
            img = m.embed_image(img, payload)
        img.save(out / f"{p['prompt_id']}.png")
        r = m.extract_image(img, payload)
        acc = float((r.bits == payload).mean())
        accs.append(acc); scores.append(r.score)
        print(f"{p['prompt_id']} gen+embed {time.time()-t0:.1f}s  bit_acc={acc:.3f} score={r.score:.3f}", flush=True)
    # unwatermarked control from the main experiment
    none = read_jsonl(ROOT / "results/main/none/gen.jsonl")[:args.n]
    for g in none:
        ref = m.fixed_payload if m.fixed_payload is not None else derive_payload(g["prompt_id"], g["gen_seed"], m.n_bits)
        r = m.extract_image(Image.open(ROOT / g["image"]).convert("RGB"), ref)
        print(f"none {g['prompt_id']} score={r.score:.3f}", flush=True)
    print(f"SMOKE {args.method}: mean bit_acc={np.mean(accs):.3f} mean score={np.mean(scores):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
