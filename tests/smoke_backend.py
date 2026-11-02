"""GPU smoke test (implementation step 4): the clean generate->invert round trip.

Reports, per radius band, how well ring signs survive SD generation + DDIM inversion with
NO attack -- the ceiling every later number is measured against.
Run on GPU:
  LD_PRELOAD=/opt/anaconda/lib/libstdc++.so.6 HF_HUB_OFFLINE=1 \
  /opt/anaconda/bin/python -u tests/smoke_backend.py [--n 2] [--steps 50]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.fourier_rings import enumerate_cells  # noqa: E402
from core.modulation import embed_bits, hard_bits, read_cells  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from utils.io import read_jsonl  # noqa: E402

PROMPTS_FALLBACK = ["a photograph of a white sailboat anchored in a calm bay, morning light",
                    "a photograph of three friends playing frisbee in a city park"]


def band(r: int) -> str:
    return "low(1-8)" if r <= 8 else ("mid(9-16)" if r <= 16 else ("high(17-24)" if r <= 24 else "top(25-31)"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--inv_steps", type=int, default=50)
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROOT / "data/prompts/sdp_1000.jsonl")
    prompts = [r["prompt"] for r in rows[:args.n]] or PROMPTS_FALLBACK[:args.n]

    t0 = time.time()
    be = SDBackend()
    print(f"[load] {time.time()-t0:.1f}s device={be.device} dtype={be.dtype}", flush=True)
    cells = enumerate_cells()
    rng = np.random.default_rng(0)
    agg = {}
    for i, prompt in enumerate(prompts):
        seed = 97 + i
        zT = SDBackend.sample_noise(seed)
        bits = rng.integers(0, 2, len(cells))
        zw = embed_bits(zT, cells, bits)
        t1 = time.time()
        img_w = be.generate(zw, prompt, steps=args.steps, seed=seed)
        t_gen = time.time() - t1
        img_w.save(out / f"smoke_{i}_wm.png")
        t1 = time.time()
        zhat = be.invert(img_w, steps=args.inv_steps).cpu()
        t_inv = time.time() - t1
        y = read_cells(zhat, cells)
        agree = (hard_bits(y) == bits)
        # null: unwatermarked generation from the same raw noise, same key read-out
        img_0 = be.generate(zT, prompt, steps=args.steps, seed=seed)
        img_0.save(out / f"smoke_{i}_clean.png")
        y0 = read_cells(be.invert(img_0, steps=args.inv_steps).cpu(), cells)
        agree0 = (hard_bits(y0) == bits)
        print(f"[img{i}] gen {t_gen:.1f}s inv {t_inv:.1f}s | sign-agreement wm={agree.mean():.3f} "
              f"null={agree0.mean():.3f} | mean|y| wm={np.abs(y).mean():.3f} null={np.abs(y0).mean():.3f}",
              flush=True)
        for cell, a, yy in zip(cells, agree, y):
            b = band(cell.r)
            agg.setdefault(b, []).append((a, abs(yy)))
        for c in range(4):
            sel = [k for k, cell in enumerate(cells) if cell.c == c]
            print(f"        ch{c}: agree={agree[sel].mean():.3f} mean|y|={np.abs(y[sel]).mean():.3f}", flush=True)
    print("\n=== clean round-trip survival by radius band (all channels, all images) ===")
    for b in ["low(1-8)", "mid(9-16)", "high(17-24)", "top(25-31)"]:
        v = np.array(agg[b])
        print(f"  {b:12s} sign-agreement={v[:,0].mean():.3f}  mean|y|={v[:,1].mean():.3f}  n={len(v)}")
    print("SMOKE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
