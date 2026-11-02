"""GPU check: batched inversion == per-image inversion, and its speed-up.
Run: LD_PRELOAD=... HF_HUB_OFFLINE=1 python tests/smoke_batch_invert.py [--batch 8]
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.sd_backend import SDBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
paths = sorted(glob.glob(str(ROOT / "results/probe_smoke/atk/*.png")))[:args.batch]
ims = [Image.open(p).convert("RGB") for p in paths]
be = SDBackend()
t0 = time.time()
singles = [be.invert(im, steps=50).cpu() for im in ims]
t_single = (time.time() - t0) / len(ims)
torch.cuda.synchronize()
t0 = time.time()
Z = be.invert_batch(ims, steps=50).cpu()
t_batch = (time.time() - t0) / len(ims)
diffs = [float((Z[i:i + 1] - singles[i]).abs().max()) for i in range(len(ims))]
scale = float(singles[0].abs().max())
print(f"n={len(ims)} single {t_single:.2f}s/img  batch {t_batch:.2f}s/img  speedup {t_single/t_batch:.2f}x")
print(f"max|diff| = {max(diffs):.4f} (latent scale {scale:.2f}) -> {'OK' if max(diffs) < 0.05 * scale else 'MISMATCH'}")
print("BATCH SMOKE DONE")
