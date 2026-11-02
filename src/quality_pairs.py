"""Paired fidelity metrics against the unwatermarked image of the same prompt/seed (PSNR / SSIM / LPIPS-free
proxy), plus mean CLIP-B/32 prompt consistency. Meaningful for post-hoc / decoder methods (VINE, Stable
Signature): their unwatermarked counterpart is exactly results/<exp>/none/gen/<pid>.png. For initial-noise
methods the image content changes with the noise, so PSNR/SSIM vs 'none' is reported but flagged N/A-by-design.
usage: python src/quality_pairs.py --exp finalq --methods none,stylering,vine,...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.io import read_jsonl  # noqa: E402

PAIRED_OK = {"vine", "stable_signature"}          # same latents -> pixel-comparable to 'none'


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a.mean(-1), b.mean(-1), data_range=255.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--methods", required=True)
    a = ap.parse_args()
    res = ROOT / "results" / a.exp
    none = {g["prompt_id"]: g for g in read_jsonl(res / "none" / "gen.jsonl")}
    out = {}
    for m in a.methods.split(","):
        if m == "none":
            continue
        gen = read_jsonl(res / m / "gen.jsonl")
        ps, ss = [], []
        for g in gen:
            ref = none.get(g["prompt_id"])
            if ref is None:
                continue
            x = np.asarray(Image.open(ROOT / g["image"]).convert("RGB"), np.float32)
            y = np.asarray(Image.open(ROOT / ref["image"]).convert("RGB").resize(Image.open(ROOT / g["image"]).size), np.float32)
            ps.append(psnr(x, y)); ss.append(ssim_gray(x, y))
        out[m] = {"n": len(ps), "psnr_vs_none": float(np.mean(ps)) if ps else None, "ssim_vs_none": float(np.mean(ss)) if ss else None,
                  "paired_meaningful": m in PAIRED_OK}
        print(f"{m:>18}: n={len(ps)} PSNR={out[m]['psnr_vs_none']:.2f} SSIM={out[m]['ssim_vs_none']:.4f} "
              f"({'paired' if m in PAIRED_OK else 'N/A by design: content changes with the noise'})")
    (res / "quality_pairs.json").write_text(json.dumps(out, indent=1))
    print("QUALITY_PAIRS DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
