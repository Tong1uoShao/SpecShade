"""Diagnostic: per-element sign survival p_e of Gaussian Shading under our attacks, and what majority
voting over n copies predicts (256 bit -> 64 copies, 16 bit -> 1024 copies). Reads the GS images of an
experiment, inverts a sample per attack cell, and compares sign(z_hat) XOR key with the tiled message.
usage: python src/proto_gs_survival.py --exp final --n 20 --cells clean,img2img_anime@0.5,regen_sd@2.0
"""
from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.ecc import hex_to_bits  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config, read_jsonl  # noqa: E402


def maj_acc(pe: float, n: int) -> float:
    """P(majority of n i.i.d. votes with per-vote accuracy pe is correct), normal approximation."""
    if n <= 1:
        return pe
    z = (pe - 0.5) * sqrt(n) / sqrt(pe * (1 - pe) + 1e-12)
    return 0.5 * (1 + erf(z / sqrt(2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="final")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cells", default="clean,img2img_anime@0.3,img2img_anime@0.5,img2img_anime@0.7,regen_sd@1.0,regen_sd@2.0,regen_sd@4.0,adain_anime@1.0")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    base = ROOT / "results" / a.exp / "gaussian_shading"
    gen = {g["image"]: g for g in read_jsonl(base / "gen.jsonl")}
    atk = read_jsonl(base / "atk.jsonl")
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    m = build_method("gaussian_shading", cfg, be)
    for cell in a.cells.split(","):
        if cell == "clean":
            rows = [{"image": g["image"], "payload_hex": g["payload_hex"]} for g in list(gen.values())[: a.n]]
        else:
            name, s = cell.split("@")
            rows = [{"image": r["image"], "payload_hex": gen[r["src_image"]]["payload_hex"]}
                    for r in atk if r["attack"] == name and float(r["strength"]) == float(s)][: a.n]
        pes = []
        for r in rows:
            z = be.invert_batch([Image.open(ROOT / r["image"]).convert("RGB")], steps=cfg["backend"]["inv_steps"]).cpu()
            m_hat = (z.numpy() > 0).astype(int)
            sd_hat = (m_hat + m.keystream) % 2
            sd_true = m._tile(hex_to_bits(r["payload_hex"]))
            pes.append(float((sd_hat == sd_true).mean()))
        pe = float(np.mean(pes))
        print(f"{cell:>20}: n={len(pes)} per-element survival p_e={pe:.3f} (min {min(pes):.3f}, max {max(pes):.3f}) "
              f"-> predicted bit acc: 64 copies {maj_acc(pe, 64):.3f} | 256 copies {maj_acc(pe, 256):.3f} | 1024 copies {maj_acc(pe, 1024):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
