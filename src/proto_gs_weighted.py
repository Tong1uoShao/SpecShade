"""Route B check: attack-aware frequency weighting of the inverted latent before Gaussian Shading's
sign vote. Weight per (channel, ring) comes from the offline survival map S (sign-survival rate of
that frequency cell under the style attacks); frequencies outside the measured rings (|f| > r_max)
get w_out. For every attacked GS image we invert once, then decode (a) baseline and (b) filtered
variants, and report per-element survival p_e, per-bit accuracy and exact-decode rate at the run's
capacity (use exp final_gs256 for the 256-bit case).
usage: python src/proto_gs_weighted.py --exp final_gs256 --config configs/final_gs256.yaml --n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.allocation import SurvivalMap  # noqa: E402
from core.ecc import hex_to_bits  # noqa: E402
from core.fourier_rings import ring_mask  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config, read_jsonl  # noqa: E402

CELLS = "clean,img2img_anime@0.3,img2img_anime@0.5,img2img_anime@0.7,regen_sd@2.0,adain_anime@1.0,ip2p_anime@1.0,blur@8.0,jpeg@25.0"


def weight_maps(smap: SurvivalMap, latent: int, gamma: float, w_out: float, attack_subset=None) -> np.ndarray:
    """(4, latent, latent) weights in the centred FFT layout: w = (2S-1)^gamma per ring, w_out beyond r_max."""
    S = smap.scores(attack_subset)
    W = np.full((4, latent, latent), w_out, dtype=np.float32)
    for cell, s in zip(smap.cells, S):
        if cell.q is not None:
            continue
        m = ring_mask(latent, latent, cell.r)
        W[cell.c][m] = max(2.0 * float(s) - 1.0, 0.0) ** gamma
    W[:, latent // 2, latent // 2] = 1.0                                   # DC untouched
    return W


def filt(z: torch.Tensor, W: np.ndarray) -> torch.Tensor:
    F = torch.fft.fftshift(torch.fft.fft2(z[0].float()), dim=(-2, -1))
    F = F * torch.from_numpy(W)
    return torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1))).real.unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="final_gs256")
    ap.add_argument("--config", default="configs/final_gs256.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cells", default=CELLS)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    m = build_method("gaussian_shading", cfg, be)
    smap = SurvivalMap.load(str(ROOT / cfg["paths"]["survival_map"]))
    variants = {"baseline": None,
                "w(2S-1)^1,out=1": weight_maps(smap, be.latent, 1.0, 1.0),
                "w(2S-1)^1,out=0": weight_maps(smap, be.latent, 1.0, 0.0),
                "w(2S-1)^2,out=0": weight_maps(smap, be.latent, 2.0, 0.0),
                "w=1,out=0 (drop corners)": weight_maps(smap, be.latent, 0.0, 0.0),
                "w(2S-1)^0.5,out=0.3": weight_maps(smap, be.latent, 0.5, 0.3)}
    base = ROOT / "results" / a.exp / "gaussian_shading"
    gen = {g["image"]: g for g in read_jsonl(base / "gen.jsonl")}
    atk = read_jsonl(base / "atk.jsonl")
    print(f"exp={a.exp} n_bits={m.n_bits} copies/bit={m.ch*m.hw*m.hw}")
    print(f"{'cell':>20} | {'variant':>26} | {'p_e':>6} | {'bitacc':>6} | {'exact':>6}")
    for cell in a.cells.split(","):
        if cell == "clean":
            rows = [{"image": g["image"], "payload_hex": g["payload_hex"]} for g in list(gen.values())[: a.n]]
        else:
            name, s = cell.split("@")
            rows = [{"image": r["image"], "payload_hex": gen[r["src_image"]]["payload_hex"]}
                    for r in atk if r["attack"] == name and float(r["strength"]) == float(s)][: a.n]
        stats = {v: [] for v in variants}
        for r in rows:
            z = be.invert_batch([Image.open(ROOT / r["image"]).convert("RGB")], steps=cfg["backend"]["inv_steps"]).cpu()
            payload = hex_to_bits(r["payload_hex"])
            sd_true = m._tile(payload)
            for v, W in variants.items():
                zz = z if W is None else filt(z, W)
                m_hat = (zz.numpy() > 0).astype(int)
                sd_hat = (m_hat + m.keystream) % 2
                pe = float((sd_hat == sd_true).mean())
                bits, _ = m._vote(sd_hat)
                acc = float((bits == payload).mean())
                stats[v].append((pe, acc, float(acc == 1.0)))
        for v in variants:
            arr = np.array(stats[v])
            print(f"{cell:>20} | {v:>26} | {arr[:,0].mean():6.3f} | {arr[:,1].mean():6.3f} | {arr[:,2].mean():6.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
