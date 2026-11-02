"""Per-ring sign survival of the Fourier coefficients vs. the spatial sign survival, measured on SpecShade-256 images.

For every attacked image the inverted noise z_hat is compared with the watermarked noise z_T^w that generated it
(re-created from the stored payload and seed). Written to results/freq_survival/summary.json:
  per ring r = 1..44: fraction of embedded Fourier signs (real and imaginary parts, 4 channels) that are still read
  correctly (hard vote) and the magnitude-weighted fraction (what the weighted vote of Sec. III-C uses), and the same two
  numbers for the spatial signs of the latent (what Gaussian Shading reads).
Usage (GPU): python -u src/proto_freq_survival.py --exp final256 --n 50 --attacks none:0 regen_sd:2.0 img2img_anime:0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import hex_to_bits  # noqa: E402
from core.modulation import combine_positions, read_cells  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import ensure_dir, load_config, read_jsonl  # noqa: E402

N, R_MAX = 64, 44


def ring_masks():
    k = np.arange(N) - N // 2
    kx, ky = np.meshgrid(k, k, indexing="ij")
    r = np.rint(np.hypot(kx, ky)).astype(int)
    valid = (r >= 1) & (r <= R_MAX) & (kx != -N // 2) & (ky != -N // 2)      # no DC, no Nyquist row / column
    return r, valid


def survival(zr: torch.Tensor, zh: torch.Tensor, r: np.ndarray, valid: np.ndarray) -> dict:
    """zr, zh: [1,4,N,N]. Returns per-ring hard / weighted survival of the Fourier signs and the spatial equivalents."""
    fr = torch.fft.fftshift(torch.fft.fft2(zr.float()), dim=(-2, -1))[0]
    fh = torch.fft.fftshift(torch.fft.fft2(zh.float()), dim=(-2, -1))[0]
    a = torch.cat([fr.real, fr.imag]); b = torch.cat([fh.real, fh.imag])                  # [8,N,N]
    agree = (torch.sign(a) == torch.sign(b)).float().numpy()
    wgt = b.abs().numpy()
    hard, weighted = np.zeros(R_MAX + 1), np.zeros(R_MAX + 1)
    for ring in range(1, R_MAX + 1):
        m = valid & (r == ring)
        hard[ring] = agree[:, m].mean()
        weighted[ring] = (wgt[:, m] * agree[:, m]).sum() / wgt[:, m].sum()
    sp_agree = (torch.sign(zr) == torch.sign(zh)).float().numpy()
    sp_w = zh.abs().float().numpy()
    return {"hard": hard[1:], "weighted": weighted[1:],
            "all_hard": float(agree[:, valid].mean()), "all_weighted": float((wgt[:, valid] * agree[:, valid]).sum() / wgt[:, valid].sum()),
            "spatial_hard": float(sp_agree.mean()), "spatial_weighted": float((sp_w * sp_agree).sum() / sp_w.sum())}


def bit_accuracies(method, payload_bits, zr: torch.Tensor, zh: torch.Tensor) -> dict:
    """Bit accuracy of the same error field under two groupings of the votes (hard = majority, soft = magnitude-weighted):
    SpecShade (ring cells allocated to the payload positions, read by its own reader) and a Gaussian-Shading-style
    spatial tiling (bit = (channel, i mod 8, j mod 8), 64 copies, copies read as sign agreement with the embedded noise)."""
    code = np.asarray(method._code(np.asarray(payload_bits).astype(int))).astype(int)[:method._n_bits]
    out = {}
    for name, hard in (("f_hard", True), ("f_soft", False)):
        y = read_cells(zh.float(), method.alloc, method.chips, hard=hard)
        yp = combine_positions(y, method.pos_of, method._n_bits, method.sizes)
        out[name] = float(((yp > 0).astype(int) == code).mean())
    agree = (torch.sign(zr) == torch.sign(zh)).float().numpy()[0]            # [4,N,N]
    c, i, j = np.meshgrid(np.arange(4), np.arange(N), np.arange(N), indexing="ij")
    grp = (c * 64 + (i % 8) * 8 + (j % 8)).ravel()                           # 256 groups x 64 copies
    for name, w in (("s_hard", np.ones(agree.size)), ("s_soft", zh.abs().float().numpy()[0].ravel())):
        vs = np.bincount(grp, weights=w * (2 * agree.ravel() - 1), minlength=256)
        out[name] = float((vs > 0).mean() + 0.5 * (vs == 0).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="final256")
    ap.add_argument("--config", default="configs/final256.yaml")
    ap.add_argument("--n", type=int, default=50, help="first N prompts")
    ap.add_argument("--seed", type=int, default=97, help="attack seed")
    ap.add_argument("--attacks", nargs="+", default=["none:0", "regen_sd:2.0", "img2img_anime:0.5"])
    ap.add_argument("--out", default="results/freq_survival")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    base = ROOT / cfg["paths"]["results"] / args.exp / "stylering"
    gen = {g["prompt_id"]: g for g in read_jsonl(base / "gen.jsonl")}
    atk = read_jsonl(base / "atk.jsonl")
    pids = sorted(gen)[:args.n]
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    method = build_method("stylering", cfg, be)
    zref = {p: method.embed(hex_to_bits(gen[p]["payload_hex"]), gen[p]["gen_seed"]).cpu().float() for p in pids}
    r, valid = ring_masks()
    summary = {"n_prompts": len(pids), "rings": list(range(1, R_MAX + 1)), "attacks": {}}
    for spec in args.attacks:
        name, s = spec.split(":")
        if name == "none":
            items = [(p, ROOT / gen[p]["image"]) for p in pids]
        else:
            items = [(a["src_key"], ROOT / a["image"]) for a in atk if a["attack"] == name and abs(float(a["strength"]) - float(s)) < 1e-6
                     and a["atk_seed"] == args.seed and a["src_key"] in zref]
        acc = []
        for b0 in range(0, len(items), 8):
            batch = items[b0:b0 + 8]
            Z = be.invert_batch([Image.open(path).convert("RGB") for _, path in batch], steps=cfg["backend"]["inv_steps"]).cpu()
            for (p, _), z in zip(batch, Z):
                d = survival(zref[p], z.unsqueeze(0), r, valid)
                d.update(bit_accuracies(method, hex_to_bits(gen[p]["payload_hex"]), zref[p], z.unsqueeze(0)))
                acc.append(d)
        keys = ("hard", "weighted")
        entry = {"n": len(acc)}
        for k in keys:
            m = np.stack([x[k] for x in acc])
            entry[k] = m.mean(0).tolist(); entry[k + "_sem"] = (m.std(0, ddof=1) / np.sqrt(len(acc))).tolist()
        for k in ("all_hard", "all_weighted", "spatial_hard", "spatial_weighted", "f_hard", "f_soft", "s_hard", "s_soft"):
            entry[k] = float(np.mean([x[k] for x in acc]))
        summary["attacks"][spec] = entry
        print(spec, {k: round(v, 4) for k, v in entry.items() if k.startswith(("all_", "spatial_", "f_", "s_"))}, "n =", len(acc), flush=True)
    out = ensure_dir(ROOT / args.out)
    (out / "summary.json").write_text(json.dumps(summary))
    print("FREQ SURVIVAL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
