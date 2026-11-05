"""Smoke test (GPU node, ~1 min): Gaussian Shading + SpecShade-style synchronisation (hard and soft vote) on synthetic
latents -- no diffusion model needed. Checks: aligned read, rotation, zoom+shift (crop), flip, joint fallback,
unwatermarked latent (all gated, no crash) and timings.   Usage: python tests/smoke_gs_sync.py"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.fourier_rings import rotate_latents, shift_latent, zoom_latents  # noqa: E402
from models.baseline.gaussian_shading import GaussianShadingWM  # noqa: E402
from utils.io import load_config  # noqa: E402


class FakeBackend:
    latent = 64

    def sample_noise(self, seed, latent=64):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, 4, latent, latent, generator=g)


def run(cfg_path):
    cfg = load_config(ROOT / cfg_path)
    gs = GaussianShadingWM(cfg, FakeBackend())
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, gs.n_bits)
    z = gs.embed(payload, 7)
    acc = lambda r: float((r.bits == payload).mean())
    print(f"\n=== {cfg_path}  soft_vote={gs.soft_vote} dev={gs.dev}")
    for name, fn in (("rot search", lambda: gs._sync_rot(z)), ("crop search", lambda: gs._sync_crop(z)),
                     ("joint search", lambda: gs._sync_joint(z)), ("refine", lambda: gs._refine(z, 0.9, 1.0, -1.0, 5.0))):
        fn(); torch.cuda.synchronize() if gs.dev.type == "cuda" else None
        t = time.time(); out = fn(); torch.cuda.synchronize() if gs.dev.type == "cuda" else None
        extra = out[1] if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict) else ""
        print(f"  [time] {name:12s} {time.time()-t:.2f}s  {({k: round(v, 2) if isinstance(v, float) else v for k, v in extra.items()} if extra else '')}")
    t = time.time(); r = gs.extract(z); print(f"aligned   acc={acc(r):.3f} score={r.score:.3f} meta={ {k: round(v, 2) if isinstance(v, float) else v for k, v in r.meta.items()} } ({time.time()-t:.2f}s)")
    assert acc(r) == 1.0
    t = time.time(); r = gs.extract(rotate_latents(z, [30.0])); print(f"rot 30    acc={acc(r):.3f} theta={r.meta['theta']} gated={r.meta['gated']} ({time.time()-t:.2f}s)")
    assert abs((r.meta["theta"] - 330.0 + 180) % 360 - 180) <= 1.5 and acc(r) > 0.85
    zc = shift_latent(zoom_latents(z, [1 / 0.866]), 3.0, -5.0)          # crop of area 0.75, off-centre
    t = time.time(); r = gs.extract(zc); print(f"crop .75  acc={acc(r):.3f} scale={r.meta['scale']:.3f} ty={r.meta['ty']:.2f} tx={r.meta['tx']:.2f} gated={r.meta['crop_gated']} ({time.time()-t:.2f}s)")
    assert abs(r.meta["scale"] - 0.866) <= 0.02 and acc(r) > 0.8
    t = time.time(); r = gs.extract(torch.flip(z, dims=[3])); print(f"mirror    acc={acc(r):.3f} flipped={r.meta['flipped']} ({time.time()-t:.2f}s)")
    assert r.meta["flipped"] and acc(r) > 0.95
    zj = shift_latent(rotate_latents(zoom_latents(z, [1 / 0.866]), [30.0]), 2.0, 4.0)   # crop + rotation
    t = time.time(); r = gs.extract(zj)
    print(f"crop+rot  acc={acc(r):.3f} fallback={r.meta.get('geo_fallback')} geo_theta={r.meta.get('geo_theta')} geo_scale={r.meta.get('geo_scale')} geo_z={round(r.meta.get('geo_z', float('nan')), 1)} gated={r.meta.get('geo_gated')} ({time.time()-t:.2f}s)")
    assert acc(r) > 0.7
    zn = torch.randn(1, 4, 64, 64)
    t = time.time(); r = gs.extract(zn)
    print(f"unwaterm. acc={acc(r):.3f} score={r.score:.3f} rot_gated={r.meta['gated']} crop_gated={r.meta['crop_gated']} fallback={r.meta.get('geo_fallback')} geo_z={round(r.meta.get('geo_z', float('nan')), 1)} geo_gated={r.meta.get('geo_gated')} ({time.time()-t:.2f}s)")
    assert r.meta.get("geo_gated", True) and r.score < 0.3


if __name__ == "__main__":
    run("configs/final_gs256_sync_gsh.yaml")
    run("configs/final_gs256_sync_gss.yaml")
    print("\nSMOKE OK")
