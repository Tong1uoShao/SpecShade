"""Smoke test (CPU, seconds): T2SMark port on synthetic latents -- aligned read recovers the message, the detection statistic of
a watermarked latent is far above that of an unwatermarked one, and a noisy latent still decodes most bits.
Usage: python tests/smoke_t2smark.py"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402


class FakeBackend:
    latent = 64

    @staticmethod
    def sample_noise(seed, latent=64, n_ch=4):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, n_ch, latent, latent, generator=g)


cfg = load_config(ROOT / "configs/final_t2s.yaml")
m = build_method("t2smark", cfg, FakeBackend())
print("codec: key r =", m.c_key.r, "k =", m.c_key.k, "| msg r =", m.c_msg.r, "k =", m.c_msg.k)
rng = np.random.default_rng(0)
payload = rng.integers(0, 2, m.n_bits)
z = m.embed(payload, 7)
print("embedded latent: shape", tuple(z.shape), "mean %.3f std %.3f" % (z.mean(), z.std()))
r = m.extract(z); print(f"aligned   acc={(r.bits == payload).mean():.3f} score={r.score:.1f}")
assert (r.bits == payload).mean() == 1.0
rn = m.extract(torch.randn(1, 4, 64, 64)); print(f"unwaterm. acc={(rn.bits == payload).mean():.3f} score={rn.score:.1f}")
zn = z + 1.2 * torch.randn_like(z)
r2 = m.extract(zn); print(f"noisy     acc={(r2.bits == payload).mean():.3f} score={r2.score:.1f}")
assert r.score > 3 * rn.score and (r2.bits == payload).mean() > 0.8
z2 = m.embed(payload, 8); assert not torch.equal(z, z2) and torch.equal(m.embed(payload, 7), z), "determinism per seed"
print("SMOKE OK")
