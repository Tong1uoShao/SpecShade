"""Gaussian Shading, OFFICIAL implementation (vendor/Gaussian-Shading/watermark.py::Gaussian_Shading)
run inside our pipeline: the official class does the sampling (truncSampling), the key/vote arithmetic
and the decoding; we only (a) fix the key deterministically from the config key, (b) inject the
payload bits the protocol derives per image, and (c) move tensors to/from the backend.

# OURS: cross-check of our re-implementation (baseline/gaussian_shading.py): extract() also decodes the
#   same latent with our class using the official key and records the bit agreement in meta.
Requires CUDA (the official code hard-codes .cuda()).
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

from models.base import ExtractResult, WatermarkMethod

_VENDOR = Path(__file__).resolve().parent / "vendor" / "Gaussian-Shading"
_FACTORS = {16: (4, 16), 32: (2, 16), 64: (1, 16), 256: (1, 8)}


def _import_official():
    """Import the official watermark module; stub pycryptodome if absent (only the ChaCha variant needs it)."""
    try:
        import Crypto.Cipher.ChaCha20  # noqa: F401
    except Exception:
        crypto = types.ModuleType("Crypto")
        cipher = types.ModuleType("Crypto.Cipher")
        chacha = types.ModuleType("Crypto.Cipher.ChaCha20")
        rnd = types.ModuleType("Crypto.Random")
        rnd.get_random_bytes = lambda n: b"\0" * n
        cipher.ChaCha20 = chacha
        crypto.Cipher, crypto.Random = cipher, rnd
        sys.modules.update({"Crypto": crypto, "Crypto.Cipher": cipher, "Crypto.Cipher.ChaCha20": chacha, "Crypto.Random": rnd})
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    import watermark as wm  # noqa: E402  (vendor module)
    return wm


class GaussianShadingOfficialWM(WatermarkMethod):
    name = "gs_official"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        if not torch.cuda.is_available():
            raise RuntimeError("gs_official needs CUDA (official code hard-codes .cuda())")
        m = cfg.get("methods", {}).get("gs_official", {})
        self._n_bits = int(m.get("n_bits", cfg["stylering"]["n_bits"]))
        self.ch, self.hw = _FACTORS[self._n_bits] if "ch" not in m else (int(m["ch"]), int(m["hw"]))
        self.latent = backend.latent
        assert self.latent == 64, "official code is written for 64x64 latents"
        wm = _import_official()
        self.gs = wm.Gaussian_Shading(self.ch, self.hw, float(m.get("fpr", 1e-2)), int(m.get("user_number", 1)))
        key_str = str(m.get("key", cfg["stylering"]["key"])).encode()
        seed = int.from_bytes(hashlib.sha256(b"gs-official-key" + key_str).digest()[:8], "little") % (2 ** 63 - 1)
        g = torch.Generator().manual_seed(seed)
        self.gs.key = torch.randint(0, 2, [1, 4, 64, 64], generator=g).cuda()     # fixed key (official: random per call)
        # our re-implementation with the SAME key, for the bit-level cross-check
        from models.baseline.gaussian_shading import GaussianShadingWM
        c2 = dict(cfg)
        c2["methods"] = {"gaussian_shading": {"n_bits": self._n_bits, "ch": self.ch, "hw": self.hw}}
        self.ours = GaussianShadingWM(c2, backend)
        self.ours.keystream = self.gs.key.cpu().numpy().astype(int)

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def _set_watermark(self, payload: np.ndarray) -> None:
        w = np.asarray(payload).astype(int).reshape(1, 4 // self.ch, 64 // self.hw, 64 // self.hw)
        self.gs.watermark = torch.from_numpy(w).cuda()

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        # official create_watermark_and_return_w() with the watermark injected and scipy's RNG seeded
        self._set_watermark(payload)
        sd = self.gs.watermark.repeat(1, self.ch, self.hw, self.hw)
        msg = ((sd + self.gs.key) % 2).flatten().cpu().numpy()
        np.random.seed(int(seed) % (2 ** 32))
        w = self.gs.truncSampling(msg)                                   # official half-normal sampling
        return w.float().cpu()

    def _votes(self, z: torch.Tensor) -> torch.Tensor:
        """Official split/sum vote counts without the threshold (for a payload-agnostic score)."""
        reversed_m = (z > 0).int()
        reversed_sd = (reversed_m + self.gs.key) % 2
        ch_stride, hw_stride = 4 // self.ch, 64 // self.hw
        s1 = torch.cat(torch.split(reversed_sd, tuple([ch_stride] * self.ch), dim=1), dim=0)
        s2 = torch.cat(torch.split(s1, tuple([hw_stride] * self.hw), dim=2), dim=0)
        s3 = torch.cat(torch.split(s2, tuple([hw_stride] * self.hw), dim=3), dim=0)
        return torch.sum(s3, dim=0)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().float().cuda()
        reversed_m = (z > 0).int()
        reversed_sd = (reversed_m + self.gs.key) % 2
        bits = self.gs.diffusion_inverse(reversed_sd).flatten().cpu().numpy().astype(int)   # official decode
        votes = self._votes(z).flatten().float().cpu().numpy()
        n_copy = self.ch * self.hw * self.hw
        conf = np.abs(2.0 * votes / n_copy - 1.0)
        ours = self.ours.extract(zT_hat)                                   # our re-implementation, same key
        agree = float((ours.bits == bits).mean())
        return ExtractResult(bits=bits, llr=2 * votes / n_copy - 1.0, accepted=None, score=float(conf.mean()),
                             meta={"ours_agree": agree, "ours_score": float(ours.score)})
