"""RingID baseline [3] in multi-bit form.

# PAPER [3]: discretised rings (each ring value = +alpha or -alpha, alpha=64 in the Fourier
#   domain, capacity 2^n for n rings), rings on channel 3 with radii from 3 upward, lossless
#   imprinting (real part only, aligned at the Fourier centre), spatial shift = multiply the
#   pattern by the chessboard (-1)^(u+v) damped by eta in [0.8,0.9]; heterogeneous Gaussian-noise
#   watermark on channel 0 (fixed key noise inside a radius-14 disk); matching by L1 distance.
# OURS (multi-bit protocol, documented in Part 3): bit i <-> sign of ring i; decoding by
#   comparing the L1 distance to +alpha vs -alpha per ring (== argmin over the 2^n codebook,
#   since L1 decomposes over disjoint rings); n_bits rings = radii r_start .. r_start+n_bits-1;
#   no ECC (RingID has none). Score for TPR@FPR = mean per-ring sign-decision margin.
"""
from __future__ import annotations

import numpy as np
import torch

from core.ecc import decrypt, encrypt
from models.base import ExtractResult, WatermarkMethod
from models.baseline._treering import circle_mask, fft2c, ifft2c, key_noise, ring_band


class RingIDWM(WatermarkMethod):
    name = "ringid"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("ringid", {})
        self._n_bits = int(m.get("n_bits", cfg["stylering"]["n_bits"]))
        self.ch = int(m.get("channel", 3))
        self.r0 = int(m.get("r_start", 3))
        self.alpha = float(m.get("alpha", 64.0))
        self.eta = float(m.get("eta", 0.85))
        self.shift = bool(m.get("spatial_shift", True))
        self.key = str(m.get("key", cfg["stylering"]["key"])).encode()
        self.size = backend.latent
        self.masks = [torch.from_numpy(ring_band(self.size, self.r0 + i, self.r0 + i - 1)) for i in range(self._n_bits)]
        uu, vv = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing="ij")
        self.chess = ((-1.0) ** (uu + vv)).float() if self.shift else torch.ones(self.size, self.size)
        # heterogeneous channel-0 noise watermark (verification only): fixed key noise inside r=14 disk
        self.het_mask = torch.from_numpy(circle_mask(self.size, 14))
        self.het_F = fft2c(key_noise(self.key + b"-het", self.size))[0, 0]

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        bits = encrypt(payload, self.key)      # OURS: keep payload uniformly random (key-stream)
        z = self.backend.sample_noise(seed, latent=self.size)
        F = fft2c(z[0, self.ch].to(torch.float32))
        re, im = F.real.clone(), F.imag.clone()
        for mask, b in zip(self.masks, bits):
            val = (2 * int(b) - 1) * self.alpha * self.eta * self.chess[mask]
            re[mask] = val
            im[mask] = 0.0                       # PAPER: lossless imprinting -> real part only
        z[0, self.ch] = ifft2c(torch.complex(re, im)).real.to(z.dtype)
        # heterogeneous Gaussian-noise watermark on channel 0
        F0 = fft2c(z[0, 0].to(torch.float32))
        F0[self.het_mask] = self.het_F[self.het_mask].to(F0.dtype)
        z[0, 0] = ifft2c(F0).real.to(z.dtype)
        return z

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        zT_hat = zT_hat.detach().cpu().float()
        F = fft2c(zT_hat[0, self.ch])
        bits, margins, y = [], [], []
        for mask in self.masks:
            re = F.real[mask] * self.chess[mask]        # undo spatial shift
            a = self.alpha * self.eta
            d_pos = float((re - a).abs().sum() + F.imag[mask].abs().sum())
            d_neg = float((re + a).abs().sum() + F.imag[mask].abs().sum())
            bits.append(1 if d_pos < d_neg else 0)
            margins.append(abs(d_neg - d_pos) / (2 * a * int(mask.sum())))
            y.append(float(re.mean() / a))
        bits = decrypt(np.array(bits), self.key)
        return ExtractResult(bits=bits, llr=np.array(y), accepted=None, score=float(np.mean(margins)))
