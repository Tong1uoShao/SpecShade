"""Shared Tree-Ring-family helpers for the RingID / GaussMarker baselines.

# PAPER [1,3,4]: circle/ring masks on the fftshift'ed spectrum; 'ring' pattern = key noise FFT
#   whose value is made constant along each ring; complex injection F[mask] = pattern[mask];
#   detection = L1 distance to the pattern inside the mask.
# OURS: masks centred at the true Fourier centre (size//2, size//2) -- RingID's corrected
#   alignment (§5.2 'lossless imprinting') -- instead of Tree-Ring's off-by-one anchor.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def circle_mask(size: int, r: float) -> np.ndarray:
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    return ((xx - c) ** 2 + (yy - c) ** 2) <= r ** 2


def ring_band(size: int, r_out: float, r_in: float) -> np.ndarray:
    return circle_mask(size, r_out) & ~circle_mask(size, r_in)


def key_noise(key: bytes, size: int, n_ch: int = 4) -> torch.Tensor:
    """Deterministic N(0,1) key noise [1,n_ch,size,size] from a byte key."""
    seed = int.from_bytes(hashlib.sha256(b"treering-key" + key).digest()[:8], "little") % (2 ** 31 - 1)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_ch, size, size, generator=g)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


def ifft2c(F: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1)))


def ring_pattern(key: bytes, size: int, channel: int, radius: int) -> torch.Tensor:
    """Tree-Ring 'ring' pattern [size,size] complex: key-noise FFT made constant per ring r<=radius."""
    G = fft2c(key_noise(key, size))[0, channel]
    pat = torch.zeros_like(G)
    c = size // 2
    for r in range(1, radius + 1):
        band = torch.from_numpy(ring_band(size, r, r - 1))
        pat[band] = G[c, c + r]                        # PAPER: value sampled from one ring location
    return pat


def inject_complex(zT: torch.Tensor, channel: int, mask: np.ndarray, pattern: torch.Tensor) -> torch.Tensor:
    """F[channel][mask] := pattern[mask] (complex injection), IFFT, keep the real part."""
    z = zT.clone()
    F = fft2c(z[0, channel].to(torch.float32))
    m = torch.from_numpy(mask)
    F[m] = pattern[m].to(F.dtype)
    z[0, channel] = ifft2c(F).real.to(z.dtype)
    return z


def l1_complex(zT_hat: torch.Tensor, channel: int, mask: np.ndarray, pattern: torch.Tensor) -> float:
    F = fft2c(zT_hat[0, channel].to(torch.float32))
    m = torch.from_numpy(mask)
    return float((F[m] - pattern[m].to(F.dtype)).abs().mean())
