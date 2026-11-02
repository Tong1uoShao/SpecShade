"""Bit -> noise modulation (embedding) and noise -> soft read-out (extraction).

# OURS: embed_bits writes one code bit per allocated cell as the common sign of the
#   ring's real Fourier coefficients (magnitudes kept, imaginary part kept).
#   to_llr turns normalised read-outs y_g into log-likelihood ratios using the
#   per-cell statistics measured by the offline probe (Method 3.6, kappa_g).
# Pure functions, CPU-testable, no model access.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .fourier_rings import (Cell, SectorChips, apply_to_cells, read_cells as _read_cells,
                            ring_mask, write_cell_sign)


def embed_bits(zT_raw: torch.Tensor, alloc: list[Cell], code_bits,
               chips: SectorChips | None = None, pos_of: np.ndarray | None = None) -> torch.Tensor:
    """Return a watermarked latent: cell alloc[i] carries code_bits[i] in {0,1}.

    chips: keyed sector chips (# OURS artifact fix); None == legacy full-ring coherence.
    pos_of: # OURS full-budget repetition -- cell alloc[i] carries code_bits[pos_of[i]], so
            several cells repeat one codeword position (soft-combined at read-out).
    """
    bits = np.asarray(code_bits).astype(int).ravel()
    if pos_of is not None:
        bits = bits[np.asarray(pos_of, dtype=int)]
    if len(alloc) != len(bits):
        raise ValueError(f"alloc ({len(alloc)}) and code_bits ({len(bits)}) length mismatch")
    sign_of = {cell: int(2 * b - 1) for cell, b in zip(alloc, bits)}
    return apply_to_cells(zT_raw, alloc, lambda F, cell: write_cell_sign(
        F, cell, sign_of[cell], None if chips is None else chips(F, cell)))


def read_cells(zT_hat: torch.Tensor, alloc: list[Cell], chips: SectorChips | None = None,
               hard: bool = False) -> np.ndarray:
    """Normalised read-outs y in [-1,1], one per allocated cell (chips must match embedding).
    hard=True: mean of coefficient signs (majority vote) instead of the magnitude-weighted read."""
    return _read_cells(zT_hat, alloc, chips, hard)


def hard_bits(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y) > 0).astype(int)


@dataclass
class CellCalib:
    """Per-cell attack-channel statistics from the probe: y_g | bit=1 ~ N(mu_g, sigma_g^2)."""
    mu: dict[tuple[int, int, int], float] = field(default_factory=dict)
    sigma: dict[tuple[int, int, int], float] = field(default_factory=dict)
    attack_set: str = ""

    def get(self, cell: Cell) -> tuple[float, float]:
        k = cell.key()
        return self.mu.get(k, 1.0), self.sigma.get(k, 1.0)

    def save(self, path: str) -> None:
        keys = np.array(list(self.mu.keys()), dtype=np.int64).reshape(-1, 3)
        np.savez(path, keys=keys, mu=np.array([self.mu[tuple(k)] for k in keys]),
                 sigma=np.array([self.sigma[tuple(k)] for k in keys]),
                 attack_set=np.array(self.attack_set))

    @classmethod
    def load(cls, path: str) -> "CellCalib":
        d = np.load(path, allow_pickle=False)
        keys = [tuple(int(v) for v in k) for k in d["keys"]]
        return cls(mu=dict(zip(keys, d["mu"].tolist())),
                   sigma=dict(zip(keys, d["sigma"].tolist())),
                   attack_set=str(d["attack_set"]))


def to_llr(y: np.ndarray, alloc: list[Cell], calib: CellCalib | None = None,
           sigma_floor: float = 0.05) -> np.ndarray:
    """LLR_g = 2 * mu_g * y_g / sigma_g^2 (Gaussian channel, symmetric under bit flip).

    # OURS: with calib=None every cell gets kappa=1 (LLR = 2y), i.e. plain soft sign.
    """
    y = np.asarray(y, dtype=np.float64)
    if calib is None:
        return 2.0 * y
    llr = np.empty_like(y)
    for i, cell in enumerate(alloc):
        mu, sigma = calib.get(cell)
        sigma = max(float(sigma), sigma_floor)
        llr[i] = 2.0 * float(mu) * y[i] / (sigma ** 2)
    return llr


def count_calib(cells: list[Cell], latent: int) -> CellCalib:
    """# OURS full-budget: calibration from geometry alone -- mu = 1, sigma = 1/sqrt(n_pixels),
    i.e. LLR = 2 * n_pixels * y: the matched filter when every coefficient's sign survives
    with the same probability (the 'white' regime of per-pair chips), so a cell counts in
    proportion to the number of votes it holds. Needs no probe."""
    size = latent // 2 if any(c.q is not None for c in cells) else latent
    mu, sigma = {}, {}
    for c in cells:
        n = int(ring_mask(size, size, c.r).sum())
        mu[c.key()] = 1.0
        sigma[c.key()] = 1.0 / np.sqrt(max(n, 1))
    return CellCalib(mu=mu, sigma=sigma, attack_set="count")


def combine_positions(values: np.ndarray, pos_of: np.ndarray, n_code: int,
                      weights: np.ndarray | None = None) -> np.ndarray:
    """Fold per-cell values onto codeword positions: sum (LLRs) or weighted mean (read-outs)."""
    pos_of = np.asarray(pos_of, dtype=int)
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        return np.bincount(pos_of, weights=v, minlength=n_code)
    w = np.asarray(weights, dtype=np.float64)
    return np.bincount(pos_of, weights=v * w, minlength=n_code) / np.maximum(
        np.bincount(pos_of, weights=w, minlength=n_code), 1e-12)
