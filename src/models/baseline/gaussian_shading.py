"""Gaussian Shading baseline [2], re-implemented faithfully from the official watermark.py
(vendor/Gaussian-Shading) so that it runs inside our offline pipeline (no .cuda() hard-coding).

# PAPER [2]: watermark bits (1, 4/ch, 64/hw, 64/hw) are tiled by (ch, hw, hw), XOR'ed with a
#   random key stream (ChaCha20 variant), and each latent element is sampled from the half of
#   N(0,1) selected by its bit (truncnorm with denominator 2 == |N(0,1)| * sign). Extraction:
#   sign(z_hat) XOR key -> majority vote over each tile -> bits.
# OURS: capacity set by (ch, hw) so that 4*64*64/(ch*hw*hw) == n_bits (16 -> ch=4, hw=16;
#   32 -> ch=2, hw=16); key stream derived deterministically from cfg key; score for TPR@FPR =
#   mean vote confidence |2*frac-1| (payload-agnostic).
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from core.ecc import encrypt, decrypt
from core.fourier_rings import rotate_latents, shift_latent, zoom_latents
from models.base import ExtractResult, WatermarkMethod

_FACTORS = {16: (4, 16), 32: (2, 16), 64: (1, 16), 256: (1, 8)}


class GaussianShadingWM(WatermarkMethod):
    name = "gaussian_shading"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("gaussian_shading", {})
        self._n_bits = int(m.get("n_bits", cfg["stylering"]["n_bits"]))
        self.ch, self.hw = _FACTORS[self._n_bits] if "ch" not in m else (int(m["ch"]), int(m["hw"]))
        self.latent = backend.latent
        assert 4 * self.latent ** 2 // (self.ch * self.hw * self.hw) == self._n_bits
        self.key = str(m.get("key", cfg["stylering"]["key"])).encode()
        h = hashlib.sha256(b"gs-keystream" + self.key).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        self.keystream = rng.integers(0, 2, (1, 4, self.latent, self.latent))   # PAPER: random key bits
        self.threshold = 1 if (self.hw == 1 and self.ch == 1) else self.ch * self.hw * self.hw // 2
        # OURS (control): the same detection-side synchronisation searches as StyleRing, applied to GS's own
        #   hard-vote decoder: rotate / zoom+shift the inverted latent, keep the candidate whose bipolar
        #   vote pattern agrees best with the key (mean |vote| over bits), peak-gated. Off by default (= paper).
        self.rot_search = bool(int(m.get("rot_search", 0)))
        self.rot_step = float(m.get("rot_step", 1.0))
        self.crop_search = bool(int(m.get("crop_search", 0)))
        self.scales = np.round(np.arange(float(m.get("scale_min", 0.55)), float(m.get("scale_max", 1.0)) + 1e-9,
                                         float(m.get("scale_step", 0.01))), 4)
        self.max_shift = float(m.get("max_shift", 12.0))
        self.sync_z = float(m.get("sync_z", 6.0))
        self.crop_z = float(m.get("crop_z", 8.0))
        self.n_copy = self.ch * self.hw * self.hw
        bit_of = self._tile(np.arange(self._n_bits))[0]                    # bit index per latent element
        kb = (1 - 2 * self.keystream[0]).astype(np.float32)                 # bipolar key
        K = np.stack([np.where(bit_of == b, kb, 0.0) for b in range(self._n_bits)]).astype(np.float32)
        self.K = torch.from_numpy(K)                                        # (n_bits, 4, h, w)
        self._kb = torch.from_numpy(kb)                                     # (4, h, w) bipolar key
        self._FKc = torch.conj(torch.fft.fft2(self.K)) if self.crop_search else None

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def _tile(self, bits: np.ndarray) -> np.ndarray:
        w = np.asarray(bits).astype(int).reshape(1, 4 // self.ch, self.latent // self.hw, self.latent // self.hw)
        return np.tile(w, (1, self.ch, self.hw, self.hw))

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        sd = self._tile(payload)
        m = (sd + self.keystream) % 2                                    # PAPER: XOR with key stream
        z = self.backend.sample_noise(seed, latent=self.latent)
        sign = torch.from_numpy(2 * m - 1).float()
        return z.abs() * sign                                           # == truncnorm half sampling

    def _vote(self, sd_hat: np.ndarray) -> np.ndarray:
        s = sd_hat.reshape(1, self.ch, 4 // self.ch, self.hw, self.latent // self.hw, self.hw, self.latent // self.hw)
        votes = s.sum(axis=(1, 3, 5))                                   # sum over the tile copies
        return (votes > self.threshold).astype(int).ravel(), votes.ravel()

    # ---------- synchronisation searches (control, see __init__) ----------
    def _votes(self, Z: torch.Tensor) -> torch.Tensor:
        """(B, 4, h, w) -> (B, n_bits) bipolar vote means: sign(z) . key, averaged over each bit's copies."""
        S = torch.sign(Z.detach().float())
        S[S == 0] = 1.0
        A = S * self._kb[None]                                              # per-element agreement with the key
        B = A.shape[0]
        A = A.reshape(B, self.ch, 4 // self.ch, self.hw, self.latent // self.hw, self.hw, self.latent // self.hw)
        return A.sum(dim=(1, 3, 5)).reshape(B, -1) / self.n_copy              # same copy structure as _vote

    @staticmethod
    def _zscore(vals: torch.Tensor, peak: float) -> float:
        med = float(vals.median())
        mad = 1.4826 * float((vals - med).abs().median()) + 1e-9
        return (peak - med) / mad

    def _sync_rot(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        angles = np.arange(0.0, 360.0, self.rot_step)
        Z = rotate_latents(z, angles)
        stat = self._votes(Z).abs().mean(1)
        b = int(torch.argmax(stat))
        zs = self._zscore(stat, float(stat[b]))
        gated = self.sync_z > 0 and zs < self.sync_z
        if gated:
            b = 0
        return (z if b == 0 else Z[b:b + 1]), {"theta": float(angles[b]), "sync_stat": float(stat[b]),
                                                "sync_stat0": float(stat[0]), "sync_z": zs, "gated": bool(gated)}

    def _sync_crop(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = w = self.latent
        Zs = zoom_latents(z, self.scales)                                   # (S, 4, h, w)
        S = torch.sign(Zs)
        S[S == 0] = 1.0
        FS = torch.fft.fft2(S)                                              # (S, 4, h, w)
        acc = torch.zeros(len(self.scales), h, w)
        for b in range(self._n_bits):                                       # correlation with the key of bit b
            acc += torch.fft.ifft2((FS * self._FKc[b][None]).sum(1)).real.abs()   # over all integer shifts
        stat = acc / self._n_bits / self.n_copy
        idx = torch.arange(h)
        px = torch.where(idx > h // 2, idx - h, idx).float()
        ok = px.abs() <= self.max_shift
        win = (ok[:, None] & ok[None, :])[None].expand_as(stat)
        vals = stat[win]
        stat = stat.masked_fill(~win, -1.0)
        flat = int(torch.argmax(stat.reshape(-1)))
        si, rest = divmod(flat, h * w)
        yi, xi = divmod(rest, w)
        peak = float(stat[si, yi, xi])
        zs = self._zscore(vals, peak)
        s0, ty0, tx0 = float(self.scales[si]), float(px[yi]), float(px[xi])
        best = (s0, ty0, tx0, peak)
        if not (self.crop_z > 0 and zs < self.crop_z):                      # sub-pixel refinement around the peak
            for s in (s0 - 0.01, s0, s0 + 0.01):
                zz = zoom_latents(z, [s])
                for dy in np.arange(-0.75, 0.76, 0.25):
                    for dx in np.arange(-0.75, 0.76, 0.25):
                        v = float(self._votes(shift_latent(zz, ty0 + dy, tx0 + dx)).abs().mean())
                        if v > best[3]:
                            best = (s, ty0 + dy, tx0 + dx, v)
        s, ty, tx, st = best
        gated = self.crop_z > 0 and zs < self.crop_z
        if gated or (s, ty, tx) == (1.0, 0.0, 0.0):
            s, ty, tx, out = 1.0, 0.0, 0.0, z
        else:
            out = shift_latent(zoom_latents(z, [s]), ty, tx)
        return out, {"scale": s, "ty": ty, "tx": tx, "crop_stat": st, "crop_z": zs, "crop_gated": bool(gated)}

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().cpu().float()
        meta = None
        if self.rot_search:
            z, meta = self._sync_rot(z)
        if self.crop_search:
            z, m2 = self._sync_crop(z)
            meta = {**(meta or {}), **m2}
        m_hat = (z.numpy() > 0).astype(int)
        sd_hat = (m_hat + self.keystream) % 2
        bits, votes = self._vote(sd_hat)
        n_copy = self.ch * self.hw * self.hw
        conf = np.abs(2.0 * votes / n_copy - 1.0)                       # per-bit vote confidence
        return ExtractResult(bits=bits, llr=2 * votes / n_copy - 1.0, accepted=None,
                             score=float(conf.mean()), meta=meta)
