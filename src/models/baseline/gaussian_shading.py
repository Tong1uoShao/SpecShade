"""Gaussian Shading baseline [2], re-implemented faithfully from the official watermark.py
(vendor/Gaussian-Shading) so that it runs inside our offline pipeline (no .cuda() hard-coding).

# PAPER [2]: watermark bits (1, 4/ch, 64/hw, 64/hw) are tiled by (ch, hw, hw), XOR'ed with a
#   random key stream (ChaCha20 variant), and each latent element is sampled from the half of
#   N(0,1) selected by its bit (truncnorm with denominator 2 == |N(0,1)| * sign). Extraction:
#   sign(z_hat) XOR key -> majority vote over each tile -> bits.
# OURS: capacity set by (ch, hw) so that 4*64*64/(ch*hw*hw) == n_bits (16 -> ch=4, hw=16;
#   32 -> ch=2, hw=16); key stream derived deterministically from cfg key; score for TPR@FPR =
#   mean vote confidence |2*frac-1| (payload-agnostic).
# OURS (control, off by default): the detection-side synchronisation of SpecShade applied to GS's own
#   spatial key pattern -- rotation search, zoom + shift search (all integer shifts by FFT correlation with
#   the key pattern of every bit, sub-pixel refinement), flip candidate and the gated joint fallback -- and
#   the magnitude-weighted (soft) vote. Used for the domain x vote-rule comparison under geometric attacks.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from core.ecc import encrypt, decrypt
from core.fourier_rings import _sync_device, rotate_latents, shift_latent, zoom_latents
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
        # OURS (control): the same detection-side synchronisation searches as SpecShade, applied to GS's own
        #   decoder: rotate / zoom+shift the inverted latent, keep the candidate whose vote pattern agrees best
        #   with the key (mean |vote| over bits), peak-gated. Off by default (= paper).
        self.rot_search = bool(int(m.get("rot_search", 0)))
        self.rot_step = float(m.get("rot_step", 1.0))
        self.crop_search = bool(int(m.get("crop_search", 0)))
        self.scales = np.round(np.arange(float(m.get("scale_min", 0.55)), float(m.get("scale_max", 1.0)) + 1e-9,
                                         float(m.get("scale_step", 0.01))), 4)
        self.max_shift = float(m.get("max_shift", 12.0))
        self.sync_z = float(m.get("sync_z", 6.0))
        self.crop_z = float(m.get("crop_z", 8.0))
        self.flip_search = bool(int(m.get("flip_search", 0)))          # also try the mirrored latent
        self.geo_fallback = bool(int(m.get("geo_fallback", 0)))        # joint angle x zoom x shift search when both gated
        self.geo_step = float(m.get("geo_step", 5.0))
        self.geo_z = float(m.get("geo_z", 8.0))
        self.geo_scales = np.round(np.arange(float(m.get("geo_scale_min", 0.55)), float(m.get("geo_scale_max", 1.0)) + 1e-9,
                                             float(m.get("geo_scale_step", 0.02))), 4)
        self.n_copy = self.ch * self.hw * self.hw
        # OURS (ablation): magnitude-weighted vote over the copies of a bit (Gaussian Shading++ style soft decision)
        #   instead of the hard majority of the paper; off by default (= paper). With the searches on, the rotation
        #   statistic and the refinement follow the same vote rule (as the cell reads of SpecShade do).
        self.soft_vote = bool(int(m.get("soft_vote", 0)))
        self.dev = _sync_device() if (self.rot_search or self.crop_search or self.geo_fallback) else torch.device("cpu")
        bit_of = self._tile(np.arange(self._n_bits))[0]                    # bit index per latent element
        kb = (1 - 2 * self.keystream[0]).astype(np.float32)                 # bipolar key
        K = np.stack([np.where(bit_of == b, kb, 0.0) for b in range(self._n_bits)]).astype(np.float32)
        self.K = torch.from_numpy(K).to(self.dev)                           # (n_bits, 4, h, w)
        self._kb = torch.from_numpy(kb).to(self.dev)                        # (4, h, w) bipolar key
        self._FKc = torch.conj(torch.fft.fft2(self.K)) if (self.crop_search or self.geo_fallback) else None
        self._shape = (self.ch, 4 // self.ch, self.hw, self.latent // self.hw, self.hw, self.latent // self.hw)

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
        """(B, 4, h, w) -> (B, n_bits) vote means in [-1, 1]: hard = mean of sign(z) . key over the copies of a bit;
        soft = sum(z . key) / sum(|z|) over the copies (magnitude-weighted, the same rule as the soft read)."""
        Zf = Z.detach().float().to(self.dev)
        B = Zf.shape[0]
        if self.soft_vote:
            num = (Zf * self._kb[None]).reshape(B, *self._shape).sum(dim=(1, 3, 5)).reshape(B, -1)
            den = Zf.abs().reshape(B, *self._shape).sum(dim=(1, 3, 5)).reshape(B, -1) + 1e-8
            return num / den
        S = torch.sign(Zf)
        S[S == 0] = 1.0
        A = (S * self._kb[None]).reshape(B, *self._shape)                   # per-element agreement with the key
        return A.sum(dim=(1, 3, 5)).reshape(B, -1) / self.n_copy              # same copy structure as _vote

    def _vstat(self, z: torch.Tensor) -> float:
        return float(self._votes(z).abs().mean())

    @staticmethod
    def _zscore(vals: torch.Tensor, peak: float) -> float:
        med = float(vals.median())
        mad = 1.4826 * float((vals - med).abs().median()) + 1e-9
        return (peak - med) / mad

    def _corr_maps(self, Zs: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        """(S, 4, h, w) -> (S, h, w): payload-agnostic alignment statistic of every candidate for every integer shift:
        the sum over bits of |correlation of the latent with the key pattern of the bit| (magnitude-weighted, like the
        shift-scale statistic of SpecShade), normalised by the total magnitude of the candidate (~1 when aligned)."""
        Zs = Zs.detach().float().to(self.dev)
        FS = torch.fft.fft2(Zs)                                             # (S, 4, h, w)
        acc = torch.zeros(Zs.shape[0], Zs.shape[-2], Zs.shape[-1], device=self.dev)
        for b0 in range(0, self._n_bits, chunk):                            # correlation with the key of every bit
            G = (FS[:, None] * self._FKc[None, b0:b0 + chunk]).sum(2)       # (S, chunk, h, w), over all integer shifts
            acc += torch.fft.ifft2(G).real.abs().sum(1)
        return acc / (Zs.abs().sum(dim=(1, 2, 3)) + 1e-8)[:, None, None]

    def _standardize(self, stat: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
        """(N, h, w) -> robust z-maps: every candidate map is standardised over its own shift window
        ((M - median) / 1.4826 MAD). # OURS: the spatial statistic has a zoom-dependent bias (few copies of a bit
        remain in a strongly zoomed-out latent), so the gate compares a peak with the candidates of its own zoom;
        outside the window the value is -inf."""
        W = win[None].expand_as(stat)
        flat = stat.masked_fill(~W, float("nan")).reshape(stat.shape[0], -1)
        med = flat.nanmedian(dim=1, keepdim=True).values
        mad = (flat - med).abs().nanmedian(dim=1, keepdim=True).values * 1.4826 + 1e-9
        return ((flat - med) / mad).reshape(stat.shape).masked_fill(~W, float("-inf"))

    def _window(self) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent
        idx = torch.arange(h, device=self.dev)
        px = torch.where(idx > h // 2, idx - h, idx).float()
        ok = px.abs() <= self.max_shift
        return px, ok[:, None] & ok[None, :]

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

    def _refine(self, z: torch.Tensor, s0: float, ty0: float, tx0: float, th0: float | None = None):
        """Local refinement of a coarse alignment with the vote statistic: angle +-geo_step/2 in 0.5 deg steps
        (joint search only), zoom +-0.02 in 0.01 steps, shift +-0.75 px in quarter-pixel steps."""
        def warp(s, th, ty, tx):
            zz = z if th is None else rotate_latents(z, [th % 360.0])
            return shift_latent(zoom_latents(zz, [s]), ty, tx)
        best = (self._vstat(warp(s0, th0, ty0, tx0)), s0, ty0, tx0, th0)
        if th0 is not None:
            for th in th0 + np.arange(-self.geo_step / 2, self.geo_step / 2 + 1e-9, 0.5):
                v = self._vstat(warp(s0, th, ty0, tx0))
                if v > best[0]:
                    best = (v, s0, ty0, tx0, float(th % 360.0))
            th0 = best[4]
        for s in (s0 - 0.02, s0 - 0.01, s0 + 0.01, s0 + 0.02):
            if 0.1 <= s <= 1.6:
                v = self._vstat(warp(s, th0, ty0, tx0))
                if v > best[0]:
                    best = (v, float(round(s, 4)), ty0, tx0, th0)
        s0 = best[1]
        for dy in np.arange(-0.75, 0.76, 0.25):
            for dx in np.arange(-0.75, 0.76, 0.25):
                if dy == 0 and dx == 0:
                    continue
                v = self._vstat(warp(s0, th0, ty0 + dy, tx0 + dx))
                if v > best[0]:
                    best = (v, s0, float(ty0 + dy), float(tx0 + dx), th0)
        return best[1], best[2], best[3], best[4], best[0]

    def _sync_crop(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = w = self.latent
        Zs = zoom_latents(z, self.scales)                                   # (S, 4, h, w)
        stat = self._corr_maps(Zs)                                          # (S, h, w)
        px, win = self._window()
        Zm = self._standardize(stat, win)
        flat = int(torch.argmax(Zm.reshape(-1)))
        zs, peak = float(Zm.reshape(-1)[flat]), float(stat.reshape(-1)[flat])
        si, rest = divmod(flat, h * w)
        yi, xi = divmod(rest, w)
        s0, ty0, tx0 = float(self.scales[si]), float(px[yi]), float(px[xi])
        gated = self.crop_z > 0 and zs < self.crop_z
        if gated or (s0, ty0, tx0) == (1.0, 0.0, 0.0):
            return z, {"scale": 1.0, "ty": 0.0, "tx": 0.0, "crop_stat": peak, "crop_z": zs, "crop_gated": bool(gated)}
        s, ty, tx, _, st = self._refine(z, s0, ty0, tx0)
        return shift_latent(zoom_latents(z, [s]), ty, tx), {"scale": s, "ty": ty, "tx": tx, "crop_stat": st,
                                                             "crop_z": zs, "crop_gated": False}

    def _sync_joint(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Joint angle x zoom x shift search (coarse grid geo_step / geo_scales, all integer shifts per candidate,
        every (zoom, angle) map standardised over its shift window), gated at geo_z, then the local refinement."""
        h = w = self.latent
        px, win = self._window()
        angles = np.arange(0.0, 360.0, self.geo_step)
        best = None
        for s in self.geo_scales:                                           # zoom once, rotate all angles at once
            Zr = rotate_latents(zoom_latents(z, [float(s)]), angles)        # (A, 4, h, w)
            stat = self._corr_maps(Zr)                                      # (A, h, w)
            Zm = self._standardize(stat, win)
            flat = int(torch.argmax(Zm.reshape(-1)))
            zs = float(Zm.reshape(-1)[flat])
            if best is None or zs > best[0]:
                ai, rest = divmod(flat, h * w)
                yi, xi = divmod(rest, w)
                best = (zs, float(stat.reshape(-1)[flat]), float(angles[ai]), float(s), float(px[yi]), float(px[xi]))
        zs, peak, th0, s0, ty0, tx0 = best
        gated = self.geo_z > 0 and zs < self.geo_z
        meta = {"geo_theta": th0, "geo_scale": s0, "geo_ty": ty0, "geo_tx": tx0, "geo_stat": peak, "geo_z": zs,
                "geo_gated": bool(gated)}
        if gated:
            return z, meta
        s, ty, tx, th, st = self._refine(z, s0, ty0, tx0, th0)
        meta.update(geo_theta=th, geo_scale=s, geo_ty=ty, geo_tx=tx, geo_stat=st)
        return shift_latent(zoom_latents(rotate_latents(z, [th]), [s]), ty, tx), meta

    # ---------- read-out ----------
    def _read(self, z: torch.Tensor, meta: dict | None) -> ExtractResult:
        zn = z.detach().cpu().float().numpy()
        m_hat = (zn > 0).astype(int)
        sd_hat = (m_hat + self.keystream) % 2
        if self.soft_vote:
            v = (2 * sd_hat - 1).astype(np.float64)                     # +-1 copy value
            w = np.abs(zn).astype(np.float64)                            # magnitude of the latent entry
            shape = (1, *self._shape)
            y = (v * w).reshape(shape).sum(axis=(1, 3, 5)).ravel() / (w.reshape(shape).sum(axis=(1, 3, 5)).ravel() + 1e-8)
            return ExtractResult(bits=(y > 0).astype(int), llr=y, accepted=None, score=float(np.abs(y).mean()), meta=meta)
        bits, votes = self._vote(sd_hat)
        conf = np.abs(2.0 * votes / self.n_copy - 1.0)                  # per-bit vote confidence
        return ExtractResult(bits=bits, llr=2 * votes / self.n_copy - 1.0, accepted=None,
                             score=float(conf.mean()), meta=meta)

    def _extract_one(self, zT_hat: torch.Tensor) -> ExtractResult:
        z0 = zT_hat.detach().cpu().float()
        z, meta = z0, {}
        if self.rot_search:
            z, meta = self._sync_rot(z)
        if self.crop_search:
            z, m2 = self._sync_crop(z)
            meta = {**meta, **m2}
        if self.geo_fallback and meta.get("gated", True) and meta.get("crop_gated", True):
            z, m3 = self._sync_joint(z0)                                # both sequential searches gated -> joint
            meta = {**meta, **m3, "geo_fallback": True}
        return self._read(z, meta or None)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        if not self.flip_search:
            return self._extract_one(zT_hat)
        r0, r1 = self._extract_one(zT_hat), self._extract_one(torch.flip(zT_hat, dims=[3]))
        best, flipped = (r1, True) if r1.score > r0.score else (r0, False)
        best.meta = {**(best.meta or {}), "flipped": flipped, "score_asis": float(r0.score), "score_flipped": float(r1.score)}
        return best
