"""StyleRing: survival-map-driven bit allocation + distribution-preserving ring modulation
+ soft-decision BCH decoding (idea_report Part 2, Method 3.3-3.6).

# OURS throughout. Decode variants (ablation 4 of Part 3):
#   soft   : LLR from probe calibration -> Chase-II BCH            (the method)
#   hard   : sign-only LLR (kappa=1, |llr| uniform) -> hard BCH   (w/o soft decision)
#   argmin : per-cell hard sign on the first k cells, no ECC       (RingID-style codebook)
#   no_ecc : payload embedded raw on the k best cells, hard sign   (w/o ECC, embed differs!)
# Allocation strategies (ablations 2-3): measured | uniform | lowcut | highring | spatial_sign
#   spatial_sign = Gaussian-Shading/GaussMarker-style per-element sign carrier [2,4].
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch

from core.allocation import SurvivalMap, allocate, allocate_repeat
from core.ecc import BCHCode, chase2_decode, decrypt, encrypt, soft_distance
from core.fourier_rings import (CellReader, SectorChips, ShiftScaleSync, enumerate_cells, ring_mask,
                                rotate_latents, zoom_latents)
from core.modulation import (CellCalib, combine_positions, count_calib, embed_bits, hard_bits,
                             read_cells, to_llr)
from models.base import ExtractResult, WatermarkMethod


def _flat_map(cells) -> SurvivalMap:
    n = len(cells)
    return SurvivalMap(cells, np.ones(n), np.ones(n), np.ones(n), {}, {"note": "flat (no probe)"})


class StyleRing(WatermarkMethod):
    name = "stylering"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        sr = cfg["stylering"]
        self._n_bits = int(sr["n_bits"])
        self.key = str(sr["key"]).encode()
        self.strategy = str(sr.get("alloc_strategy", "measured"))
        self.variant = str(sr.get("decode_variant", "soft"))
        if self.variant == "no_ecc":                  # repetition only: no code, any capacity (e.g. 256 bit)
            self.bch = types.SimpleNamespace(n=self._n_bits, k=self._n_bits, t=0)
        else:
            self.bch = BCHCode(int(sr["bch"]["n"]), int(sr["bch"]["k"]))
            if self.bch.k != self._n_bits:
                raise ValueError(f"BCH k={self.bch.k} must equal n_bits={self._n_bits}")
        self.n_flip = int(sr.get("chase_flip", 4))
        # OURS (artifact fix): keyed sector chips inside each ring; 1 = legacy full-ring
        #   coherence (bit-exact with earlier runs), K>=2 = K angular sectors, 0 = per-pair chips.
        # ASSUMPTION: the probe calibration (mu/sigma per cell) was measured with full-ring
        #   coherence and is reused unchanged for sector variants (only relative LLR weights).
        self.n_sectors = int(sr.get("sectors", 1))
        self.sector_rotate = bool(int(sr.get("sector_rotate", 0)))   # keyed per-ring wedge rotation
        self.chips = (SectorChips(self.key, self.n_sectors, self.sector_rotate)
                      if self.n_sectors != 1 else None)
        # OURS full-budget: carrier = real (legacy) | both (adds the imaginary part of every ring);
        #   r_max may extend to the partial corner rings (44 for a 64 latent); alloc_strategy
        #   'repeat' spreads the codeword over ALL cells (balanced repetition, LLR soft-combining);
        #   calib 'count' weights cells by their pixel count instead of the probe statistics.
        parts = (0, 1) if str(sr.get("carrier", "real")) == "both" else (0,)
        self.cells = enumerate_cells(r_min=int(sr.get("r_min", 1)), r_max=int(sr.get("r_max", 31)),
                                     latent=backend.latent, parts=parts)
        smap_path = Path(cfg["paths"].get("survival_map", ""))
        if not smap_path.is_absolute():                       # resolve against code/ not CWD
            smap_path = Path(__file__).resolve().parents[2] / smap_path
        self.smap = SurvivalMap.load(str(smap_path)) if smap_path.is_file() else None
        if self.strategy == "measured" and self.smap is None:
            raise FileNotFoundError(f"alloc_strategy=measured needs a survival map at {smap_path}; "
                                    "run probe.py first or use lowcut/uniform")
        calib_mode = str(sr.get("calib", "probe"))
        self.calib: CellCalib | None = (count_calib(self.cells, backend.latent) if calib_mode == "count"
                                        else self.smap.to_calib() if self.smap else None)
        self.spatial = self.strategy == "spatial_sign"
        self.spatial_read = str(sr.get("spatial_read", "hard"))    # hard (GS majority) | soft (magnitude-weighted)
        self.cell_read_hard = str(sr.get("cell_read", "soft")) == "hard"   # ring cells: hard sign votes instead of magnitude-weighted
        self.pos_of: np.ndarray | None = None
        self.sizes: np.ndarray | None = None
        n_code = self.bch.n if self.variant != "no_ecc" else self._n_bits
        if self.strategy == "repeat":
            sizes = [int(ring_mask(backend.latent, backend.latent, c.r).sum()) for c in self.cells]
            self.alloc, self.pos_of = allocate_repeat(self.cells, n_code, sizes,
                                                      pair_parts=bool(int(sr.get("pair_parts", 0))))
            self.sizes = np.asarray(sizes, dtype=np.float64)
        elif not self.spatial:
            sub = sr.get("attack_subset")                      # YAML list or "--override a@0.5,b@0.3"
            sub = [s for s in sub.split(",") if s] if isinstance(sub, str) else sub
            self.alloc = allocate(self.smap or _flat_map(self.cells), n_code, self.strategy,
                                  attack_subset=sub or None)
        else:
            self.alloc = None
            self._perm = np.random.default_rng(int.from_bytes(self.key[:8].ljust(8, b"\0"), "little")
                                               ).permutation(4 * backend.latent ** 2)
        # OURS rotation synchronisation: the keyed sign pattern is its own template. At detection the
        #   inverted latent is rotated over the circle (rot_step degrees), every angle is read with the
        #   vectorised reader, and the angle with the strongest mean |read-out| per code position wins;
        #   the null calibration runs the same search, so the FPR stays at its nominal level.
        self.rot_search = bool(int(sr.get("rot_search", 0)))
        self.rot_step = float(sr.get("rot_step", 1.0))
        self.sync_z = float(sr.get("sync_z", 6.0))       # peak gating for the rotation search (0 = always accept)
        rs = sr.get("rot_scales", [1.0])
        self.rot_scales = [float(x) for x in (rs.split(",") if isinstance(rs, str) else rs)]
        self.crop_z = float(sr.get("crop_z", 8.0))       # peak gating for the crop search
        self.reader = CellReader(self.alloc, self.chips, backend.latent) if self.alloc is not None else None
        self.n_code = n_code
        # OURS crop synchronisation (translation + scale search, see ShiftScaleSync); off by default.
        self.crop_search = bool(int(sr.get("crop_search", 0)))
        self.scales = np.round(np.arange(float(sr.get("scale_min", 0.55)), float(sr.get("scale_max", 1.0)) + 1e-9,
                                         float(sr.get("scale_step", 0.01))), 4)
        # OURS joint search (rotation x zoom x shift in one pass) for combined geometric attacks; when on it
        #   replaces the two sequential searches. geo_step = coarse angle step (degrees).
        self.flip_search = bool(int(sr.get("flip_search", 0)))
        self.geo_search = bool(int(sr.get("geo_search", 0)))
        self.geo_step = float(sr.get("geo_step", 3.0))
        self.geo_z = float(sr.get("geo_z", 8.0))
        # OURS cascade: run the cheap sequential searches first; only when BOTH are gated (no rotation and no
        #   crop explains the latent) fall back to the joint search (compound geometric edits, ~10x the cost).
        self.geo_fallback = bool(int(sr.get("geo_fallback", 0)))
        self.geo_scales = np.round(np.arange(float(sr.get("geo_scale_min", sr.get("scale_min", 0.55))),
                                             float(sr.get("geo_scale_max", sr.get("scale_max", 1.0))) + 1e-9,
                                             float(sr.get("scale_step", 0.01))), 4)
        self.sync = (ShiftScaleSync(self.alloc, self.chips, self.pos_of, n_code, backend.latent,
                                    up=int(sr.get("shift_up", 4)), max_shift=float(sr.get("max_shift", 12.0)))
                     if ((self.crop_search or self.geo_search or self.geo_fallback) and self.alloc is not None)
                     else None)

    @property
    def n_bits(self) -> int:
        return self._n_bits

    # ---------- code bits ----------
    def _code(self, payload: np.ndarray) -> np.ndarray:
        enc = encrypt(payload, self.key)
        return enc if self.variant == "no_ecc" else self.bch.encode(enc)

    # ---------- spatial-sign control (GS / GaussMarker style) ----------
    def _spatial_positions(self, n_code: int) -> list[np.ndarray]:
        total = self._perm.size
        per = total // n_code
        return [self._perm[i * per:(i + 1) * per] for i in range(n_code)]

    def _embed_spatial(self, zT: torch.Tensor, code: np.ndarray) -> torch.Tensor:
        flat = zT.flatten().clone()
        sign = torch.sign(flat)                 # unassigned leftovers keep their own sign
        sign[sign == 0] = 1.0
        for pos, b in zip(self._spatial_positions(len(code)), code):
            sign[torch.from_numpy(pos)] = float(2 * int(b) - 1)
        return (flat.abs() * sign).view_as(zT)      # |z| * (2s-1), GaussMarker Eq.(2) [4]

    def _read_spatial(self, zT_hat: torch.Tensor, n_code: int) -> np.ndarray:
        # OURS: spatial_read 'hard' = mean sign (Gaussian Shading's majority vote [2]);
        #   'soft' = magnitude-weighted vote sum(z)/sum(|z|), the spatial analogue of read_cell.
        flat = zT_hat.flatten().float()
        if self.spatial_read == "soft":
            return np.array([float(flat[torch.from_numpy(pos)].sum() / (flat[torch.from_numpy(pos)].abs().sum() + 1e-8))
                             for pos in self._spatial_positions(n_code)])
        flat = torch.sign(flat)
        return np.array([float(flat[torch.from_numpy(pos)].mean()) for pos in self._spatial_positions(n_code)])

    # ---------- rotation synchronisation (# OURS) ----------
    def sync_stat(self, Z: torch.Tensor) -> np.ndarray:
        """Blind synchronisation statistic per candidate latent: mean |read-out| over code positions
        (message-agnostic; ~0 when the chips are misaligned, ~1 on a freshly modulated latent)."""
        Y = self.reader.read(Z, hard=self.cell_read_hard)                       # (B, n_cells)
        if self.pos_of is not None:
            W = np.zeros((len(self.alloc), self.n_code))
            W[np.arange(len(self.alloc)), self.pos_of] = self.sizes
            Y = (Y @ W) / W.sum(0)[None, :]
        return np.abs(Y).mean(1)

    def _sync_rotation(self, zT_hat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        angles = np.arange(0.0, 360.0, self.rot_step)
        # OURS: optional joint zoom candidates (rot_scales) for rotations that expanded the canvas
        #   (content shrinks by 1/(|cos|+|sin|)); default [1.0] = pure rotation search.
        cand = [(s, a) for s in self.rot_scales for a in angles]
        Z = torch.cat([rotate_latents(zoom_latents(zT_hat, [s]) if s != 1.0 else zT_hat, angles) for s in self.rot_scales])
        stat = self.sync_stat(Z)
        b = int(np.argmax(stat))
        # OURS peak gating: accept the searched angle only if its peak stands out of the angular noise
        #   floor (robust z-score over all candidates); otherwise keep the direct read. This keeps the
        #   always-on search from picking a random angle on heavily attacked (weak-signal) images,
        #   which would both raise the null threshold and destroy the message.
        med = float(np.median(stat))
        mad = 1.4826 * float(np.median(np.abs(stat - med))) + 1e-9
        z = (float(stat[b]) - med) / mad
        gated = self.sync_z > 0 and z < self.sync_z
        if gated:
            b = 0
        s_b, a_b = cand[b]
        return (zT_hat if b == 0 else Z[b:b + 1]), {"theta": float(a_b), "rot_scale": float(s_b), "sync_stat": float(stat[b]),
                                                     "sync_stat0": float(stat[0]), "sync_z": float(z), "gated": bool(gated)}

    # ---------- interface ----------
    def _sync_joint(self, zT_hat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        s, th, ty, tx, st, st0, zs = self.sync.search_joint(zT_hat, self.geo_scales, self.geo_step)
        gated = self.geo_z > 0 and zs < self.geo_z
        if not gated and (s, th, ty, tx) != (1.0, 0.0, 0.0, 0.0):
            zT_hat = ShiftScaleSync.apply_joint(zT_hat, s, th, ty, tx)
        else:
            s, th, ty, tx = 1.0, 0.0, 0.0, 0.0
        return zT_hat, {"geo_scale": s, "geo_theta": th, "geo_ty": ty, "geo_tx": tx, "geo_stat": st,
                        "geo_stat0": st0, "geo_z": float(zs), "geo_gated": bool(gated)}

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        code = self._code(np.asarray(payload).astype(int))
        zT = self.backend.sample_noise(seed, latent=self.backend.latent)
        if self.spatial:
            return self._embed_spatial(zT, code)
        return embed_bits(zT, self.alloc, code, self.chips, self.pos_of)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        # OURS mirror handling: a reflection is not in the rotation group, so the detector also tries the
        #   horizontally flipped latent (vertical flip = horizontal flip + 180 deg, covered by the rotation
        #   search) and keeps the candidate with the higher final score; the null runs the same procedure.
        if not self.flip_search:
            return self._extract_one(zT_hat)
        z = zT_hat.detach().cpu().float()
        r0, r1 = self._extract_one(z), self._extract_one(torch.flip(z, dims=[3]))
        best, flipped = (r1, True) if r1.score > r0.score else (r0, False)
        best.meta = {**(best.meta or {}), "flipped": flipped, "score_asis": float(r0.score), "score_flipped": float(r1.score)}
        return best

    def _extract_one(self, zT_hat: torch.Tensor) -> ExtractResult:
        zT_hat = zT_hat.detach().cpu().float()
        n_code = self.bch.n if self.variant != "no_ecc" else self._n_bits
        meta = None
        if self.geo_search and self.sync is not None:
            zT_hat, meta = self._sync_joint(zT_hat)
            meta.update(scale=meta["geo_scale"], theta=meta["geo_theta"], ty=meta["geo_ty"], tx=meta["geo_tx"])
        elif self.rot_search and not self.spatial:
            zT_hat, meta = self._sync_rotation(zT_hat)
        if self.sync is not None and self.crop_search and not self.geo_search:
            z_in = zT_hat
            s, ty, tx, st, st0, z = self.sync.search(zT_hat, self.scales)
            gated = self.crop_z > 0 and z < self.crop_z
            if not gated and (s, ty, tx) != (1.0, 0.0, 0.0):
                zT_hat = ShiftScaleSync.apply(zT_hat, s, ty, tx)
            else:
                s, ty, tx = 1.0, 0.0, 0.0
            meta = {**(meta or {}), "scale": s, "ty": ty, "tx": tx, "crop_stat": st, "crop_stat0": st0,
                    "crop_z": float(z), "crop_gated": bool(gated)}
            rot_gated = bool((meta or {}).get("gated", True))
            if self.geo_fallback and gated and rot_gated:
                z_geo, gmeta = self._sync_joint(z_in)
                meta = {**meta, **gmeta, "geo_fallback": True}
                if not gmeta["geo_gated"]:
                    zT_hat = z_geo
        y = (self._read_spatial(zT_hat, n_code) if self.spatial
             else read_cells(zT_hat, self.alloc, self.chips, hard=self.cell_read_hard))
        rep = self.pos_of is not None
        if self.variant in ("argmin", "no_ecc"):
            # per-position hard sign; message = first k positions (systematic) / raw payload
            if rep:                                # vote-weighted mean read-out per position
                y = combine_positions(y, self.pos_of, n_code, self.sizes)
            bits = decrypt(hard_bits(y)[:self._n_bits], self.key)
            return ExtractResult(bits=bits, llr=2 * y, accepted=None,
                                 score=float(np.abs(y[:self._n_bits]).mean()), meta=meta)
        if self.variant == "hard":
            llr = 4.0 * np.sign(y)                 # equal reliability everywhere
            if rep:
                llr = combine_positions(llr, self.pos_of, n_code)
            msg, acc, d, nerr = chase2_decode(llr, self.bch, n_flip=0)
        else:
            llr = to_llr(y, self.alloc, None if self.spatial else self.calib)
            if rep:                                # sum of per-cell LLRs = soft combining
                llr = combine_positions(llr, self.pos_of, n_code)
            msg, acc, d, nerr = chase2_decode(llr, self.bch, n_flip=self.n_flip)
        if not acc:
            return ExtractResult(bits=None, llr=llr, accepted=False, score=-float("inf"), n_corrected=-1, meta=meta)
        cw = self.bch.encode(msg)
        score = -soft_distance(llr, cw) / len(llr)     # mean LLR agreement with the decoded codeword
        return ExtractResult(bits=decrypt(msg, self.key), llr=llr, accepted=True, score=float(score),
                             n_corrected=int(nerr), meta=meta)
