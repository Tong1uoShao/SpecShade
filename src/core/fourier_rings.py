"""Ring-cell geometry and Hermitian-safe frequency-domain read/write.

# OURS: A "cell" is the atomic placement unit of StyleRing: (channel c, radius r,
#   quadrant q | None). Full-image mode operates on the 2D FFT of one latent channel
#   (64x64, DC at (32,32) after fftshift); quadrant mode splits the latent into 2x2
#   spatial blocks (32x32, DC at (16,16)) and is used only by the probe as a spatial
#   control axis (idea_report Part 3, ablation 1).
# PAPER: ring masks follow Tree-Ring's "pixels at rounded distance r from centre"
#   [1]; reading only the real part and writing sign-coherently along a ring keeps
#   the spectrum conjugate-symmetric, i.e. the IFFT is exactly real (RingID's
#   "lossless imprinting" argument [3]).
# Pure functions, CPU-testable, no IO.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class Cell:
    c: int
    r: int
    q: int | None = None
    part: int = 0            # OURS full-budget: 0 = real part (default), 1 = imaginary part

    def key(self) -> tuple:
        k = (self.c, self.r, -1 if self.q is None else self.q)
        return k if self.part == 0 else k + (1,)   # 3-tuple keeps old calib/survival files valid

    def __str__(self) -> str:
        return (f"c{self.c}r{self.r}" + ("" if self.q is None else f"q{self.q}")
                + ("" if self.part == 0 else "i"))


@lru_cache(maxsize=None)
def _ring_mask_np(h: int, w: int, r: int) -> np.ndarray:
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    m = rad == r
    # OURS: the Nyquist line (index 0 of an even axis) pairs with itself under k -> -k and
    #   carries no imaginary part at (0,0); dropping it keeps every ring a clean set of
    #   antipodal pairs. No-op for r <= size//2 - 1 (bit-exact with earlier runs).
    m[0, :] = False
    m[:, 0] = False
    return m


def ring_mask(h: int, w: int, r: int) -> np.ndarray:
    """Boolean mask of pixels at rounded Euclidean distance r from the FFT centre."""
    return _ring_mask_np(h, w, r).copy()


def max_radius(size: int, full: bool = False) -> int:
    # OURS: default r_max keeps every ring inside the inscribed disc (complete rings);
    #   full=True extends to the corner pixel (1,1) off the Nyquist lines (partial rings,
    #   full-spectrum budget: every non-DC, non-Nyquist coefficient belongs to some ring).
    if full:
        return int(np.round(np.sqrt(2.0) * (size // 2 - 1)))
    return size // 2 - 1


def enumerate_cells(n_ch: int = 4, r_min: int = 1, r_max: int | None = None,
                    quadrants: bool = False, latent: int = 64,
                    parts: tuple[int, ...] = (0,)) -> list[Cell]:
    """All placement cells in a fixed, deterministic order (channel-major, radius-minor).

    parts: (0,) real-part carrier only (default, legacy); (0, 1) adds the imaginary-part
    carrier of every ring (# OURS full-budget)."""
    size = latent // 2 if quadrants else latent
    if r_max is None:
        r_max = max_radius(size)
    if r_max > max_radius(size, full=True):
        raise ValueError(f"r_max={r_max} exceeds {max_radius(size, full=True)} for size {size}")
    qs = [0, 1, 2, 3] if quadrants else [None]
    return [Cell(c, r, q, p) for c in range(n_ch) for q in qs for r in range(r_min, r_max + 1)
            for p in parts]


def quadrant_slice(q: int, latent: int) -> tuple[slice, slice]:
    half = latent // 2
    ys = slice(0, half) if q in (0, 1) else slice(half, latent)
    xs = slice(0, half) if q in (0, 2) else slice(half, latent)
    return ys, xs


def group_cells(cells: list[Cell]) -> dict[tuple[int, int | None], list[Cell]]:
    """Group cells by (channel, quadrant) so each spatial block is FFT'd once."""
    groups: dict[tuple[int, int | None], list[Cell]] = {}
    for cell in cells:
        groups.setdefault((cell.c, cell.q), []).append(cell)
    return groups


def block_fft(zT: torch.Tensor, c: int, q: int | None) -> torch.Tensor:
    """Centred 2D FFT (complex64) of latent channel c (optionally one quadrant)."""
    x = zT[0, c].to(torch.float32)
    if q is not None:
        ys, xs = quadrant_slice(q, zT.shape[-1])
        x = x[ys, xs]
    return torch.fft.fftshift(torch.fft.fft2(x))


def block_ifft_real(F: torch.Tensor, atol: float = 1e-3) -> torch.Tensor:
    """Inverse of block_fft; asserts the result is (numerically) real."""
    x = torch.fft.ifft2(torch.fft.ifftshift(F))
    resid = x.imag.abs().max().item()
    scale = x.real.abs().max().item() + 1e-8
    if resid > atol * scale:
        raise RuntimeError(f"non-real IFFT: imag residual {resid:.3e} vs scale {scale:.3e}")
    return x.real


def cell_mask_t(F: torch.Tensor, cell: Cell) -> torch.Tensor:
    h, w = F.shape[-2], F.shape[-1]
    return torch.from_numpy(ring_mask(h, w, cell.r)).to(F.device)


# ---------------------------------------------------------------------------
# Sector-coherent modulation (# OURS, artifact fix)
#
# Full-ring sign coherence (every coefficient of a ring shares the bit sign) makes the
# ring's real spectrum add up in phase, which is a radial ripple in the spatial latent
# (visible as concentric arcs at the four corners). Sector chips break that coherence:
# each ring is split into angular sectors, each sector gets a keyed +-1 "chip", and the
# bit is written as bit_sign * chip. The reader multiplies the chips back before summing,
# so the matched-filter gain against i.i.d. noise is unchanged, while the spatial pattern
# of the watermark loses its radial structure. Antipodal pixels (k, -k) always share a
# chip, so the spectrum stays conjugate-symmetric (IFFT exactly real).
#   n_sectors = 1  -> chips all +1 (legacy full-ring coherence; bit-exact with old runs)
#   n_sectors = K  -> K angular sectors of width pi/K over the folded angle [0, pi)
#   n_sectors = 0  -> one chip per antipodal pair (chip sequence; shortest coherence length)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _ring_pairs_np(h: int, w: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per ring pixel (mask order): canonical (dy, dx) of its antipodal pair and the parity
    (+1 for the representative k, -1 for its antipode -k)."""
    m = _ring_mask_np(h, w, r)
    ii, jj = np.nonzero(m)                                    # row-major == boolean-index order
    dy, dx = ii - h // 2, jj - w // 2
    flip = (dy < 0) | ((dy == 0) & (dx < 0))                  # canonical rep of the pair (k, -k)
    cdy, cdx = np.where(flip, -dy, dy), np.where(flip, -dx, dx)
    parity = np.where(flip, -1.0, 1.0).astype(np.float32)
    return cdy, cdx, parity


def ring_parity(h: int, w: int, r: int) -> np.ndarray:
    """+-1 per ring pixel (mask order): the sign that folds Im F(-k) = -Im F(k) onto Im F(k)."""
    return _ring_pairs_np(h, w, r)[2].copy()


@lru_cache(maxsize=None)
def _ring_groups_np(h: int, w: int, r: int, n_sectors: int, offset: float = 0.0) -> tuple[np.ndarray, int]:
    """Group id per ring pixel (mask order) and the number of groups.

    offset: angular rotation of the sector partition (radians, keyed per ring when
    `sector_rotate` is on) so that the wedges of different rings do not line up.
    """
    cdy, cdx, _ = _ring_pairs_np(h, w, r)
    if n_sectors <= 0:
        pairs: dict[tuple[int, int], int] = {}
        g = np.array([pairs.setdefault((int(a), int(b)), len(pairs)) for a, b in zip(cdy, cdx)])
        return g, len(pairs)
    theta = np.mod(np.arctan2(cdy, cdx) - offset, np.pi)      # folded, rotated angle in [0, pi)
    g = np.minimum((theta / (np.pi / n_sectors)).astype(int), n_sectors - 1)
    return g, n_sectors


def ring_groups(h: int, w: int, r: int, n_sectors: int, offset: float = 0.0) -> np.ndarray:
    return _ring_groups_np(h, w, r, n_sectors, offset)[0].copy()


def ring_chips(h: int, w: int, r: int, n_sectors: int, seed: int, rotate: bool = False) -> np.ndarray:
    """+-1 chip per ring pixel (mask order), constant within a group, keyed by `seed`.
    n_sectors == 1 returns all +1 (legacy full-ring coherence, bit-exact with chip-less code).
    rotate: keyed angular offset of the sector partition (# OURS: de-aligns the wedges of
    different rings so their spatial 'spokes' do not add up)."""
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(0.0, np.pi / n_sectors)) if (rotate and n_sectors >= 2) else 0.0
    g, n = _ring_groups_np(h, w, r, n_sectors, offset)
    if n_sectors == 1:
        return np.ones(len(g), dtype=np.float32)
    signs = rng.integers(0, 2, n) * 2 - 1
    return signs[g].astype(np.float32)


class SectorChips:
    """Keyed chip provider: (F, cell) -> +-1 tensor aligned with F.real[ring_mask]."""

    def __init__(self, key: bytes, n_sectors: int, rotate: bool = False) -> None:
        self.key = bytes(key)
        self.n_sectors = int(n_sectors)
        self.rotate = bool(rotate)
        self._cache: dict[tuple[int, int, tuple[int, int, int]], torch.Tensor] = {}

    def seed(self, cell: Cell) -> int:
        h = hashlib.sha256(self.key + b"|sector|" + repr(cell.key()).encode()).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 63 - 1)

    def __call__(self, F: torch.Tensor, cell: Cell) -> torch.Tensor:
        h, w = F.shape[-2], F.shape[-1]
        k = (h, w, cell.key())
        if k not in self._cache:
            self._cache[k] = torch.from_numpy(
                ring_chips(h, w, cell.r, self.n_sectors, self.seed(cell), self.rotate))
        return self._cache[k].to(F.device)


def _carrier(F: torch.Tensor, cell: Cell) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(values, mask, parity): the real-valued carrier of a cell in mask order.

    part 0: Re F[ring] (symmetric under k -> -k).
    part 1: parity * Im F[ring]  (# OURS full-budget: Im F(-k) = -Im F(k), so the parity-folded
            imaginary part is symmetric exactly like the real part and can carry a sign)."""
    m = cell_mask_t(F, cell)
    if cell.part == 0:
        return F.real[m], m, None
    par = torch.from_numpy(ring_parity(F.shape[-2], F.shape[-1], cell.r)).to(F.device)
    return F.imag[m] * par, m, par


def read_cell(F: torch.Tensor, cell: Cell, chip: torch.Tensor | None = None, hard: bool = False) -> float:
    """Normalised sign-coherence read-out y_g in [-1, 1].

    # OURS: y_g = mean(chip * Re F[ring]) / mean(|Re F[ring]|). A freshly modulated ring gives
    #   exactly +-1; unmodulated noise gives ~0 with std ~ 1/sqrt(K_ring). Normalising by
    #   the ring's own magnitude makes the statistic scale-free across images/attacks.
    #   chip=None is the legacy full-ring-coherent reader. Imaginary-part cells read the
    #   parity-folded imaginary part instead (see _carrier).
    #   hard=True: y_g = mean(chip * sign(v)) -- plain majority vote (control: is the
    #   magnitude weighting worth anything?).
    """
    v, _, _ = _carrier(F, cell)
    if hard:
        v = torch.sign(v)
        return float(v.mean() if chip is None else (v * chip).mean())
    num = v.mean() if chip is None else (v * chip).mean()
    return float(num / (v.abs().mean() + 1e-8))


def write_cell_sign(F: torch.Tensor, cell: Cell, sign: int, chip: torch.Tensor | None = None) -> torch.Tensor:
    """Set Re F[ring] := |Re F[ring]| * sign * chip (magnitudes untouched, Im untouched).

    # OURS: half-Gaussian magnitude x bit sign == Gaussian Shading's quantile-interval
    #   sampling with l=1 applied per ring [2]; ring symmetry keeps conjugate symmetry.
    #   With chips (SectorChips) the per-coefficient sign is keyed, see above.
    #   Imaginary-part cells set Im F[ring] := |Im F[ring]| * sign * chip * parity, which is
    #   antisymmetric under k -> -k (chip and |Im| are pair-symmetric), so the IFFT stays real.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +-1")
    v, m, par = _carrier(F, cell)
    s = v.abs() * sign if chip is None else v.abs() * (sign * chip)
    if cell.part == 0:
        re = F.real.clone()
        re[m] = s
        return torch.complex(re, F.imag.clone())
    im = F.imag.clone()
    im[m] = s * par
    return torch.complex(F.real.clone(), im)


def apply_to_cells(zT: torch.Tensor, cells: list[Cell],
                   fn) -> torch.Tensor:
    """Generic 'FFT once per block, apply fn(F, cell) sequentially, IFFT back' driver.

    fn: (F: complex tensor, cell) -> F' (must preserve conjugate symmetry).
    Returns a new latent of the same shape/dtype; untouched blocks are copied verbatim.
    """
    out = zT.clone()
    latent = zT.shape[-1]
    for (c, q), cs in group_cells(cells).items():
        F = block_fft(zT, c, q)
        for cell in cs:
            F = fn(F, cell)
        x = block_ifft_real(F).to(zT.dtype)
        if q is None:
            out[0, c] = x
        else:
            ys, xs = quadrant_slice(q, latent)
            out[0, c, ys, xs] = x
    return out


def _sync_device() -> torch.device:
    """Synchronisation searches run on the GPU when one is present (hundreds of FFTs per image)."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CellReader:
    """Vectorised read_cells for a fixed full-image cell list (and chips): precomputes flat pixel
    indices, chip x parity weights and cell ids per channel, then reads a batch (B, C, h, w) of
    latents into (B, n_cells) read-outs with one FFT per channel. Identical to read_cells up to
    float error; used by the rotation synchronisation search (hundreds of reads per image)."""

    def __init__(self, cells: list[Cell], chips: SectorChips | None, latent: int) -> None:
        h = w = int(latent)
        self.n = len(cells)
        per: dict[int, dict[str, list[np.ndarray]]] = {}
        for i, cell in enumerate(cells):
            if cell.q is not None:
                raise ValueError("CellReader supports full-image cells only")
            ii, jj = np.nonzero(_ring_mask_np(h, w, cell.r))
            chip = (np.ones(len(ii), np.float32) if chips is None else
                    ring_chips(h, w, cell.r, chips.n_sectors, chips.seed(cell), chips.rotate))
            par = ring_parity(h, w, cell.r) if cell.part == 1 else np.ones(len(ii), np.float32)
            d = per.setdefault(cell.c, {"idx": [], "w": [], "cid": [], "im": []})
            d["idx"].append(ii * w + jj)
            d["w"].append(chip * par)
            d["cid"].append(np.full(len(ii), i, dtype=np.int64))
            d["im"].append(np.full(len(ii), cell.part == 1))
        self.blocks = {c: {"idx": torch.from_numpy(np.concatenate(d["idx"])),
                           "w": torch.from_numpy(np.concatenate(d["w"]).astype(np.float32)),
                           "cid": torch.from_numpy(np.concatenate(d["cid"])),
                           "im": torch.from_numpy(np.concatenate(d["im"]))} for c, d in per.items()}

    def read(self, z: torch.Tensor, hard: bool = False) -> np.ndarray:
        dev = _sync_device()
        z = z.detach().float().to(dev)
        B = z.shape[0]
        num = torch.zeros(B, self.n, dtype=torch.float32, device=dev)
        den = torch.zeros(B, self.n, dtype=torch.float32, device=dev)
        for c, d0 in self.blocks.items():
            d = {k: v.to(dev) for k, v in d0.items()}
            F = torch.fft.fftshift(torch.fft.fft2(z[:, c]), dim=(-2, -1)).reshape(B, -1)
            vals = torch.where(d["im"][None, :], F.imag[:, d["idx"]], F.real[:, d["idx"]])   # (B, n_idx)
            v = vals * d["w"][None, :]                                    # chip * (parity-folded) value
            if hard:
                v = torch.sign(v)
                num.index_add_(1, d["cid"], v)
                den.index_add_(1, d["cid"], torch.ones_like(v))
            else:
                num.index_add_(1, d["cid"], v)
                den.index_add_(1, d["cid"], vals.abs())
        return (num / (den + 1e-8)).cpu().numpy().astype(np.float64)


def _affine_latents(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """(1, C, h, w) x (B, 2, 3) -> (B, C, h, w): bilinear resampling about the centre, zero padding."""
    z = z.detach().float()
    B = theta.shape[0]
    grid = torch.nn.functional.affine_grid(theta, (B, z.shape[1], z.shape[2], z.shape[3]), align_corners=False)
    return torch.nn.functional.grid_sample(z.expand(B, -1, -1, -1), grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=False)


def rotate_latents(z: torch.Tensor, angles_deg) -> torch.Tensor:
    """(1, C, h, w) -> (B, C, h, w): the latent rotated about its centre by each angle
    (bilinear, zero padding; angle 0 is the identity). # OURS rotation synchronisation."""
    ang = torch.as_tensor(np.deg2rad(np.asarray(angles_deg, dtype=np.float64)), dtype=torch.float32)
    B = ang.numel()
    theta = torch.zeros(B, 2, 3, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang), -torch.sin(ang)
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang), torch.cos(ang)
    return _affine_latents(z, theta)


def zoom_latents(z: torch.Tensor, scales) -> torch.Tensor:
    """(1, C, h, w) -> (B, C, h, w): the latent zoomed about its centre by each factor
    (s < 1 shrinks the content: undoes a crop-and-resize of area fraction f when s = sqrt(f))."""
    s = torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32)
    B = s.numel()
    theta = torch.zeros(B, 2, 3, dtype=torch.float32)
    theta[:, 0, 0] = 1.0 / s
    theta[:, 1, 1] = 1.0 / s
    return _affine_latents(z, theta)


def shift_latent(z: torch.Tensor, ty: float, tx: float) -> torch.Tensor:
    """Sub-pixel circular shift of a (1, C, h, w) latent by (ty, tx) pixels via the Fourier phase ramp
    (conjugate symmetry is kept, so the result is exactly real)."""
    z = z.detach().float()
    h, w = z.shape[-2:]
    ky = torch.fft.fftfreq(h)[:, None]
    kx = torch.fft.fftfreq(w)[None, :]
    ramp = torch.exp(2j * np.pi * (ky * ty + kx * tx)).to(torch.complex64)
    return torch.fft.ifft2(torch.fft.fft2(z) * ramp).real


class ShiftScaleSync:
    """# OURS crop synchronisation: blind translation + isotropic-scale search on the inverted latent.

    A crop of area fraction f resized back is, in the latent, a zoom by 1/sqrt(f) about the crop
    centre plus a translation. Scale: candidate zoom-out factors s (zoom_latents). Translation: for
    each candidate scale the per-position correlation for ALL sub-pixel shifts comes from one
    zero-padded inverse FFT,  y_b(t) = Re IFFT[ c ⊙ F ⊙ 1_b ](t),  where c is the chip on real-part
    cells and -i * parity * chip on imaginary-part cells, so that the sum over a position equals the
    parity-folded read-out of the shifted latent. Statistic: mean_b |y_b(t)| / sum|F| (message-
    agnostic); (s = 1, t = 0) is always among the candidates. The null calibration runs the same search.
    """

    def __init__(self, cells: list[Cell], chips: SectorChips | None, pos_of: np.ndarray | None,
                 n_pos: int, latent: int, up: int = 4, max_shift: float = 12.0) -> None:
        h = w = int(latent)
        n_ch = max(c.c for c in cells) + 1
        C = np.zeros((n_pos, n_ch, h, w), dtype=np.complex64)
        for i, cell in enumerate(cells):
            ii, jj = np.nonzero(_ring_mask_np(h, w, cell.r))
            chip = (np.ones(len(ii), np.float32) if chips is None else
                    ring_chips(h, w, cell.r, chips.n_sectors, chips.seed(cell), chips.rotate))
            val = chip.astype(np.complex64) if cell.part == 0 else (-1j * ring_parity(h, w, cell.r) * chip).astype(np.complex64)
            # accumulate: the real- and imaginary-part cells of one ring share pixels (and, with the paired
            # allocation, the same position), so their weights add instead of overwriting each other
            C[pos_of[i] if pos_of is not None else i, cell.c, ii, jj] += val
        self.C = torch.from_numpy(C)                  # (n_pos, n_ch, h, w), centred (fftshift) layout
        self.n_pos, self.h, self.w, self.up = n_pos, h, w, int(up)
        self.max_shift = float(max_shift)

    def stat_maps(self, Z: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) -> (B, U, U): statistic for every sub-pixel shift (U = up * h; index i <-> shift i/up,
        wrapped: indices above U/2 are negative shifts)."""
        dev = _sync_device()
        Z = Z.detach().float().to(dev)
        C = self.C.to(dev)
        B, U = Z.shape[0], self.up * self.h
        F = torch.fft.fftshift(torch.fft.fft2(Z), dim=(-2, -1))                     # (B, C, h, w)
        norm = F.abs().sum(dim=(1, 2, 3)) + 1e-8
        o = (U - self.h) // 2
        acc = torch.zeros(B, U, U, dtype=torch.float32, device=dev)
        for b in range(self.n_pos):
            G = (C[b][None] * F).sum(1)                                             # (B, h, w)
            Gp = torch.zeros(B, U, U, dtype=torch.complex64, device=dev)
            Gp[:, o:o + self.h, o:o + self.w] = G
            acc += torch.fft.ifft2(torch.fft.ifftshift(Gp, dim=(-2, -1))).real.abs() * (U * U)
        return (acc / norm[:, None, None]).cpu()  # ~1.1 on a freshly modulated latent, ~0.05 when misaligned

    def search(self, z: torch.Tensor, scales) -> tuple[float, float, float, float, float, float]:
        """Return (s, ty, tx, stat, stat0, z): best zoom-out factor and shift (px), the statistic there,
        the statistic at (1, 0), and the robust z-score of the peak over all candidates (peak gating)."""
        scales = np.asarray(scales, dtype=np.float64)
        Z = zoom_latents(z, scales)
        M = self.stat_maps(Z)                                                       # (S, U, U)
        U = M.shape[-1]
        idx = torch.arange(U)
        px = torch.where(idx > U // 2, idx - U, idx).float() / self.up               # shift in px per index
        ok = px.abs() <= self.max_shift
        win = (ok[:, None] & ok[None, :])[None].expand_as(M)
        vals = M[win]
        med = float(vals.median())
        mad = 1.4826 * float((vals - med).abs().median()) + 1e-9
        M = M.masked_fill(~win, -1.0)
        flat = int(torch.argmax(M.reshape(-1)))
        si, rest = divmod(flat, U * U)
        yi, xi = divmod(rest, U)
        s1 = int(np.argmin(np.abs(scales - 1.0)))
        peak = float(M[si, yi, xi])
        return float(scales[si]), float(px[yi]), float(px[xi]), peak, float(M[s1, 0, 0]), (peak - med) / mad

    @staticmethod
    def apply(z: torch.Tensor, s: float, ty: float, tx: float) -> torch.Tensor:
        """Undo the estimated crop: zoom out by s about the centre, then shift by (ty, tx) px."""
        return shift_latent(zoom_latents(z, [s]), ty, tx)

    # ---- joint rotation x zoom x shift (# OURS: combined geometric attacks, e.g. crop + rotation) ----
    def _maps_for(self, z: torch.Tensor, scales, angles, up: int, chunk: int = 512):
        """Stat maps for every (scale, angle) candidate: returns (cands [(s, a)], M (N, U, U) on CPU)."""
        cands = [(float(s), float(a)) for s in scales for a in angles]
        up0, self.up = self.up, int(up)
        maps = []
        try:
            for s in scales:
                zs = zoom_latents(z, [float(s)])
                for i in range(0, len(angles), chunk):
                    Z = rotate_latents(zs, np.asarray(angles[i:i + chunk], dtype=np.float64))
                    maps.append(self.stat_maps(Z))
        finally:
            self.up = up0
        return cands, torch.cat(maps)

    def _peak(self, M: torch.Tensor, up: int):
        U = M.shape[-1]
        idx = torch.arange(U)
        px = torch.where(idx > U // 2, idx - U, idx).float() / up
        ok = px.abs() <= self.max_shift
        win = (ok[:, None] & ok[None, :])[None].expand_as(M)
        vals = M[win]
        med = float(vals.median())
        mad = 1.4826 * float((vals - med).abs().median()) + 1e-9
        Mm = M.masked_fill(~win, -1.0)
        flat = int(torch.argmax(Mm.reshape(-1)))
        ci, rest = divmod(flat, U * U)
        yi, xi = divmod(rest, U)
        peak = float(Mm[ci, yi, xi])
        return ci, float(px[yi]), float(px[xi]), peak, (peak - med) / mad

    def search_joint(self, z: torch.Tensor, scales, angle_step: float = 3.0, fine: bool = True):
        """Coarse grid over (zoom, angle) with all shifts per candidate (up=2), then a local refinement
        (angle +-step, zoom +-0.02, up=4). Returns (s, theta, ty, tx, stat, stat0, z)."""
        scales = [float(x) for x in np.asarray(scales, dtype=np.float64)]
        angles = list(np.arange(0.0, 360.0, angle_step))
        cands, M = self._maps_for(z, scales, angles, up=2)
        ci, ty, tx, peak, zsc = self._peak(M, 2)
        s0, a0 = cands[ci]
        stat0 = float(M[cands.index((1.0, 0.0))][0, 0]) if (1.0, 0.0) in cands else float("nan")
        if fine:
            fs = [round(s0 + d, 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02) if 0.3 <= s0 + d <= 1.6]
            fa = [(a0 + d) % 360.0 for d in np.arange(-angle_step, angle_step + 1e-9, 0.5)]
            c2, M2 = self._maps_for(z, fs, fa, up=4)
            ci2, ty2, tx2, peak2, _ = self._peak(M2, 4)
            if peak2 >= peak:
                s0, a0, ty, tx, peak = c2[ci2][0], c2[ci2][1], ty2, tx2, peak2
        return s0, a0, ty, tx, peak, stat0, zsc

    @staticmethod
    def apply_joint(z: torch.Tensor, s: float, theta: float, ty: float, tx: float) -> torch.Tensor:
        return shift_latent(rotate_latents(zoom_latents(z, [s]), [theta]), ty, tx)


def read_cells(zT: torch.Tensor, cells: list[Cell], chips: SectorChips | None = None,
               hard: bool = False) -> np.ndarray:
    """Vector of y_g for every cell (same order as `cells`), FFT'd once per block."""
    y = np.zeros(len(cells), dtype=np.float64)
    index = {cell: i for i, cell in enumerate(cells)}
    for (c, q), cs in group_cells(cells).items():
        F = block_fft(zT, c, q)
        for cell in cs:
            y[index[cell]] = read_cell(F, cell, None if chips is None else chips(F, cell), hard)
    return y
