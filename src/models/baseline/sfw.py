"""SFW baselines [lee2025sfw]: HSTR and HSQR (centre-aware variants), re-implemented from the official code
(github.com/thomas11809/SFWMark, CC BY-NC 4.0) so that they run in our offline pipeline with the same backend.

# PAPER: the watermark is written into the Fourier spectrum of the central 44x44 window (rows/cols 10..53) of the
#   initial noise, so that cropping keeps it.
#   HSTR: Tree-Ring-like constant rings (radii 1..21, one key-noise value per ring, all channels) with Hermitian
#     symmetry enforced, plus a Gaussian key noise in the ring band r in (3, 14] of channel 0 (the heterogeneous
#     watermark of RingID). Detection: min over channels {0, 3} of the mean L1 distance between the pattern and the
#     spectrum of the central window inside the masks (channel 3: disc r <= 14, channel 0: ring band).
#   HSQR: a version-1 QR code (21x21 modules, 2 px per module -> 42x42) encoding "HSQR{idx}" is written as the signs of
#     the real (left half) and imaginary (right half) parts of the rfft of the central window of channel 3.
#     Detection: mean L1 distance between the +-45 QR pattern and the corresponding spectrum slice.
#   Protocol: 2048 keys (indices 0..2047; HSTR seeds 7433 + idx); verification by the distance to the known pattern,
#     identification by the argmin over all keys; SD2.1-base in the official code.
# OURS: the payload is the 11-bit key index. extract() runs the identification over all 2048 patterns (the detector is
#   not told the message) and returns score = -min distance and the bits of the nearest key. The images are generated
#   with the same SD1.5 backend as every other method.
"""
from __future__ import annotations

import numpy as np
import torch

from models.base import ExtractResult, WatermarkMethod

N, START, END = 64, 10, 54                 # latent size and the central window
CEN = (slice(None), slice(None), slice(START, END), slice(START, END))
RADIUS, RADIUS_CUTOFF = 14, 3
CH_TREE, CH_HET = 3, 0
W_SEED, N_KEYS = 7433, 2048
QR_LEN, QR_HALF = 42, 21


def _fft(x):
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-1, -2))


def _ifft(x):
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-1, -2)))


def circle_mask(size: int, r: int) -> np.ndarray:
    x0 = y0 = size // 2
    y, x = np.ogrid[:size, :size]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def rounder_ring_mask(r_out: int = RADIUS, r_in: int = RADIUS_CUTOFF, size: int = 65) -> np.ndarray:
    """The 'rounder' ring band of the official code: a line of ring labels rotated over 360 degrees, majority label per
    pixel; mask = pixels whose label belongs to the rings r_in .. r_out-1 (cropped to 64x64)."""
    import torchvision.transforms.functional as TF
    ring_vector = torch.tensor([(200 - i * 4) * (-1) ** i for i in range(r_out)], dtype=torch.float32)
    bg = torch.zeros(size, size)
    c = size // 2
    bg[c, c:c + r_out] = ring_vector
    bg = bg[None, None]
    res = torch.stack([bg[0, 0]] + [TF.rotate(bg, angle=a)[0, 0] for a in range(1, 360)]).numpy()
    pure = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            values, count = np.unique(res[:, x, y], return_counts=True)
            if len(count) > 2:
                pure[x, y] = values[count == max(count[values != 0])][0]
            elif len(count) == 2:
                pure[x, y] = values[values != 0][0]
    right_end = 0 if r_in - 1 < 0 else r_in - 1
    cand = ring_vector.numpy()[r_out - 1:right_end:-1]
    mask = np.isin(pure, cand)
    return mask[:size - 1, :size - 1]


def enforce_hermitian_symmetry(f: torch.Tensor) -> torch.Tensor:
    """Official SFW enforcement on an fftshift'ed (B, C, H, W) spectrum with even H == W."""
    B, C, H, W = f.shape
    f = f.clone()
    t = f.clone()
    f[:, :, H // 2, W // 2] = torch.real(t[:, :, H // 2, W // 2])
    f[:, :, 0, 0] = torch.real(t[:, :, 0, 0])
    f[:, :, H // 2, 0] = torch.real(t[:, :, H // 2, 0])
    f[:, :, 0, W // 2] = torch.real(t[:, :, 0, W // 2])
    f[:, :, 0, 1:W // 2] = torch.conj(torch.flip(t[:, :, 0, W // 2 + 1:], dims=[2]))
    f[:, :, H // 2, 1:W // 2] = torch.conj(torch.flip(t[:, :, H // 2, W // 2 + 1:], dims=[2]))
    f[:, :, 1:H // 2, 0] = torch.conj(torch.flip(t[:, :, H // 2 + 1:, 0], dims=[2]))
    f[:, :, 1:H // 2, W // 2] = torch.conj(torch.flip(t[:, :, H // 2 + 1:, W // 2], dims=[2]))
    f[:, :, 1:H // 2, 1:W // 2] = torch.conj(torch.flip(t[:, :, H // 2 + 1:, W // 2 + 1:], dims=[2, 3]))
    f[:, :, H // 2 + 1:, 1:W // 2] = torch.conj(torch.flip(t[:, :, 1:H // 2, W // 2 + 1:], dims=[2, 3]))
    return f


def hstr_pattern(idx: int, rmin: torch.Tensor) -> torch.Tensor:
    """(1, 4, 64, 64) complex HSTR pattern of key idx. rmin[x, y] = smallest radius r in 1..21 whose disc contains
    (x, y) on the 64x64 grid (0 outside r = 21): the official loop assigns the ring values from r = 21 down to 1, so the
    value that survives at a pixel is that of the smallest disc containing it."""
    g = torch.Generator().manual_seed(W_SEED + idx)
    gt_init = torch.randn((1, 4, N, N), generator=g)
    gt_patch = _fft(gt_init[CEN])                                      # (1, 4, 44, 44)
    c = gt_patch.shape[-1] // 2                                        # 22
    vals = gt_patch[0, :, c, c:c + 22]                                 # (4, 22): value of ring r at column c + r
    patt = torch.zeros((1, 4, N, N), dtype=torch.complex64)
    inside = rmin > 0
    patt[0, :, inside] = vals[:, rmin[inside]]
    patt[:, CH_HET, START:END, START:END] = gt_patch[:, CH_HET]      # heterogeneous Gaussian key in channel 0
    return enforce_hermitian_symmetry(patt)


def qr_bool(idx: int) -> torch.Tensor:
    """(42, 42) boolean QR module image of "HSQR{idx}" (version 1, 2 px per module, no border), as in the official code."""
    import qrcode
    qr = qrcode.QRCode(version=1, box_size=2, border=0, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(f"HSQR{idx % 10000}")
    qr.make(fit=True)
    arr = np.array(qr.make_image(fill_color="black", back_color="white"))
    assert arr.shape == (QR_LEN, QR_LEN), arr.shape
    return torch.from_numpy(arr).bool()


class SFWWM(WatermarkMethod):
    name = "sfw"

    def __init__(self, cfg: dict, backend, variant: str = "hstr") -> None:
        super().__init__(cfg, backend)
        self.variant = variant
        self.n_keys = N_KEYS
        self.size = backend.latent if backend is not None else N
        assert self.size == N
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if variant == "hstr":
            m_tree = torch.from_numpy(circle_mask(N, RADIUS))
            m_het = torch.from_numpy(rounder_ring_mask())
            self.inj_mask = torch.zeros((1, 4, N, N), dtype=torch.bool)
            self.inj_mask[:, CH_TREE] = m_tree
            self.inj_mask[:, CH_HET] = m_het
            self.m_het_c, self.m_tree_c = m_het[START:END, START:END], m_tree[START:END, START:END]
            d = torch.sqrt(((torch.arange(N) - N // 2)[:, None] ** 2 + (torch.arange(N) - N // 2)[None, :] ** 2).float())
            rmin = torch.ceil(d - 1e-6).long()
            rmin[(rmin < 1) | (rmin > 21)] = 0
            rmin[(d <= 1e-6)] = 1                                     # the centre lies in every disc; smallest radius wins
            self.rmin = rmin
            P0, P3, self.patterns = [], [], []
            for k in range(self.n_keys):
                p = hstr_pattern(k, rmin)
                self.patterns.append(p)
                P0.append(p[0, CH_HET, START:END, START:END][self.m_het_c])
                P3.append(p[0, CH_TREE, START:END, START:END][self.m_tree_c])
            self.P0, self.P3 = torch.stack(P0).to(dev), torch.stack(P3).to(dev)
        elif variant == "hsqr":
            self.qrs = torch.stack([qr_bool(k) for k in range(self.n_keys)])              # (K, 42, 42)
            q = torch.where(self.qrs, torch.tensor(45.0), torch.tensor(-45.0))
            self.Q = torch.complex(q[:, :, :QR_HALF], q[:, :, QR_HALF:]).to(dev)         # (K, 42, 21)
        else:
            raise ValueError(variant)
        self.dev = dev

    @property
    def n_bits(self) -> int:
        return 11                                                     # log2(2048) key index

    @staticmethod
    def _idx(payload: np.ndarray) -> int:
        return int("".join(str(int(b)) for b in payload[:11]), 2)

    @staticmethod
    def _bits(idx: int) -> np.ndarray:
        return np.array([int(c) for c in format(idx, "011b")], dtype=np.int64)

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        idx = self._idx(payload)
        z = self.backend.sample_noise(seed, latent=self.size).detach().cpu().float().clone()
        if self.variant == "hstr":
            cf = _fft(z[CEN])
            m = self.inj_mask[CEN]
            cf[m] = self.patterns[idx][CEN][m]
            z[CEN] = _ifft(cf).real
        else:
            qr = self.qrs[idx]
            r = torch.fft.fftshift(torch.fft.rfft2(z[CEN], dim=(-2, -1)), dim=-2)          # (1, 4, 44, 23)
            re, im = r.real.clone(), r.imag.clone()
            sl = (slice(None), [CH_TREE], slice(1, 1 + QR_LEN), slice(1, 1 + QR_HALF))
            re[sl] = torch.where(qr[:, :QR_HALF], re[sl].abs(), -re[sl].abs())
            im[sl] = torch.where(qr[:, QR_HALF:], im[sl].abs(), -im[sl].abs())
            z[CEN] = torch.fft.irfft2(torch.fft.ifftshift(torch.complex(re, im), dim=-2), dim=(-2, -1), s=(END - START, END - START))
        z[z == float("inf")] = 4.0
        z[z == float("-inf")] = -4.0
        return z

    @torch.no_grad()
    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().cpu().float()
        cf = _fft(z[CEN])[0]                                                                # (4, 44, 44)
        if self.variant == "hstr":
            t0, t3 = cf[CH_HET][self.m_het_c].to(self.dev), cf[CH_TREE][self.m_tree_c].to(self.dev)
            d = torch.minimum((self.P0 - t0[None]).abs().mean(1), (self.P3 - t3[None]).abs().mean(1))
        else:
            t = cf[CH_TREE, 1:1 + QR_LEN, 23:23 + QR_HALF].to(self.dev)                      # rows 11:53, cols 33:54 on the 64 grid
            d = (self.Q - t[None]).abs().mean(dim=(1, 2))
        k = int(torch.argmin(d))
        return ExtractResult(bits=self._bits(k), llr=None, accepted=None, score=-float(d[k]),
                             meta={"idx": k, "d_min": float(d[k]), "d_med": float(d.median())})
