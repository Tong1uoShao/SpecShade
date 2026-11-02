"""MaXsive baseline [maxsive2025], ported from the official code (github.com/Mao718/MaXsive, release 2025/08) so that it
runs in our pipeline with the same backend.

# PAPER: the watermark is a block w of 4 x 32 x 32 Gaussian values (4,096 values with hw_copy = 2, channel_copy = 1),
#   normalised, and the initial noise is made of 4 copies of w, each permuted by its own secret permutation (the
#   shuffle keys). An X-shaped template is injected during sampling: at every step, peaks (mean + 5 std of the
#   spectrum, with a 2 x 2 neighbourhood zeroed) are written into the Fourier spectrum of channel 3 of the x0
#   prediction, on two lines (1 deg and 135 deg) at radii 6, 9, 12 and 16. Detection: DDIM inversion, a Hough-style
#   line search (5 deg steps, lines 45 deg apart) on the windowed spectrum of channel 3 estimates the rotation (and a
#   canvas-expansion scale), the latent is rotated back, the 4 copies are un-permuted and averaged, and the score is
#   the absolute Pearson correlation with w.
# PAPER protocol: w and the shuffle keys are drawn per image and stored with it; the detector reads them back
#   (known-message verification). Here one fixed w and one set of keys, derived from the method key, are used for all
#   images (the same single-message protocol as GaussMarker in Sec. IV-C), so detection needs no per-image record.
# OURS: images are generated with the DDIM sampler of our backend (the official code uses DPM-Solver++ on SD2.1-base);
#   the template is injected into the x0 prediction of every DDIM step through SDBackend.x0_hook.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torchvision.transforms.functional as TF

from models.base import ExtractResult, WatermarkMethod

N = 64


class MaXsiveWM(WatermarkMethod):
    name = "maxsive"
    fixed_payload = np.array([1])

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("maxsive", {}) or {}
        self.ch, self.hw = int(m.get("channel_copy", 1)), int(m.get("hw_copy", 2))
        self.template_c = int(m.get("template_c", 3))
        self.template_scale = float(m.get("template_scale", 5.0))
        self.distant = str(m.get("distant_func", "corr"))
        self.theta, self.lengths = [1, 135], [0.2, 0.3, 0.4, 0.5]
        key = str(m.get("key", "maxsive-key-v1")).encode()
        g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 31))
        self.block = (1, 4 // self.ch, N // self.hw, N // self.hw)
        w = torch.randn(*self.block, generator=g)
        self.w = w / w.std() - w.mean()                                  # as in the official code
        n = int(np.prod(self.block))
        self.keys = [torch.randperm(n, generator=g) for _ in range(self.ch * self.hw * self.hw)]
        self.z_w = self._shuffle(self.w)
        win = torch.bartlett_window(N)
        self.window = torch.sqrt(torch.outer(win, win))
        self.points = self._template_points()
        self._init_hough(32, 32, N, 5)

    # ---------- noise ----------
    def _shuffle(self, w: torch.Tensor) -> torch.Tensor:
        flat, shape = w.reshape(-1), w.shape
        chans = []
        for i in range(self.ch):
            rows = []
            for j in range(self.hw):
                rows.append(torch.cat([flat[self.keys[i * self.hw * self.hw + j * self.hw + k]].reshape(shape)
                                       for k in range(self.hw)], dim=3))
            chans.append(torch.cat(rows, dim=2))
        return torch.cat(chans, dim=1)

    def _unshuffle(self, z: torch.Tensor) -> list[torch.Tensor]:
        ch_stride, hw_stride = 4 // self.ch, N // self.hw
        s1 = torch.cat(torch.split(z, hw_stride, dim=3), dim=0)
        s2 = torch.cat(torch.split(s1, hw_stride, dim=2), dim=0)
        s3 = torch.cat(torch.split(s2, ch_stride, dim=1), dim=0)
        n = int(np.prod(self.block))
        out = []
        for i in range(self.ch):
            for j in range(self.hw):
                for k in range(self.hw):
                    idx = i * self.hw * self.hw + j * self.hw + k
                    order = torch.zeros(n, dtype=torch.long)
                    order[self.keys[idx]] = torch.arange(n)
                    out.append(s3[idx].reshape(-1)[order].reshape(self.block))
        return out

    # ---------- template (sampling hook) ----------
    def _template_points(self):
        pts = []
        for length in self.lengths:
            L = int(32 * length)
            for th in self.theta:
                pts.append((32 + int(L * math.cos(math.radians(th))), 32 + int(L * math.sin(math.radians(th)))))
        return pts

    def peak_injection(self, x0: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fftshift(torch.fft.fft2(x0.float()), dim=(-1, -2))
        for x, y in self.points:
            mean, std = f.mean(), f.std()
            f = f.clone()
            f[0, self.template_c, x - 1:x + 1, y - 1:y + 1] = 0
            f[0, self.template_c, x, y] = mean + self.template_scale * std
        return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-1, -2))).real

    def prepare_backend(self, backend) -> None:
        backend.x0_hook = self.peak_injection

    # ---------- interface ----------
    @property
    def n_bits(self) -> int:
        return 1

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        return self.z_w.clone().float()

    # ---------- detection: Hough line search on the template, rotation restore, vote, correlation ----------
    def _init_hough(self, x: int, y: int, dim: int, interval: int) -> None:
        self.degrees = [interval * i for i in range(int(180 / interval))]
        self.interval = interval
        mask = torch.zeros(len(self.degrees), dim, dim)
        for i, deg in enumerate(self.degrees):
            for r in range(int(dim / 2 * 2 ** 0.5)):
                xr, yr = int(r * math.cos(math.radians(deg))), int(r * math.sin(math.radians(deg)))
                if abs(xr) > int(dim / 2) - 1 or abs(yr) > int(dim / 2) - 1:
                    break
                mask[i, x + xr, y + yr] = 1
                mask[i, x - xr, y - yr] = 1
        self.hmask, self.hw_sum = mask, mask.sum(dim=(1, 2))

    def _detect_scale(self, spec: torch.Tensor, include_angle: int, predict_angle: float) -> bool:
        hist = []
        for bw in (-2, -1, 0, 1, 2):
            degs = [predict_angle - include_angle + bw, predict_angle + bw]
            mask = torch.zeros(int(N / 2 * 2 ** 0.5), N, N)
            for r in range(int(N / 2 * 2 ** 0.5)):
                for deg in degs:
                    xr, yr = int(r * math.cos(math.radians(deg))), int(r * math.sin(math.radians(deg)))
                    if abs(xr) > int(N / 2) - 1 or abs(yr) > int(N / 2) - 1:
                        break
                    mask[r, 32 + xr, 32 + yr] = 1
                    mask[r, 32 - xr, 32 - yr] = 1
            test = spec.abs() * mask
            rs = test.reshape(45, -1).sum(1) / mask.reshape(45, -1).sum(1)
            if rs[16] > 50 and rs[16] > rs[17] and rs[16] > rs[18]:
                hist.append(False)
            hist.append(True)
        return False not in hist

    def _template_restore(self, z: torch.Tensor, include_angle: int = 45, pre_angle: int = 135):
        f = torch.fft.fftshift(torch.fft.fft2(z.float()), dim=(-1, -2))
        spec = f[0, self.template_c] * self.window
        x = spec.abs()
        score = (x[None] * self.hmask).sum(dim=(1, 2)) / self.hw_sum
        gap = int(include_angle / self.interval)
        n = len(self.degrees)
        pair = torch.tensor([float(score[i] + score[(i + gap) % n]) for i in range(n)])
        best = int(torch.argmax(pair))
        degree = self.degrees[best] - pre_angle
        if_scale = self._detect_scale(spec, include_angle, degree)
        if degree < 0:
            degree += 180
        if if_scale and degree != 0:
            scale = math.sin(abs(degree % 90) / 180 * math.pi) + math.cos(abs(degree % 90) / 180 * math.pi)
            z = TF.resize(z, int(N / scale))
            z = TF.resize(TF.pad(z, int((N - z.shape[2]) / 2)), N)
        return TF.rotate(z, -degree), {"degree": float(degree), "if_scale": bool(if_scale), "line_score": float(pair[best])}

    @torch.no_grad()
    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().cpu().float()
        z, meta = self._template_restore(z)
        vote = torch.mean(torch.cat(self._unshuffle(z)), dim=0).reshape(1, -1)
        w = self.w.reshape(1, -1)
        if self.distant == "corr":
            score = abs(float(torch.corrcoef(torch.cat([w, vote]))[0, 1]))
        elif self.distant == "cos":
            score = float(torch.nn.functional.cosine_similarity(vote, w))
        elif self.distant == "mse":
            score = -float(torch.nn.functional.mse_loss(vote, w))
        else:
            score = -float((vote - w).abs().mean())
        return ExtractResult(bits=None, llr=None, accepted=None, score=score, meta=meta)
