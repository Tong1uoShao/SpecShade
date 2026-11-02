"""GaussMarker baseline [4] -- 'lite' (dual-domain injector + score fusion, without the
learnable Gaussian Noise Restorer unless a trained GNR checkpoint path is configured).

# PAPER [4] §3.1: (1) spatial multi-bit = Gaussian-Shading-chacha sign instruction
#   z_s = |z| * (2s-1) with s = shuffle(upsample(bits)); (2) frequency zero-bit ring
#   (Tree-Ring 'ring' pattern, channel 3, small radius) injected on top of z_s;
#   detection fuses spatial bit-vote score and the ring L1 score.
# OURS: fusion = spatial vote confidence - lambda * normalised ring L1 (their exact fusion
#   is score-level too; constants documented in Part 3). GNR: if `gnr_path` is set, the
#   restorer from vendor/GaussMarker (train_GNR.py) is applied to z_hat before reading;
#   otherwise this is GaussMarker-lite (noted in dev_log / results tables).
"""
from __future__ import annotations

import numpy as np
import torch

from models.base import ExtractResult, WatermarkMethod
from models.baseline._treering import circle_mask, inject_complex, l1_complex, ring_pattern
from models.baseline.gaussian_shading import GaussianShadingWM


class GaussMarkerWM(WatermarkMethod):
    name = "gaussmarker"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("gaussmarker", {})
        sub = {"methods": {"gaussian_shading": {"n_bits": m.get("n_bits", cfg["stylering"]["n_bits"]),
                                                 "key": m.get("key", cfg["stylering"]["key"])}},
               "stylering": cfg["stylering"]}
        self.spatial = GaussianShadingWM(sub, backend)
        self.key = str(m.get("key", cfg["stylering"]["key"])).encode()
        self.w_channel = int(m.get("w_channel", 3))
        self.w_radius = int(m.get("w_radius", 4))
        self.lam = float(m.get("fuse_lambda", 1.0))
        self.size = backend.latent
        self.mask = circle_mask(self.size, self.w_radius)
        self.mask[self.size // 2, self.size // 2] = False        # keep DC untouched
        self.pattern = ring_pattern(self.key + b"-ring", self.size, self.w_channel, self.w_radius)
        self.l1_norm = float(self.pattern[torch.from_numpy(self.mask)].abs().mean()) + 1e-6
        self.gnr = None
        gnr_path = str(m.get("gnr_path", "") or "")
        if gnr_path:
            self.gnr = self._load_gnr(gnr_path, int(m.get("model_nf", 128)))

    @property
    def n_bits(self) -> int:
        return self.spatial.n_bits

    def _load_gnr(self, path: str, nf: int):
        import sys
        from pathlib import Path
        vend = Path(__file__).resolve().parent / "vendor" / "GaussMarker"
        sys.path.insert(0, str(vend))
        from unet.unet_model import UNet  # type: ignore  # vendored GaussMarker GNR
        net = UNet(4, 4, nf=nf)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net.eval()

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        z_s = self.spatial.embed(payload, seed)                               # spatial multi-bit
        return inject_complex(z_s, self.w_channel, self.mask, self.pattern)     # + zero-bit ring

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        zT_hat = zT_hat.detach().cpu().float()
        if self.gnr is not None:
            with torch.no_grad():
                zT_hat = self.gnr(zT_hat)
        sp = self.spatial.extract(zT_hat)
        l1 = l1_complex(zT_hat, self.w_channel, self.mask, self.pattern) / self.l1_norm
        return ExtractResult(bits=sp.bits, llr=sp.llr, accepted=None,
                             score=float(sp.score - self.lam * l1))
