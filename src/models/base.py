"""Uniform watermark-method interface shared by StyleRing and all baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExtractResult:
    bits: np.ndarray | None        # decoded payload bits (None for 0-bit methods / rejected)
    llr: np.ndarray | None         # soft read-outs used for decoding (None if n/a)
    accepted: bool | None          # ECC/syndrome accept flag (None if method has no reject rule)
    score: float                   # 'larger = more watermark-like', used for TPR@FPR
    n_corrected: int = -1          # corrected bit errors (ECC methods)
    meta: dict | None = None       # optional extras written to detect.jsonl (e.g. synchronisation angle)


class WatermarkMethod:
    name: str = "base"
    # OURS: image-level methods (post-hoc encoders, decoder fine-tuning) do not live in the
    #   initial noise: they watermark / read the *image* and need no DDIM inversion.
    image_level: bool = False
    fixed_payload: np.ndarray | None = None    # methods with a single baked-in key (Stable Signature)

    def __init__(self, cfg: dict, backend) -> None:
        self.cfg = cfg
        self.backend = backend

    @property
    def n_bits(self) -> int:
        raise NotImplementedError

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        """payload bits + generation seed -> watermarked initial noise [1,4,H,W] (float32, CPU)."""
        raise NotImplementedError

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        """estimated initial noise -> ExtractResult."""
        raise NotImplementedError

    # ---- image-level hooks (no-ops for noise-level methods) ----
    def prepare_backend(self, backend) -> None:
        """Called once before generation (e.g. swap in a fine-tuned VAE decoder)."""

    def embed_image(self, image, payload: np.ndarray):
        """Post-hoc: watermark a generated PIL image (returns a PIL image)."""
        return image

    def extract_image(self, image, ref_payload: np.ndarray | None = None) -> ExtractResult:
        """Read the watermark from a PIL image; ref_payload = message the image is expected to carry
        (score = bit agreement with it for methods without a payload-free statistic)."""
        raise NotImplementedError


class NoWatermark(WatermarkMethod):
    """Control: plain generation (FPR calibration, quality t-test)."""
    name = "none"

    @property
    def n_bits(self) -> int:
        return 0

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        return self.backend.sample_noise(seed, latent=self.backend.latent)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        return ExtractResult(bits=None, llr=None, accepted=None, score=float("nan"))
