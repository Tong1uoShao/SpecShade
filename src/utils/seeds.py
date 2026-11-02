"""Fixed seed protocol and reproducible payload derivation."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

SEEDS = [97, 101, 202, 303, 404]   # OURS: multi-seed protocol (generation & attacks)


def derive_payload(prompt_id: str, gen_seed: int, n_bits: int) -> np.ndarray:
    """Reproducible pseudo-random payload for (prompt_id, gen_seed) -- avoids a single
    global watermark being 'memorised' by the detector's FPR calibration."""
    h = hashlib.sha256(f"{prompt_id}|{gen_seed}|{n_bits}".encode()).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    while bits.size < n_bits:
        h = hashlib.sha256(h).digest()
        bits = np.concatenate([bits, np.unpackbits(np.frombuffer(h, dtype=np.uint8))])
    return bits[:n_bits].astype(int)


def torch_gen(seed: int, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def per_image_seed(base_seed: int, index: int) -> int:
    """Distinct but reproducible seed per image within one run."""
    return int(base_seed * 100003 + index) % (2 ** 31 - 1)
