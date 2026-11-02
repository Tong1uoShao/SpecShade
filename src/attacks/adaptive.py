"""Extension B2: method-aware (white-box) low-frequency perturbation attack.

# OURS: attacker knows StyleRing lives in low-radius rings: invert the image, add random
#   perturbation of relative amplitude eps to Re F on rings r <= r_cut of every channel,
#   then re-generate from the perturbed noise with the original caption. Reported together
#   with the image-quality cost (PSNR/CLIP vs. the unattacked image) in evaluate.py.
"""
from __future__ import annotations

import torch
from PIL import Image

from attacks.registry import Attack
from core.fourier_rings import apply_to_cells, cell_mask_t, enumerate_cells


class LowFreqPerturb(Attack):
    engine = "adaptive_lowfreq"
    heavy = True

    def __init__(self, cfg: dict, r_cut: int = 12):
        self.cfg = cfg
        self.backend = None
        self.r_cut = r_cut

    def _load(self):
        if self.backend is None:
            from core.sd_backend import SDBackend
            b = self.cfg["backend"]
            self.backend = SDBackend(model_id=b["model_id"], prefer_fp16=b["prefer_fp16"], size=b["size"])

    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "") -> Image.Image:
        self._load()
        b = self.cfg["backend"]
        zhat = self.backend.invert(image, steps=b["inv_steps"]).cpu()
        g = torch.Generator().manual_seed(int(seed))
        cells = [c for c in enumerate_cells(latent=self.backend.latent) if c.r <= self.r_cut]

        def perturb(F, cell):
            m = cell_mask_t(F, cell)
            re = F.real.clone()
            noise = torch.randn(int(m.sum()), generator=g) * float(strength) * re[m].abs().mean()
            re[m] = re[m] + noise
            # keep central symmetry of the real part: symmetrise explicitly
            re = 0.5 * (re + torch.flip(torch.roll(re, shifts=(-1, -1), dims=(0, 1)), dims=(0, 1)))
            return torch.complex(re, F.imag.clone())

        z_pert = apply_to_cells(zhat, cells, perturb)
        return self.backend.generate(z_pert, prompt, steps=b["gen_steps"], cfg=b["cfg"], seed=seed)

    def free(self) -> None:
        if self.backend is not None:
            self.backend.free()
            self.backend = None
