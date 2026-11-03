"""Regeneration-type attacks (GaussMarker conventions [4]): diffusion re-noising cycles and
learned VAE compression (CompressAI).

# ASSUMPTION: diffusion_regen = `cycles` rounds of SDXL img2img with an EMPTY prompt at
#   strength 0.3 (same-content noising/denoising, no semantic guidance) -- stands in for
#   GaussMarker's ImageNet-DM regeneration on our cached models.
"""
from __future__ import annotations

import torch
from PIL import Image

from attacks.registry import Attack


class DiffusionRegen(Attack):
    engine = "diffusion_regen"
    heavy = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.size = int(cfg["backend"]["size"])

    def _load(self):
        if self.pipe is None:
            from core.sd_backend import SDXLImg2Img
            self.pipe = SDXLImg2Img(size=self.size)

    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "") -> Image.Image:
        self._load()
        im = image.convert("RGB")
        for k in range(int(strength)):
            im = self.pipe(im, "", 0.3, int(seed) + k, guidance_scale=1.0, steps=50)
        return im

    def batch(self, images: list[Image.Image], strength: float, seeds: list[int], prompts=None) -> list[Image.Image]:
        """Batched form of __call__ (same cycles and per-image seeds)."""
        self._load()
        ims = [im.convert("RGB") for im in images]
        for k in range(int(strength)):
            ims = self.pipe.batch(ims, [""] * len(ims), 0.3, [int(s) + k for s in seeds], guidance_scale=1.0, steps=50)
        return ims

    def free(self) -> None:
        if self.pipe is not None:
            self.pipe.free()
            self.pipe = None


class VAECompress(Attack):
    engine = "vae_compress"
    heavy = False

    def __init__(self, cfg: dict, model: str = "cheng2020-anchor"):
        import compressai.zoo as zoo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model
        self.zoo = zoo
        self.nets: dict[int, torch.nn.Module] = {}
        self.size = int(cfg["backend"]["size"])

    def _net(self, quality: int):
        if quality not in self.nets:
            fn = self.zoo.cheng2020_anchor if self.model_name == "cheng2020-anchor" else self.zoo.bmshj2018_factorized
            self.nets[quality] = fn(quality=int(quality), pretrained=True).eval().to(self.device)
        return self.nets[quality]

    @torch.no_grad()
    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "") -> Image.Image:
        import numpy as np
        net = self._net(int(strength))
        a = np.asarray(image.convert("RGB").resize((self.size, self.size)), np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = net(x)["x_hat"].clamp(0, 1)
        arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)
