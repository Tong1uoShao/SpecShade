"""Style attack, engine 1: SDXL image-to-image with '{caption}, {style words}'.

# ASSUMPTION: img2img + style prompt approximates one-click stylisation products that
#   re-paint the image under a style description; strength = diffusion strength [6-pipeline].
"""
from __future__ import annotations

from PIL import Image

from attacks.registry import Attack

STYLES = {
    "anime": "anime style, Studio Ghibli inspired illustration, clean line art, cel shading",
    "comic": "comic book style, bold ink outlines, flat colors, halftone shading",
    "watercolor": "watercolor painting, soft color washes, paper texture",
}


class StyleImg2Img(Attack):
    engine = "style_img2img"
    heavy = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.size = int(cfg["backend"]["size"])

    def _load(self):
        if self.pipe is None:
            from core.sd_backend import SDXLImg2Img
            self.pipe = SDXLImg2Img(size=self.size)

    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "",
                 style: str = "anime") -> Image.Image:
        self._load()
        return self.pipe(image, f"{prompt}, {STYLES[style]}", strength, seed, guidance_scale=5.0, steps=50)

    def free(self) -> None:
        if self.pipe is not None:
            self.pipe.free()
            self.pipe = None
