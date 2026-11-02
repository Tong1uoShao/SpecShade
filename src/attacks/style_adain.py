"""Style attack, engine 3: AdaIN arbitrary style transfer [18] (classic NST axis).

Weights: naoto0804/pytorch-AdaIN release files vgg_normalised.pth + decoder.pth under
data/weights/adain/; style exemplars data/styles/{anime,comic,watercolor}.png.
# ASSUMPTION: strength = alpha (content/style interpolation), 0.3 / 0.6 / 1.0.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from attacks.registry import Attack


def _vgg():
    return nn.Sequential(
        nn.Conv2d(3, 3, 1), nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(), nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3), nn.ReLU(),   # relu4_1 (index 31)
    )


def _decoder():
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(), nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
    )


def _stats(x, eps=1e-5):
    n, c = x.shape[:2]
    var = x.view(n, c, -1).var(dim=2) + eps
    return x.view(n, c, -1).mean(dim=2).view(n, c, 1, 1), var.sqrt().view(n, c, 1, 1)


def adain(content, style):
    cm, cs = _stats(content)
    sm, ss = _stats(style)
    return (content - cm) / cs * ss + sm


class StyleAdaIN(Attack):
    engine = "style_adain"
    heavy = False

    def __init__(self, cfg: dict):
        root = Path(cfg["_root"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.size = int(cfg["backend"]["size"])
        wdir = root / "data/weights/adain"
        vgg = _vgg()
        # official checkpoint carries layers beyond relu4_1 -> non-strict load (extra keys only)
        vgg.load_state_dict(torch.load(wdir / "vgg_normalised.pth", map_location="cpu", weights_only=True),
                            strict=False)
        self.enc = nn.Sequential(*list(vgg.children())[:31]).to(self.device).eval()
        self.dec = _decoder()
        self.dec.load_state_dict(torch.load(wdir / "decoder.pth", map_location="cpu", weights_only=True))
        self.dec = self.dec.to(self.device).eval()
        self.style_dir = root / "data/styles"

    def _to_t(self, im: Image.Image) -> torch.Tensor:
        import numpy as np
        a = np.asarray(im.convert("RGB").resize((self.size, self.size)), dtype="float32") / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "",
                 style: str = "anime") -> Image.Image:
        c = self._to_t(image)
        s = self._to_t(Image.open(self.style_dir / f"{style}.png"))
        fc, fs = self.enc(c), self.enc(s)
        t = adain(fc, fs)
        t = float(strength) * t + (1 - float(strength)) * fc
        out = self.dec(t).clamp(0, 1)
        arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        return Image.fromarray(arr).resize((self.size, self.size))
