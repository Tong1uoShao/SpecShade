"""Classical distortion battery (Tree-Ring [1] / Gaussian Shading [2] / RingID [3] conventions)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from attacks.registry import Attack


class Identity(Attack):
    engine = "none"

    def __call__(self, image, strength=0.0, seed=0, prompt=""):
        return image.convert("RGB")


class JPEG(Attack):
    engine = "jpeg"

    def __call__(self, image, strength, seed=0, prompt=""):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=int(strength))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class Rotate(Attack):
    engine = "rotate"

    def __call__(self, image, strength, seed=0, prompt=""):
        return image.convert("RGB").rotate(float(strength), resample=Image.BICUBIC)   # canvas kept, corners black


class CropScale(Attack):
    engine = "crop_scale"

    def __call__(self, image, strength, seed=0, prompt=""):
        im = image.convert("RGB")
        w, h = im.size
        frac = float(strength)
        cw, ch = int(w * np.sqrt(frac)), int(h * np.sqrt(frac))
        rng = np.random.default_rng(int(seed))
        x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)


class GaussBlur(Attack):
    engine = "gauss_blur"

    def __call__(self, image, strength, seed=0, prompt=""):
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(strength)))


class GaussNoise(Attack):
    engine = "gauss_noise"

    def __call__(self, image, strength, seed=0, prompt=""):
        a = np.asarray(image.convert("RGB"), np.float32) / 255.0
        rng = np.random.default_rng(int(seed))
        a = np.clip(a + rng.normal(0, float(strength), a.shape), 0, 1)
        return Image.fromarray((a * 255).round().astype(np.uint8))


class Brightness(Attack):
    engine = "brightness"

    def __call__(self, image, strength, seed=0, prompt=""):
        return ImageEnhance.Brightness(image.convert("RGB")).enhance(float(strength))


class RandDrop(Attack):
    engine = "rand_drop"

    def __call__(self, image, strength, seed=0, prompt=""):
        a = np.asarray(image.convert("RGB")).copy()
        rng = np.random.default_rng(int(seed))
        mask = rng.random(a.shape[:2]) < float(strength)
        a[mask] = 0
        return Image.fromarray(a)


class Median(Attack):
    engine = "median"

    def __call__(self, image, strength, seed=0, prompt=""):
        return image.convert("RGB").filter(ImageFilter.MedianFilter(size=int(strength)))


class Resize(Attack):
    engine = "resize"

    def __call__(self, image, strength, seed=0, prompt=""):
        im = image.convert("RGB")
        w, h = im.size
        small = im.resize((max(1, int(w * float(strength))), max(1, int(h * float(strength)))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)


class Mirror(Attack):
    engine = "mirror"

    def __call__(self, image, strength, seed=0, prompt=""):
        im = image.convert("RGB")                       # strength 0 = horizontal mirror, 1 = vertical flip
        return im.transpose(Image.FLIP_LEFT_RIGHT if int(float(strength)) == 0 else Image.FLIP_TOP_BOTTOM)


CLASSICAL = {"mirror": Mirror, "jpeg": JPEG, "rotate": Rotate, "crop_scale": CropScale, "gauss_blur": GaussBlur,
             "gauss_noise": GaussNoise, "brightness": Brightness, "rand_drop": RandDrop,
             "median": Median, "resize": Resize}
