"""Style attack, engine 2: InstructPix2Pix instruction-driven stylisation [17].

# ASSUMPTION: strength = image_guidance_scale (2.0 mild -> 1.0 strong); text guidance 7.5, 30 steps.
"""
from __future__ import annotations

import os

import torch
from PIL import Image

from attacks.registry import Attack

IP2P_MODEL_ID = "timbrooks/instruct-pix2pix"
INSTRUCTIONS = {
    "anime": "turn it into an anime illustration",
    "comic": "make it a comic book drawing",
    "watercolor": "make it a watercolor painting",
}


class StyleIP2P(Attack):
    engine = "style_ip2p"
    heavy = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.size = int(cfg["backend"]["size"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self.pipe is None:
            from diffusers import StableDiffusionInstructPix2PixPipeline
            from core.sd_backend import local_snapshot_path
            mid = IP2P_MODEL_ID
            kw = dict(torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                      safety_checker=None, requires_safety_checker=False)
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                mid = local_snapshot_path(IP2P_MODEL_ID) or mid
                kw["local_files_only"] = True
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(mid, **kw).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "",
                 style: str = "anime") -> Image.Image:
        self._load()
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(INSTRUCTIONS[style], image=image.convert("RGB"), num_inference_steps=30,
                        guidance_scale=7.5, image_guidance_scale=float(strength), generator=g).images[0]
        return out.resize((self.size, self.size))

    def free(self) -> None:
        import gc
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
