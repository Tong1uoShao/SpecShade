"""Generate the three AdaIN style exemplars (anime / comic / watercolor) with SD1.5 (GPU, once).
# ASSUMPTION: synthetic exemplars keep the whole attack battery self-contained/offline.
Run: LD_PRELOAD=... HF_HUB_OFFLINE=1 python scripts/make_styles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.sd_backend import SDBackend  # noqa: E402

PROMPTS = {
    "anime": "anime style illustration of a town street, Studio Ghibli, clean line art, cel shading, vivid colors",
    "comic": "comic book style drawing of a city scene, bold ink outlines, flat colors, halftone shading",
    "watercolor": "watercolor painting of a garden, soft color washes, paper texture, loose brush strokes",
}

if __name__ == "__main__":
    out = ROOT / "data/styles"
    out.mkdir(parents=True, exist_ok=True)
    be = SDBackend()
    for i, (name, prompt) in enumerate(PROMPTS.items()):
        zT = SDBackend.sample_noise(777 + i)
        be.generate(zT, prompt, seed=777 + i).save(out / f"{name}.png")
        print("saved", name, flush=True)
    print("STYLES DONE")
