"""CPU tests: classical attacks, matrix parsing, registry resume logic (no diffusion models)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.classical import CLASSICAL, Identity  # noqa: E402
from attacks.registry import build_matrix, get_attack  # noqa: E402
from utils.io import load_config  # noqa: E402


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))


def test_classical_shapes_and_determinism():
    im = _img()
    strengths = {"jpeg": 25, "rotate": 75, "crop_scale": 0.75, "gauss_blur": 8, "gauss_noise": 0.1,
                 "brightness": 6, "rand_drop": 0.8, "median": 7, "resize": 0.25}
    for name, cls in CLASSICAL.items():
        out = cls()(im, strengths[name], 97, "")
        assert out.size == (512, 512) and out.mode == "RGB", name
        out2 = cls()(im, strengths[name], 97, "")
        assert np.array_equal(np.asarray(out), np.asarray(out2)), f"{name} not deterministic"
        assert not np.array_equal(np.asarray(out), np.asarray(im)), f"{name} is a no-op"
    assert np.array_equal(np.asarray(Identity()(im)), np.asarray(im))


def test_matrix_parsing():
    acfg = load_config(ROOT / "configs/attacks.yaml")
    specs = build_matrix(acfg, ["style"])
    assert len(specs) == 9 and all(len(s.strengths) == 3 for s in specs)
    assert {s.engine for s in specs} == {"style_img2img", "style_ip2p", "style_adain"}
    sub = build_matrix(acfg, ["style", "classical"], names=["img2img_anime", "jpeg"])
    assert [s.name for s in sub] == ["img2img_anime", "jpeg"]
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["_root"] = str(ROOT)
    assert get_attack("jpeg", cfg).engine == "jpeg"


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("PASS", k)
    print("ALL ATTACK CPU TESTS PASSED")
