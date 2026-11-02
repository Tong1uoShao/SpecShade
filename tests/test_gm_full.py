"""GPU test for models/baseline/gaussmarker_full.py (needs CUDA + pycryptodome; run on the GPU node)."""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402

from test_method import StubBackend  # noqa: E402


def _cfg(gnr_path: str = ""):
    from models.baseline.gaussmarker_full import _load_vendor
    wm = _load_vendor("gm_watermark_t", "watermark.py")
    gs = wm.Gaussian_Shading_chacha(1, 8, 8, 1e-6, 1000000)
    gs.create_watermark_and_return_w_m()
    tmp = Path(tempfile.mkdtemp(prefix="gm_full_"))
    torch.save({"w": gs.watermark, "m": gs.m, "key": gs.key, "nonce": gs.nonce}, tmp / "w1.pth")
    cfg = copy.deepcopy(load_config(ROOT / "configs/final_gm256.yaml"))
    cfg["methods"] = {"gaussmarker_full": {"w_info_path": str(tmp / "w1.pth"), "gnr_path": gnr_path}}
    return cfg


def test_roundtrip_and_null():
    m = build_method("gaussmarker_full", _cfg(), StubBackend())
    assert m.n_bits == 256 and m.gnr is None
    z = m.embed(None, 7)
    assert z.shape == (1, 4, 64, 64) and abs(float(z.std()) - 1.0) < 0.05
    r = m.extract(z)
    assert (r.bits == m.fixed_payload).all() and r.meta["acc_known"] == 1.0, r.meta
    r0 = m.extract(StubBackend.sample_noise(4242))
    assert 0.35 < r0.meta["acc_known"] < 0.65 and r0.score < r.score, (r0.meta, r0.score, r.score)


def test_with_gnr_checkpoint():
    ck = ROOT / "results/gnr/GNR_256bits/model_final.pth"
    if not ck.exists():
        ck = ROOT / "results/gnr/GNR_256bits/checkpoint.pth"
    if not ck.exists():
        print("SKIP (no GNR checkpoint yet)")
        return
    m = build_method("gaussmarker_full", _cfg(str(ck.relative_to(ROOT))), StubBackend())
    assert m.gnr is not None
    r = m.extract(m.embed(None, 3))
    # a GNR trained for ANOTHER message (the temp one here differs) must at least not destroy a clean read-out much
    assert r.meta["acc_known_nognr"] == 1.0, r.meta


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("PASS", k)
    print("ALL GM_FULL TESTS PASSED")
