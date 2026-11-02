"""CPU tests for baseline wrappers with a stub backend: clean round trip and noise sanity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402
from utils.seeds import derive_payload  # noqa: E402


class StubBackend:
    latent = 64

    @staticmethod
    def sample_noise(seed, latent=64, n_ch=4):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, n_ch, latent, latent, generator=g)


def _cfg():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["methods"] = load_config(ROOT / "configs/methods.yaml")
    return cfg


def test_clean_roundtrip_all_baselines():
    for name in ["gaussian_shading", "ringid", "gaussmarker"]:
        m = build_method(name, _cfg(), StubBackend())
        assert m.n_bits == 16
        for seed in range(3):
            payload = derive_payload("p0", seed, 16)
            zT = m.embed(payload, seed)
            assert zT.shape == (1, 4, 64, 64)
            res = m.extract(zT)
            assert (res.bits == payload).all(), (name, seed)
            assert np.isfinite(res.score)


def test_gs_marginal_preserved_and_noise_tolerance():
    m = build_method("gaussian_shading", _cfg(), StubBackend())
    payload = derive_payload("p1", 0, 16)
    zT = m.embed(payload, 0)
    assert stats.kstest(zT.flatten().numpy(), "norm").pvalue > 0.01
    noisy = zT + 1.0 * torch.randn_like(zT)
    assert (m.extract(noisy).bits == payload).mean() >= 0.9


def test_ringid_noise_tolerance_and_null():
    m = build_method("ringid", _cfg(), StubBackend())
    payload = derive_payload("p2", 0, 16)
    zT = m.embed(payload, 0)
    noisy = zT + 1.0 * torch.randn_like(zT)
    assert (m.extract(noisy).bits == payload).mean() >= 0.9
    null = m.extract(StubBackend.sample_noise(999))
    assert null.score < m.extract(zT).score


def test_gaussmarker_fusion_orders_null_below_wm():
    m = build_method("gaussmarker", _cfg(), StubBackend())
    payload = derive_payload("p3", 0, 16)
    zT = m.embed(payload, 0)
    assert m.extract(zT).score > m.extract(StubBackend.sample_noise(1234)).score


def test_methods_yaml_loaded_by_default():
    """Regression: build_method must merge configs/methods.yaml so baselines never fall back to the stylering key."""
    cfg = load_config(ROOT / "configs/default.yaml")
    assert "methods" not in cfg
    assert build_method("gaussian_shading", cfg, StubBackend()).key == b"gs-key-v1"
    gm = build_method("gaussmarker", cfg, StubBackend())
    assert gm.key == b"gm-key-v1" and gm.spatial.key == b"gm-key-v1"
    assert build_method("ringid", cfg, StubBackend()).key == b"rid-key-v1"
    cfg_override = {**cfg, "methods": {"gaussian_shading": {"key": "override"}}}
    assert build_method("gaussian_shading", cfg_override, StubBackend()).key == b"override"


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("PASS", k)
    print("ALL BASELINE TESTS PASSED")
