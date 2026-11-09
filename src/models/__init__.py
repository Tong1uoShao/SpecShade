"""Method registry: name -> WatermarkMethod class (baselines register lazily)."""
from __future__ import annotations

from pathlib import Path

from models.base import NoWatermark, WatermarkMethod
from utils.io import load_config

_METHODS_YAML = Path(__file__).resolve().parents[2] / "configs" / "methods.yaml"
IMAGE_LEVEL = {"vine", "stable_signature"}     # methods that need no DDIM inversion at detection


def _with_methods_cfg(cfg: dict) -> dict:
    # OURS: baseline defaults live in configs/methods.yaml; explicit cfg["methods"] entries win.
    methods = {k: dict(v or {}) for k, v in (load_config(_METHODS_YAML) or {}).items()} if _METHODS_YAML.exists() else {}
    for k, v in (cfg.get("methods") or {}).items():
        methods.setdefault(k, {}).update(v or {})
    return {**cfg, "methods": methods}


def build_method(name: str, cfg: dict, backend) -> WatermarkMethod:
    cfg = _with_methods_cfg(cfg)
    if name == "none":
        return NoWatermark(cfg, backend)
    if name == "stylering":
        from models.stylering import StyleRing
        return StyleRing(cfg, backend)
    if name == "gaussian_shading":
        from models.baseline.gaussian_shading import GaussianShadingWM
        return GaussianShadingWM(cfg, backend)
    if name == "gs_official":
        from models.baseline.gs_official import GaussianShadingOfficialWM
        return GaussianShadingOfficialWM(cfg, backend)
    if name == "gaussmarker":
        from models.baseline.gaussmarker import GaussMarkerWM
        return GaussMarkerWM(cfg, backend)
    if name == "gaussmarker_full":
        from models.baseline.gaussmarker_full import GaussMarkerFullWM
        return GaussMarkerFullWM(cfg, backend)
    if name == "ringid":
        from models.baseline.ringid import RingIDWM
        return RingIDWM(cfg, backend)
    if name == "maxsive":
        from models.baseline.maxsive import MaXsiveWM
        return MaXsiveWM(cfg, backend)
    if name in ("sfw_hstr", "sfw_hsqr"):
        from models.baseline.sfw import SFWWM
        return SFWWM(cfg, backend, variant=name.split("_")[1])
    if name == "t2smark":
        from models.baseline.t2smark import T2SMarkWM
        return T2SMarkWM(cfg, backend)
    if name == "vine":
        from models.baseline.vine import VineWM
        return VineWM(cfg, backend)
    if name == "stable_signature":
        from models.baseline.stable_signature import StableSignatureWM
        return StableSignatureWM(cfg, backend)
    raise KeyError(f"unknown method: {name}")
