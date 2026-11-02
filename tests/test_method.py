"""CPU tests for models/stylering.py with a stub backend (no diffusion model)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.allocation import SurvivalMap  # noqa: E402
from core.fourier_rings import enumerate_cells  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402
from utils.seeds import derive_payload  # noqa: E402

import tempfile

SCRATCH = Path(tempfile.mkdtemp(prefix="stylering_test_"))


class StubBackend:
    latent = 64

    @staticmethod
    def sample_noise(seed, latent=64, n_ch=4):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, n_ch, latent, latent, generator=g)


def _cfg(strategy="lowcut", variant="soft", smap_path=""):
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["stylering"]["alloc_strategy"] = strategy
    cfg["stylering"]["decode_variant"] = variant
    cfg["paths"]["survival_map"] = smap_path
    return cfg


def _synthetic_smap_file() -> str:
    cells = enumerate_cells()
    r = np.array([c.r for c in cells], float)
    S = np.clip(1.0 - r / 40.0, 0.5, 1.0)
    smap = SurvivalMap(cells, S, S * 0.9, np.full(len(cells), 0.15), {"img2img_anime": S}, {})
    p = SCRATCH / "smap_method_test.npz"
    smap.save(str(p))
    return str(p)


def _roundtrip(cfg, noise_sigma=0.0, seed=5):
    m = build_method("stylering", cfg, StubBackend())
    payload = derive_payload("p0001", seed, m.n_bits)
    zT = m.embed(payload, seed)
    zhat = zT + noise_sigma * torch.randn_like(zT)
    res = m.extract(zhat)
    return m, payload, res


def test_clean_roundtrip_all_variants():
    smap = _synthetic_smap_file()
    for strategy in ["lowcut", "uniform", "highring", "measured", "spatial_sign"]:
        for variant in ["soft", "hard", "argmin", "no_ecc"]:
            m, payload, res = _roundtrip(_cfg(strategy, variant, smap))
            assert res.bits is not None and (res.bits == payload).all(), (strategy, variant)
            if variant in ("soft", "hard"):
                assert res.accepted and res.n_corrected == 0 and res.score > 0


def test_noisy_roundtrip_soft_beats_argmin():
    smap = _synthetic_smap_file()
    accs = {}
    for variant in ["soft", "argmin"]:
        ok = 0
        for seed in range(6):
            m, payload, res = _roundtrip(_cfg("measured", variant, smap), noise_sigma=1.6, seed=seed)
            ok += int(res.bits is not None and (res.bits == payload).all())
        accs[variant] = ok
    assert accs["soft"] >= accs["argmin"], accs


def test_unwatermarked_rejected():
    smap = _synthetic_smap_file()
    m = build_method("stylering", _cfg("measured", "soft", smap), StubBackend())
    rejects = 0
    for seed in range(8):
        res = m.extract(StubBackend.sample_noise(1000 + seed))
        rejects += int(not res.accepted)
    assert rejects >= 6, f"{rejects}/8 unwatermarked latents rejected"


def test_full_budget_chip_variant():
    """# OURS full-budget: configs/full_chip.yaml (44 rings, real+imag, per-pair chips, repeat, count calib)."""
    cfg = load_config(ROOT / "configs/full_chip.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    assert m.pos_of is not None and len(m.alloc) == 4 * 44 * 2 and m.chips is not None
    assert m.sizes.sum() == 2 * 4 * (64 * 64 - 1 - 127)
    for variant in ["soft", "hard", "argmin", "no_ecc"]:
        c = copy.deepcopy(cfg)
        c["stylering"]["decode_variant"] = variant
        mm, payload, res = _roundtrip(c)
        assert res.bits is not None and (res.bits == payload).all(), variant
        if variant in ("soft", "hard"):
            assert res.accepted and res.n_corrected == 0 and res.score > 0
    # heavy noise: full budget decodes where the 63-cell chip carrier (default cells, K=0) fails
    ok_full = ok_63 = 0
    c63 = _cfg("lowcut", "soft", "")
    c63["stylering"]["sectors"] = 0
    for seed in range(6):
        _, p1, r1 = _roundtrip(cfg, noise_sigma=2.0, seed=seed)
        _, p2, r2 = _roundtrip(c63, noise_sigma=2.0, seed=seed)
        ok_full += int(r1.bits is not None and (r1.bits == p1).all())
        ok_63 += int(r2.bits is not None and (r2.bits == p2).all())
    assert ok_full >= ok_63 and ok_full >= 4, (ok_full, ok_63)
    # unwatermarked latents rejected
    rejects = sum(int(not m.extract(StubBackend.sample_noise(2000 + s)).accepted) for s in range(8))
    assert rejects >= 6, rejects


def test_readout_controls():
    """spatial_read soft (magnitude-weighted) and cell_read hard (majority) round-trip on clean latents."""
    for cfg_path, key, val in [("configs/full_spatial_soft.yaml", "spatial_read", "soft"),
                               ("configs/full_chip_hard.yaml", "cell_read", "hard")]:
        cfg = copy.deepcopy(load_config(ROOT / cfg_path))
        cfg["paths"]["survival_map"] = ""
        assert cfg["stylering"][key] == val
        m, payload, res = _roundtrip(cfg)
        assert res.accepted and (res.bits == payload).all() and res.n_corrected == 0, cfg_path
        rejects = sum(int(not m.extract(StubBackend.sample_noise(3000 + s)).accepted) for s in range(6))
        assert rejects >= 4, (cfg_path, rejects)


def test_rotation_search_recovers_rotated_latent():
    """rot_search: a latent rotated by 37 deg is re-synchronised and decoded; without the search it is unreadable."""
    from core.fourier_rings import rotate_latents
    for cfg_path in ("configs/full_rep.yaml", "configs/default.yaml"):
        cfg = copy.deepcopy(load_config(ROOT / cfg_path))
        if cfg_path.endswith("default.yaml"):
            cfg["stylering"]["alloc_strategy"] = "lowcut"        # no survival map in the test env (K=8 real-part carrier)
        cfg["paths"]["survival_map"] = ""
        cfg["stylering"]["rot_search"], cfg["stylering"]["rot_step"] = 1, 1.0
        m = build_method("stylering", cfg, StubBackend())
        payload = derive_payload("p0001", 5, m.n_bits)
        zr = rotate_latents(m.embed(payload, 5), [37.0])
        res = m.extract(zr)
        assert res.meta is not None and res.bits is not None and (res.bits == payload).all(), cfg_path
        th = res.meta["theta"]
        ok = min(abs(th - 37.0), abs(th - 323.0), abs(th - 217.0), abs(th - 143.0)) <= 1.0   # 180-deg ambiguity for real-only carriers
        assert ok, (cfg_path, th)
        assert res.meta["sync_stat"] > 3 * res.meta["sync_stat0"], res.meta
        cfg2 = copy.deepcopy(cfg)
        cfg2["stylering"]["rot_search"] = 0
        r2 = build_method("stylering", cfg2, StubBackend()).extract(zr)
        acc2 = None if r2.bits is None else float((r2.bits == payload).mean())
        assert acc2 is None or acc2 < 0.9, (cfg_path, acc2)                    # no search: misaligned chips
        assert abs(th - 323.0) <= 1.0 or abs(th - 143.0) <= 1.0, (cfg_path, th)   # search rotates back by -37
        rejects = sum(int(not m.extract(StubBackend.sample_noise(4000 + s)).accepted) for s in range(4)) if m.variant != "no_ecc" else 4
        assert rejects >= 3
        # peak gating: on unwatermarked latents the search is rejected (theta stays 0, no random angle picked)
        gated = [m.extract(StubBackend.sample_noise(5000 + s)).meta for s in range(4)]
        assert all(g["gated"] and g["theta"] == 0.0 for g in gated), gated
        # ... and a weakly-watermarked, un-rotated latent keeps its direct read (no mis-rotation)
        zn = m.embed(payload, 5) + 2.5 * torch.randn_like(zr)
        rn = m.extract(zn)
        assert rn.meta["theta"] == 0.0, rn.meta


def test_gs_sync_control_recovers_rotation_and_crop():
    """Gaussian Shading + the same synchronisation searches (control): rotated / cropped clean latents decode."""
    from core.fourier_rings import rotate_latents, shift_latent, zoom_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/default.yaml"))
    cfg["paths"]["survival_map"] = ""
    cfg.setdefault("methods", {})["gaussian_shading"] = {"rot_search": 1, "crop_search": 1}
    m = build_method("gaussian_shading", cfg, StubBackend())
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    r = m.extract(zT)
    assert (r.bits == payload).all() and r.meta["theta"] == 0.0 and r.meta["scale"] == 1.0, r.meta
    r = m.extract(rotate_latents(zT, [37.0]))
    assert (r.bits == payload).mean() >= 0.9 and abs(r.meta["theta"] - 323.0) <= 1.0, r.meta
    r = m.extract(shift_latent(zoom_latents(zT, [1 / 0.8]), -2.25, 3.5))
    assert (r.bits == payload).mean() >= 0.9 and abs(r.meta["scale"] - 0.8) <= 0.011, r.meta
    r0 = m.extract(StubBackend.sample_noise(7000))
    assert r0.meta["gated"] and r0.meta["crop_gated"] and r0.meta["theta"] == 0.0 and r0.meta["scale"] == 1.0, r0.meta
    m_plain = build_method("gaussian_shading", copy.deepcopy(load_config(ROOT / "configs/default.yaml")), StubBackend())
    assert (m_plain.extract(rotate_latents(zT, [37.0])).bits == payload).mean() < 0.9   # paper decoder: no sync


def test_gs_official_matches_our_reimplementation():
    """Official Gaussian Shading code (vendor) embeds/decodes; our class with the same key decodes identically."""
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)")
        return
    cfg = copy.deepcopy(load_config(ROOT / "configs/final_gs_official.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("gs_official", cfg, StubBackend())
    for seed in (5, 6):
        payload = derive_payload("p0001", seed, m.n_bits)
        zT = m.embed(payload, seed)
        assert zT.shape == (1, 4, 64, 64)
        assert abs(float(zT.std()) - 1.0) < 0.05 and abs(float(zT.mean())) < 0.05      # half-normal sampling
        r = m.extract(zT)
        assert (r.bits == payload).all() and r.meta["ours_agree"] == 1.0 and r.score > 0.99, (r.bits, r.meta)
        rn = m.extract(zT + 1.5 * torch.randn_like(zT))                                  # noisy: still identical decoders
        assert rn.meta["ours_agree"] == 1.0, rn.meta
        assert abs(rn.score - rn.meta["ours_score"]) < 1e-6, rn.meta
    r0 = m.extract(StubBackend.sample_noise(8000))
    assert r0.meta["ours_agree"] == 1.0 and r0.score < 0.1, r0.meta


def test_full_rep_256bit():
    """256-bit (official GS capacity) repetition variant with both searches: clean round trip, gating on noise."""
    from core.fourier_rings import rotate_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep256_sync.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    assert m.n_bits == 256 and m.n_code == 256 and m.pos_of is not None and len(set(m.pos_of.tolist())) == 256
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    r = m.extract(zT)
    assert (r.bits == payload).all() and r.meta["theta"] == 0.0 and r.meta["scale"] == 1.0, r.meta
    rr = m.extract(rotate_latents(zT, [23.0]))
    assert (rr.bits == payload).mean() >= 0.98 and abs(rr.meta["theta"] - 337.0) <= 1.0, rr.meta
    r0 = m.extract(StubBackend.sample_noise(9000))
    assert r0.meta["gated"] and r0.meta["crop_gated"], r0.meta


def test_rotation_with_canvas_expansion_joint_zoom():
    """rot_scales: a rotation that also shrank the content (expanded canvas) is recovered jointly."""
    from core.fourier_rings import rotate_latents, zoom_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep_blind.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    zr = rotate_latents(zoom_latents(zT, [0.8]), [50.0])            # content shrunk to 0.8 (expanded canvas) then rotated 50 deg
    r = m.extract(zr)
    assert (r.bits == payload).all() and abs(r.meta["theta"] - 310.0) <= 1.0 and abs(r.meta["scale"] - 1.25) < 0.06, r.meta
    r0 = m.extract(StubBackend.sample_noise(9100))
    assert r0.meta["geo_gated"] and r0.meta["theta"] == 0.0 and r0.meta["scale"] == 1.0, r0.meta


def test_joint_geometric_search_crop_plus_rotation():
    """geo_search: zoom 0.75 + rotation 33 deg + sub-pixel shift recovered in one pass; noise gated."""
    from core.fourier_rings import rotate_latents, shift_latent, zoom_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep_blind.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    assert m.geo_search
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    zc = shift_latent(rotate_latents(zoom_latents(zT, [1 / 0.75]), [33.0]), 2.5, -1.75)   # magnified crop + rotation
    r = m.extract(zc)
    # unpaired allocation (full_rep_blind.yaml): the 180-degree alias is exact, so theta may come back as 327 or 147;
    # the paired allocation (full_rep_blind2.yaml, test below) resolves it.
    assert abs(r.meta["scale"] - 0.75) <= 0.02 and min(abs(r.meta["theta"] - 327.0), abs(r.meta["theta"] - 147.0)) <= 1.0, r.meta
    if abs(r.meta["theta"] - 327.0) <= 1.0:
        assert (r.bits == payload).mean() >= 0.95, (r.meta, (r.bits == payload).mean())
    r0 = m.extract(StubBackend.sample_noise(9200))
    assert r0.meta["geo_gated"] and r0.meta["theta"] == 0.0 and r0.meta["scale"] == 1.0, r0.meta


def test_paired_allocation_removes_180_alias():
    """pair_parts: every position mixes real and imaginary cells; joint search on crop+rotation finds the true angle."""
    from core.fourier_rings import rotate_latents, shift_latent, zoom_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep_blind2.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    for b in range(m.n_code):
        parts = {c.part for c, p in zip(m.alloc, m.pos_of) if p == b}
        assert parts == {0, 1}, (b, parts)
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    for ang in (33.0, 200.0):
        zc = shift_latent(rotate_latents(zoom_latents(zT, [1 / 0.75]), [ang]), 2.5, -1.75)
        r = m.extract(zc)
        assert (r.bits == payload).all(), (ang, r.meta, (r.bits == payload).mean())
        assert abs(r.meta["scale"] - 0.75) <= 0.02 and abs((r.meta["theta"] - (360 - ang)) % 360) <= 1.0, (ang, r.meta)
    z180 = rotate_latents(zT, [180.0])
    r = m.extract(z180)
    assert (r.bits == payload).all() and abs(r.meta["theta"] - 180.0) <= 1.0, r.meta
    r0 = m.extract(StubBackend.sample_noise(9300))
    assert r0.meta["geo_gated"], r0.meta


def test_paired_allocation_256bit():
    """v2 allocation at 256 positions: no empty position, 96 mixed positions, true angle beats the 180-degree alias."""
    from core.fourier_rings import rotate_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep256_sync2.yaml"))
    cfg["paths"]["survival_map"] = ""
    cfg["stylering"]["geo_search"], cfg["stylering"]["rot_search"], cfg["stylering"]["crop_search"] = 1, 0, 0
    m = build_method("stylering", cfg, StubBackend())
    counts = np.bincount(m.pos_of, minlength=m.n_code)
    assert counts.min() >= 1 and m.n_code == 256, counts.min()
    mixed = sum(1 for b in range(m.n_code) if {c.part for c, p in zip(m.alloc, m.pos_of) if p == b} == {0, 1})
    assert mixed == 96, mixed
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    for ang in (0.0, 180.0, 41.0):
        r = m.extract(rotate_latents(zT, [ang]))
        assert (r.bits == payload).mean() >= 0.98 and abs((r.meta["theta"] - (360 - ang) % 360) % 360) <= 1.0, (ang, r.meta, (r.bits == payload).mean())


def test_flip_candidate_recovers_mirrored_latent():
    cfg = copy.deepcopy(load_config(ROOT / "configs/full_rep_sync2_flip.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    r = m.extract(zT)
    assert (r.bits == payload).all() and r.meta["flipped"] is False, r.meta
    for dims in ([3], [2]):                                   # horizontal mirror, vertical flip
        r = m.extract(torch.flip(zT, dims=dims))
        assert (r.bits == payload).all() and r.meta["flipped"] is True, (dims, r.meta)
    r0 = m.extract(StubBackend.sample_noise(9400))
    assert r0.score < 0.1, r0.meta


def test_final256_detector_and_cascade():
    """final256: one detector (rotation + deep crop + flip). final256_combo adds the joint-search fallback,
    which must stay off for a single geometric edit and recover a crop + rotation that the sequential searches miss."""
    from core.fourier_rings import rotate_latents, shift_latent, zoom_latents
    cfg = copy.deepcopy(load_config(ROOT / "configs/final256.yaml"))
    cfg["paths"]["survival_map"] = ""
    m = build_method("stylering", cfg, StubBackend())
    assert m.n_bits == 256 and m.flip_search and not m.geo_fallback and m.sync.max_shift == 32.0
    payload = derive_payload("p0001", 5, m.n_bits)
    zT = m.embed(payload, 5)
    r = m.extract(zT)
    assert (r.bits == payload).all() and r.meta["theta"] == 0.0 and r.meta["scale"] == 1.0, r.meta
    r = m.extract(shift_latent(zoom_latents(zT, [1 / 0.5 ** 0.5]), 9.25, -14.5))   # deep crop: half the area, far off-centre
    assert (r.bits == payload).mean() >= 0.95 and abs(r.meta["scale"] - 0.71) <= 0.02, r.meta
    cfg_c = copy.deepcopy(load_config(ROOT / "configs/final256_combo.yaml"))
    cfg_c["paths"]["survival_map"] = ""
    mc = build_method("stylering", cfg_c, StubBackend())
    r = mc.extract(rotate_latents(zT, [30.0]))
    assert (r.bits == payload).mean() >= 0.98 and "geo_fallback" not in r.meta, r.meta
    zc = rotate_latents(shift_latent(zoom_latents(zT, [1 / 0.75]), 2.5, -1.75), [30.0])   # crop 0.75, then rotate 30
    r = mc.extract(zc)
    assert (r.bits == payload).mean() >= 0.9, (r.meta, (r.bits == payload).mean())
    r0 = mc.extract(StubBackend.sample_noise(9500))
    # 256-bit null score ~ sqrt(2/pi)/sqrt(62 votes) ~ 0.1-0.14 (16 bit: ~0.03); every search must stay gated
    assert r0.score < 0.2 and r0.meta["gated"] and r0.meta["crop_gated"] and r0.meta["geo_gated"], r0.meta


def test_combo_attack_chain():
    from PIL import Image
    from attacks.combo import parse_chain
    from attacks.registry import get_attack
    assert parse_chain("style_img2img/anime@0.3+crop_scale@0.75") == [("style_img2img", "anime", 0.3), ("crop_scale", None, 0.75)]
    im = Image.fromarray((np.random.default_rng(0).random((512, 512, 3)) * 255).astype(np.uint8))
    out = get_attack("combo", {})(im, 1, 97, "", style="crop_scale@0.75+rotate@30")
    ref = get_attack("rotate", {})(get_attack("crop_scale", {})(im, 0.75, 97), 30, 97)
    assert out.size == (512, 512) and np.array_equal(np.asarray(out), np.asarray(ref))


def test_measured_requires_map():
    try:
        build_method("stylering", _cfg("measured", "soft", ""), StubBackend())
    except FileNotFoundError:
        return
    raise AssertionError("measured strategy without a survival map must fail loudly")


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("PASS", k)
    print("ALL METHOD TESTS PASSED")
