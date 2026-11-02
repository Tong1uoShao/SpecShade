"""CPU unit tests for src/core (run: .venv/bin/python -m pytest -q code/tests or python tests/test_core.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.allocation import allocate_repeat  # noqa: E402
from core.fourier_rings import (Cell, CellReader, SectorChips, ShiftScaleSync, block_fft, enumerate_cells,  # noqa: E402
                                max_radius, read_cell, ring_chips, ring_groups, ring_mask, ring_parity,
                                rotate_latents, shift_latent, zoom_latents)
from core.modulation import (CellCalib, combine_positions, count_calib, embed_bits, hard_bits,  # noqa: E402
                             read_cells, to_llr)


def _rand_latent(seed: int, size: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, size, size, generator=g)


def radial_coherence(d: np.ndarray) -> float:
    """Fraction of a 2-D pattern's variance explained by its radial profile about the
    (periodic) origin == how much it looks like concentric rings around the corners."""
    n = d.shape[0]
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rho = np.round(np.sqrt(np.minimum(yy, n - yy) ** 2 + np.minimum(xx, n - xx) ** 2)).astype(int)
    prof = np.zeros(rho.max() + 1)
    for r in range(rho.max() + 1):
        prof[r] = d[rho == r].mean()
    return float(np.var(prof[rho]) / (np.var(d) + 1e-12))


# ---------------- sector chips (# OURS artifact fix) ----------------
def test_sector_groups_antipodal_and_counts():
    h = w = 64
    for r in range(1, max_radius(64) + 1):
        ii, jj = np.nonzero(ring_mask(h, w, r))
        pos = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(ii, jj))}
        for K in (0, 2, 4, 8):
            g = ring_groups(h, w, r, K)
            for k, (i, j) in enumerate(zip(ii, jj)):
                assert g[k] == g[pos[((h - i) % h, (w - j) % w)]], f"r={r} K={K}: antipodal pair split"
            if K == 0:
                assert g.max() + 1 == len(ii) // 2          # one chip per pair
            else:
                assert g.max() < K
        assert set(ring_chips(h, w, r, 4, 123).tolist()) <= {-1.0, 1.0}


def test_sector_roundtrip_and_key_mismatch():
    zT = _rand_latent(11)
    cells = enumerate_cells()
    bits = np.random.default_rng(5).integers(0, 2, len(cells))
    for K in (4, 0):
        chips = SectorChips(b"key-A", K)
        zw = embed_bits(zT, cells, bits, chips)
        y = read_cells(zw, cells, chips)
        assert np.allclose(np.abs(y), 1.0, atol=1e-4), f"K={K}: modulated rings must read +-1"
        assert (hard_bits(y) == bits).all()
        # wrong key or legacy (chip-less) reader: bit agreement at chance. (|y| itself is only
        # near 0 in chip mode; with K sectors it is a random +-1 average over K terms.)
        for reader in (SectorChips(b"key-B", K), None):
            y2 = read_cells(zw, cells, reader)
            if K == 0:
                assert np.abs(y2).mean() < 0.35, "chip mode leaks without the right chips"
            assert 0.35 < (hard_bits(y2) == bits).mean() < 0.65
        # spatial-domain marginal still N(0,1)
        x = zw.flatten().numpy()
        assert stats.kstest(x, "norm").pvalue > 0.01


def test_sectors_one_is_legacy():
    zT = _rand_latent(12)
    cells = enumerate_cells()[:20]
    bits = np.random.default_rng(6).integers(0, 2, len(cells))
    z_legacy = embed_bits(zT, cells, bits)
    z_k1 = embed_bits(zT, cells, bits, SectorChips(b"any", 1))
    assert torch.equal(z_legacy, z_k1)


def test_sector_chips_break_radial_coherence():
    zT = _rand_latent(13)
    cells = [c for c in enumerate_cells() if c.c == 0]
    bits = np.random.default_rng(7).integers(0, 2, len(cells))
    coh = {}
    for K in (1, 4, 0):
        zw = embed_bits(zT, cells, bits, SectorChips(b"k", K))
        coh[K] = radial_coherence((zw - zT)[0, 0].numpy())
    assert coh[1] > 3 * coh[0], f"chip mode should kill the radial pattern: {coh}"
    assert coh[1] > coh[4] > coh[0], f"coherence must fall with shorter coherence length: {coh}"


def test_ring_mask_symmetric_and_disjoint():
    h = w = 64
    total = np.zeros((h, w), bool)
    for r in range(1, max_radius(64) + 1):
        m = ring_mask(h, w, r)
        assert m.any()
        # central symmetry (i,j) <-> (h-i, w-j) for all masked pixels
        ii, jj = np.nonzero(m)
        assert m[(h - ii) % h, (w - jj) % w].all()
        assert not (total & m).any()
        total |= m


def test_enumerate_cells_counts():
    assert len(enumerate_cells()) == 4 * 31
    assert len(enumerate_cells(quadrants=True)) == 4 * 4 * 15


def test_roundtrip_full_image():
    zT = _rand_latent(1)
    cells = enumerate_cells()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, len(cells))
    zw = embed_bits(zT, cells, bits)
    assert zw.shape == zT.shape and zw.dtype == zT.dtype
    y = read_cells(zw, cells)
    assert np.allclose(np.abs(y), 1.0, atol=1e-4), "modulated rings must read +-1"
    assert (hard_bits(y) == bits).all(), "100% bit recovery on clean latent"


def test_roundtrip_quadrants():
    zT = _rand_latent(2)
    cells = enumerate_cells(quadrants=True)
    bits = np.random.default_rng(1).integers(0, 2, len(cells))
    zw = embed_bits(zT, cells, bits)
    y = read_cells(zw, cells)
    assert (hard_bits(y) == bits).all()


def test_unallocated_cells_untouched_in_frequency():
    zT = _rand_latent(3)
    all_cells = enumerate_cells()
    alloc = [c for c in all_cells if c.r <= 10]
    others = [c for c in all_cells if c.r > 10]
    zw = embed_bits(zT, alloc, np.ones(len(alloc), int))
    for c in range(4):
        F0, F1 = block_fft(zT, c, None), block_fft(zw, c, None)
        for cell in others:
            if cell.c != c:
                continue
            m = torch.from_numpy(ring_mask(64, 64, cell.r))
            assert torch.allclose(F0[m], F1[m], atol=1e-2), f"cell {cell} changed"
        # imaginary part globally untouched
        assert torch.allclose(F0.imag, F1.imag, atol=1e-2)


def test_marginal_distribution_preserved():
    # OURS: per-coefficient marginal stays N(0,1) in the spatial latent (KS test), because
    # half-Gaussian magnitude x uniformly random sign == Gaussian (Gaussian Shading argument).
    zT = _rand_latent(4)
    cells = enumerate_cells()
    bits = np.random.default_rng(2).integers(0, 2, len(cells))
    zw = embed_bits(zT, cells, bits)
    x = zw.flatten().numpy()
    p = stats.kstest(x, "norm").pvalue
    assert p > 0.01, f"KS p={p:.4f}: modulated latent not N(0,1)"
    assert abs(x.mean()) < 0.05 and abs(x.std() - 1) < 0.05


def test_null_readout_near_zero():
    zT = _rand_latent(5)
    cells = enumerate_cells()
    y = read_cells(zT, cells)
    assert np.abs(y).mean() < 0.35, "unmodulated rings should read near 0"
    # agreement with a random bit pattern is ~50%
    bits = np.random.default_rng(3).integers(0, 2, len(cells))
    acc = (hard_bits(y) == bits).mean()
    assert 0.35 < acc < 0.65


def test_llr_and_calib_io(tmp_path=None):
    import tempfile
    tmp_path = Path(tmp_path or tempfile.mkdtemp(prefix="stylering_test_"))
    cells = enumerate_cells()[:5]
    calib = CellCalib(mu={c.key(): 0.5 for c in cells}, sigma={c.key(): 0.25 for c in cells},
                      attack_set="unit")
    y = np.array([1.0, -1.0, 0.1, 0.0, 0.5])
    llr = to_llr(y, cells, calib)
    assert np.allclose(llr, 2 * 0.5 * y / 0.25 ** 2)
    assert np.allclose(to_llr(y, cells, None), 2 * y)
    p = tmp_path / "calib_test.npz"
    calib.save(str(p))
    c2 = CellCalib.load(str(p))
    assert c2.mu == calib.mu and c2.sigma == calib.sigma and c2.attack_set == "unit"


# ---------------- full-budget carrier (# OURS: all rings, real+imag, repetition) ----------------
def test_full_spectrum_rings_cover_everything_but_dc_and_nyquist():
    h = w = 64
    assert max_radius(64, full=True) == 44
    total = np.zeros((h, w), bool)
    for r in range(1, max_radius(64, full=True) + 1):
        m = ring_mask(h, w, r)
        assert m.any(), f"ring {r} empty"
        assert not m[0, :].any() and not m[:, 0].any(), "Nyquist lines must be excluded"
        ii, jj = np.nonzero(m)
        assert m[(h - ii) % h, (w - jj) % w].all(), f"ring {r} not centrally symmetric"
        assert not (total & m).any()
        total |= m
        par = ring_parity(h, w, r)
        pos = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(ii, jj))}
        for k, (i, j) in enumerate(zip(ii, jj)):
            assert par[k] == -par[pos[((h - i) % h, (w - j) % w)]], "antipodes need opposite parity"
    assert total.sum() == h * w - 1 - (h + w - 1)      # all but DC, row 0, col 0
    assert len(enumerate_cells(r_max=44, parts=(0, 1))) == 4 * 44 * 2
    assert enumerate_cells() == enumerate_cells(parts=(0,)) and len(enumerate_cells()) == 124


def test_imaginary_carrier_roundtrip_and_symmetry():
    zT = _rand_latent(21)
    cells = enumerate_cells(r_max=44, parts=(0, 1))
    bits = np.random.default_rng(8).integers(0, 2, len(cells))
    for K in (0, 8, 1):
        chips = SectorChips(b"key-full", K) if K != 1 else None
        zw = embed_bits(zT, cells, bits, chips)          # apply_to_cells asserts a real IFFT
        y = read_cells(zw, cells, chips)
        assert np.allclose(np.abs(y), 1.0, atol=1e-4), f"K={K}: modulated cells must read +-1"
        assert (hard_bits(y) == bits).all()
        yh = read_cells(zw, cells, chips, hard=True)          # majority-vote reader agrees on clean
        assert np.allclose(np.abs(yh), 1.0, atol=1e-4) and (hard_bits(yh) == bits).all()
        x = zw.flatten().numpy()
        p = stats.kstest(x, "norm").pvalue
        if K == 0:      # white chips: spatial marginal stays N(0,1) (distribution preserving)
            assert p > 0.01, f"K={K}: marginal not N(0,1) (p={p:.3g})"
        else:           # coherent sectors on the full budget align phases -> heavy-tailed latent
            assert p < 0.01, f"K={K}: full-budget coherence expected to break the marginal (p={p:.3g})"
    # imaginary-only writes leave the real spectrum untouched, and vice versa
    im_cells = [c for c in cells if c.part == 1 and c.c == 0]
    zi = embed_bits(zT, im_cells, np.ones(len(im_cells), int), SectorChips(b"k", 0))
    F0, F1 = block_fft(zT, 0, None), block_fft(zi, 0, None)
    assert torch.allclose(F0.real, F1.real, atol=1e-2)
    assert not torch.allclose(F0.imag, F1.imag, atol=1e-2)
    # wrong chips read at chance
    y2 = read_cells(embed_bits(zT, cells, bits, SectorChips(b"A", 0)), cells, SectorChips(b"B", 0))
    assert 0.4 < (hard_bits(y2) == bits).mean() < 0.6 and np.abs(y2).mean() < 0.3


def test_repeat_allocation_balanced_and_combined_readout():
    cells = enumerate_cells(r_max=44, parts=(0, 1))
    sizes = np.array([ring_mask(64, 64, c.r).sum() for c in cells])
    assert sizes.sum() == 2 * 4 * (64 * 64 - 1 - 127)   # every sign of the budget, once
    alloc, pos_of = allocate_repeat(cells, 63, sizes)
    load = np.bincount(pos_of, weights=sizes, minlength=63)
    assert load.min() > 0 and load.max() - load.min() <= sizes.max()
    zT = _rand_latent(22)
    code = np.random.default_rng(9).integers(0, 2, 63)
    chips = SectorChips(b"rep", 0)
    zw = embed_bits(zT, alloc, code, chips, pos_of)
    y = read_cells(zw, alloc, chips)
    llr = combine_positions(to_llr(y, alloc, count_calib(alloc, 64)), pos_of, 63)
    assert (hard_bits(llr) == code).all()
    assert np.allclose(llr, 2 * load * (2 * code - 1), rtol=1e-3)   # LLR = 2 * n_votes * y
    ym = combine_positions(y, pos_of, 63, sizes)
    assert np.allclose(np.abs(ym), 1.0, atol=1e-4)
    # under heavy noise the repeated read-out still recovers more than the 63-cell chip carrier
    noisy = zw + 1.5 * _rand_latent(23)
    acc_rep = (hard_bits(combine_positions(to_llr(read_cells(noisy, alloc, chips), alloc,
                                                  count_calib(alloc, 64)), pos_of, 63)) == code).mean()
    c63 = enumerate_cells()[:63]
    z63 = embed_bits(zT, c63, code, chips)
    acc_63 = (hard_bits(read_cells(z63 + 1.5 * _rand_latent(23), c63, chips)) == code).mean()
    assert acc_rep >= acc_63, (acc_rep, acc_63)


def test_cell_reader_matches_read_cells_and_rotation_helpers():
    zT = _rand_latent(31)
    cells = enumerate_cells(r_max=44, parts=(0, 1))
    bits = np.random.default_rng(12).integers(0, 2, len(cells))
    for K in (0, 8, 1):
        chips = SectorChips(b"rd", K) if K != 1 else None
        zw = embed_bits(zT, cells, bits, chips)
        rd = CellReader(cells, chips, 64)
        for hard in (False, True):
            a = rd.read(torch.cat([zw, zT]), hard=hard)
            assert np.allclose(a[0], read_cells(zw, cells, chips, hard=hard), atol=1e-4), (K, hard)
            assert np.allclose(a[1], read_cells(zT, cells, chips, hard=hard), atol=1e-4), (K, hard)
    assert torch.allclose(rotate_latents(zT, [0.0]), zT, atol=1e-5)          # angle 0 = identity
    z90 = rotate_latents(zT, [90.0])[0]
    assert (torch.allclose(z90, torch.rot90(zT[0], 1, (1, 2)), atol=1e-3)
            or torch.allclose(z90, torch.rot90(zT[0], -1, (1, 2)), atol=1e-3))
    assert rotate_latents(zT, np.arange(0, 360, 45)).shape == (8, 4, 64, 64)


def test_shift_scale_sync_recovers_crop_like_transform():
    zT = _rand_latent(41)
    cells = enumerate_cells(r_max=44, parts=(0, 1))
    chips = SectorChips(b"crop", 0)
    sizes = np.array([ring_mask(64, 64, c.r).sum() for c in cells])
    alloc, pos_of = allocate_repeat(cells, 16, sizes)
    code = np.random.default_rng(3).integers(0, 2, 16)
    zw = embed_bits(zT, alloc, code, chips, pos_of)
    sync = ShiftScaleSync(alloc, chips, pos_of, 16, 64, up=4, max_shift=12.0)
    # shift_latent is exact for integer shifts (= torch.roll) and stat_maps peaks at t = 0 on the clean latent
    assert torch.allclose(shift_latent(zw, 2.0, -3.0), torch.roll(zw, shifts=(-2, 3), dims=(2, 3)), atol=1e-4)
    s, ty, tx, st, st0, zs_ = sync.search(zw, [0.9, 1.0])
    assert (s, ty, tx) == (1.0, 0.0, 0.0) and st > 0.5 and abs(st - st0) < 1e-6 and zs_ > 8, (s, ty, tx, st, st0, zs_)
    # crop-like transform: magnify by 1/0.8 about the centre (zoom_latents with 1/s) and shift by (-2.25, 3.5) px
    zc = shift_latent(zoom_latents(zw, [1 / 0.8]), -2.25, 3.5)
    s, ty, tx, st, st0, zs_ = sync.search(zc, np.arange(0.55, 1.0001, 0.01))
    assert abs(s - 0.8) <= 0.011 and st > 3 * st0 and zs_ > 8, (s, ty, tx, st, st0, zs_)
    # an unwatermarked latent has no significant peak (gating keeps the direct read)
    *_, zn = sync.search(_rand_latent(42), np.arange(0.55, 1.0001, 0.05))
    assert zn < 8, zn
    zs = ShiftScaleSync.apply(zc, s, ty, tx)
    y = combine_positions(read_cells(zs, alloc, chips), pos_of, 16, sizes)
    assert (hard_bits(y) == code).all(), (s, ty, tx, y)
    y0 = combine_positions(read_cells(zc, alloc, chips), pos_of, 16, sizes)
    assert (hard_bits(y0) == code).mean() < 0.9                     # unsynchronised read fails


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL CORE TESTS PASSED")
