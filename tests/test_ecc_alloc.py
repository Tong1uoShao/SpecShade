"""CPU unit tests for core/ecc.py and core/allocation.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.allocation import SurvivalMap, allocate  # noqa: E402
from core.ecc import (BCHCode, bits_to_hex, chase2_decode, chase2_decode_reference,  # noqa: E402
                      decrypt, encrypt, hex_to_bits, reject_rule)
from core.fourier_rings import enumerate_cells  # noqa: E402

KEY = b"unit-test-key"


def test_encrypt_roundtrip_and_uniformity():
    rng = np.random.default_rng(0)
    p = rng.integers(0, 2, 32)
    c = encrypt(p, KEY)
    assert (decrypt(c, KEY) == p).all()
    assert not (c == p).all()
    zeros = encrypt(np.zeros(4000, int), KEY)         # keystream itself
    assert abs(zeros.mean() - 0.5) < 0.03


def test_bch_hard_roundtrip_and_correction():
    for n, k in [(63, 16), (63, 30), (31, 6)]:
        code = BCHCode(n, k)
        rng = np.random.default_rng(n + k)
        m = rng.integers(0, 2, k)
        cw = code.encode(m)
        assert cw.size == n and (cw[:k] == m).all(), "systematic"
        noisy = cw.copy()
        flip = rng.choice(n, code.t, replace=False)
        noisy[flip] ^= 1
        dec, n_err, ok = code.decode_hard(noisy)
        assert ok and n_err == code.t and (dec == m).all()


def test_chase2_beats_hard_decision():
    code = BCHCode(63, 16)
    rng = np.random.default_rng(7)
    m = rng.integers(0, 2, 16)
    cw = code.encode(m)
    bip = 2 * cw - 1
    # Gaussian channel with t+2 low-reliability sign errors: hard decoding fails, Chase-II recovers
    llr = 3.0 * bip + rng.normal(0, 0.3, 63)
    bad = rng.choice(63, code.t + 2, replace=False)
    llr[bad] = -0.4 * bip[bad]          # wrong sign but low |llr|
    _, _, ok_hard = code.decode_hard((llr > 0).astype(int))
    assert not ok_hard
    msg, accepted, d, n_err = chase2_decode(llr, code, n_flip=4)
    assert accepted and (msg == m).all()
    assert reject_rule(accepted, d, thr=d + 1) and not reject_rule(accepted, d, thr=d - 1)


def test_chase2_rejects_noise():
    code = BCHCode(63, 16)
    rng = np.random.default_rng(11)
    rejects = 0
    for _ in range(20):
        llr = rng.normal(0, 1, 63)
        _, accepted, _, _ = chase2_decode(llr, code, n_flip=4)
        rejects += (not accepted)
    assert rejects >= 15, f"random LLR should mostly fail the syndrome check ({rejects}/20 rejected)"


def test_codebook_matches_galois_encoder():
    code = BCHCode(63, 16)
    C = code.codebook
    assert C.shape == (1 << 16, 63) and C.dtype == np.int8
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.integers(0, 2, 16)
        idx = int("".join(map(str, m)), 2)
        assert (C[idx] == code.encode(m)).all()
        assert (C[idx, :16] == m).all(), "systematic: message = first k positions"


def test_chase2_table_equals_reference():
    """Table-based Chase-II must reproduce the literal galois loop (msg, accepted, soft dist,
    n_corrected) on watermarked, borderline and null LLRs, for n_flip in {0, 4, 8}."""
    code = BCHCode(63, 16)
    rng = np.random.default_rng(21)
    cases = []
    for i in range(30):
        m = rng.integers(0, 2, 16)
        bip = 2 * code.encode(m) - 1
        if i % 5 == 0:
            cases.append(rng.normal(0, 1, 63))                                # null
        else:
            cases.append(bip * rng.uniform(0.3, 3.0) + rng.normal(0, 1.5, 63))  # noisy watermark
    plan = [(4, cases)] + [(8, cases[:3])] + [(0, cases[:6])]
    n_acc = 0
    for n_flip, llrs in plan:
        for llr in llrs:
            a = chase2_decode_reference(llr, code, n_flip=n_flip)
            b = chase2_decode(llr, code, n_flip=n_flip)
            assert a[1] == b[1], (n_flip, a, b)
            if a[1]:
                n_acc += 1
                assert (a[0] == b[0]).all() and abs(a[2] - b[2]) < 1e-9 and a[3] == b[3], (n_flip, a, b)
            else:
                assert b[0] is None and b[2] == float("inf") and b[3] == -1
    assert n_acc >= 10, "test should cover accepted cases too"


def test_hex_roundtrip():
    b = np.random.default_rng(1).integers(0, 2, 19)
    assert (hex_to_bits(bits_to_hex(b)) == b).all()


def _synthetic_map():
    cells = enumerate_cells(quadrants=False)
    r = np.array([c.r for c in cells], float)
    S = np.clip(1.0 - r / 40.0, 0.5, 1.0) + 0.001 * np.array([c.c for c in cells])
    mu, sigma = S * 0.8, np.full(len(cells), 0.2)
    per = {"a": S, "b": np.clip(S - 0.1, 0.5, 1.0)}
    return SurvivalMap(cells, S, mu, sigma, per, {"note": "synthetic"})


def test_allocation_strategies(tmp=None):
    import tempfile
    tmp = Path(tmp or tempfile.mkdtemp(prefix="stylering_test_"))
    smap = _synthetic_map()
    n = 63
    meas = allocate(smap, n, "measured")
    assert len(meas) == n and len(set(meas)) == n
    assert max(c.r for c in meas) <= 16, "measured must pick low-r cells on a decreasing map"
    uni = allocate(smap, n, "uniform")
    assert min(c.r for c in uni) <= 2 and max(c.r for c in uni) >= 30
    low = allocate(smap, n, "lowcut")
    assert max(c.r for c in low) <= 16
    high = allocate(smap, n, "highring")
    assert min(c.r for c in high) >= 16
    sub = allocate(smap, n, "measured", attack_subset=["b"])
    assert len(sub) == n
    p = tmp / "smap_test.npz"
    smap.save(str(p))
    s2 = SurvivalMap.load(str(p))
    assert s2.cells == smap.cells and np.allclose(s2.S, smap.S) and set(s2.per_attack) == {"a", "b"}
    calib = smap.to_calib()
    assert len(calib.mu) == len(smap.cells)


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("PASS", k)
    print("ALL ECC/ALLOC TESTS PASSED")
