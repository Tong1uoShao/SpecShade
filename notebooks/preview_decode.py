"""Offline preview of StyleRing decoding on probe read-outs (no GPU).

For every attacked probe image we know the true sign of every cell and the read-out y.
Because BCH is linear, decoding success depends only on the error pattern, so feeding the
sign-corrected read-outs (y * true_sign) of the allocated cells as the LLRs of the all-zero
codeword emulates the real decoder exactly. Allocation is chosen leave-one-image-out to avoid
in-sample optimism. Usage: python notebooks/preview_decode.py results/probe_smoke [--n_bits 16]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.allocation import SurvivalMap, allocate  # noqa: E402
from core.ecc import BCHCode, chase2_decode  # noqa: E402
from core.fourier_rings import Cell  # noqa: E402
from utils.io import read_jsonl  # noqa: E402

BCH_OF = {16: (63, 16), 32: (63, 30), 8: (63, 7)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir")
    ap.add_argument("--n_bits", type=int, default=16)
    ap.add_argument("--strategies", default="measured,lowcut,highring,uniform")
    ap.add_argument("--n_flip", type=int, default=4)
    args = ap.parse_args()
    pd_ = ROOT / args.probe_dir
    smap = SurvivalMap.load(str(pd_ / "survival_map.npz"))
    cells = smap.cells
    gen = read_jsonl(pd_ / "gen.jsonl")
    reads = read_jsonl(pd_ / "reads.jsonl")
    signs = {g["prompt_id"]: 2 * np.random.default_rng(g["seed"]).integers(0, 2, len(cells)) - 1 for g in gen}
    n, k = BCH_OF[args.n_bits]
    code = BCHCode(n, k)
    zero_cw = np.zeros(n, int)

    # per-image sign-corrected read-outs, grouped by attack cell name
    by_attack = defaultdict(list)   # attack -> list of (img, yc)
    for r in reads:
        yc = np.array(r["y"]) * signs[r["src_key"]]
        name = "none" if r["attack"] == "none" else f"{r['attack']}@{r['strength']}"
        by_attack[name].append((r["src_key"], yc))
    imgs = sorted({g["prompt_id"] for g in gen})

    def loo_map(exclude_img):
        # rebuild S/mu/sigma from all reads except one image (leave-one-out allocation)
        Y = np.array([yc for a in by_attack for (im, yc) in by_attack[a] if im != exclude_img])
        return SurvivalMap(cells, (Y > 0).mean(0), Y.mean(0), Y.std(0), {}, {})

    print(f"BCH({n},{k}) t={code.t}, Chase n_flip={args.n_flip}, cells={len(cells)}, images={len(imgs)}")
    header = f"{'attack':>24s} " + " ".join(f"{s:>22s}" for s in args.strategies.split(","))
    print(header)
    for a in sorted(by_attack):
        cols = []
        for strat in args.strategies.split(","):
            ok, bits_err, tot = 0, 0.0, 0
            for im, yc in by_attack[a]:
                sm = loo_map(im) if strat == "measured" else smap
                alloc = allocate(sm, n, strat)
                idx = [cells.index(c) for c in alloc]
                mu, sig = sm.mu[idx], np.maximum(sm.sigma[idx], 0.05)
                # ecc convention: positive LLR <=> bit 1; emulate the all-ZERO codeword => negate
                llr = -(2.0 * mu * yc[idx] / sig ** 2) if strat == "measured" else -(2.0 * yc[idx])
                msg, acc, _, _ = chase2_decode(llr, code, n_flip=args.n_flip)
                ok += int(acc and (msg == 0).all())
                bits_err += float((yc[idx] < 0).mean())
                tot += 1
            cols.append(f"exact {ok/tot:5.2f} cellerr {bits_err/tot:4.2f}")
        print(f"{a:>24s} " + " ".join(f"{c:>22s}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
