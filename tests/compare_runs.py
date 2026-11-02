"""Compare two StyleRing runs on their common prompt ids / attack seeds (e.g. a new-machine smoke run
vs the reference experiment): per-family exact rate, bit accuracy, mean score, and clean-image scores.
usage: python tests/compare_runs.py --ref abl_sec8 --new smoke4090"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.io import read_jsonl  # noqa: E402


def load(exp):
    rows = read_jsonl(ROOT / "results" / exp / "stylering" / "detect.jsonl")
    out = {}
    for r in rows:
        pid = Path(r["image"]).name.split("__")[0].split(".")[0]
        key = (pid, r["kind"], r["attack"], r["strength"], r["atk_seed"])
        out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="abl_sec8")
    ap.add_argument("--new", default="smoke4090")
    a = ap.parse_args()
    R, N = load(a.ref), load(a.new)
    common = sorted(set(R) & set(N))
    print(f"common detections: {len(common)} (ref {len(R)}, new {len(N)})")
    fam = collections.defaultdict(lambda: {"ref": [], "new": []})
    for k in common:
        f = "clean" if k[1] == "clean" else f"{k[2].rsplit('_', 1)[0]}@{k[3]}"
        for tag, D in (("ref", R), ("new", N)):
            r = D[k]
            s = r["score"]
            fam[f][tag].append((float(r["bit_acc"] == 1.0) if r["bit_acc"] is not None else 0.0,
                                r["bit_acc"] if r["bit_acc"] is not None else 0.5,
                                s if (s is not None and np.isfinite(s)) else float("nan")))   # rejected -> nan
    print(f"{'family':>14} | {'n':>3} | {'exact ref/new':>15} | {'bitacc ref/new':>15} | {'score ref/new':>15}")
    for f in sorted(fam, key=lambda x: (x != "clean", x)):
        A, B = np.array(fam[f]["ref"]), np.array(fam[f]["new"])
        print(f"{f:>14} | {len(A):>3} | {A[:,0].mean():6.2f} /{B[:,0].mean():6.2f} | {A[:,1].mean():6.3f} /{B[:,1].mean():6.3f} | "
              f"{np.nanmean(A[:,2]):6.2f} /{np.nanmean(B[:,2]):6.2f}")
    cl = [(k[0], R[k]["score"], N[k]["score"]) for k in common if k[1] == "clean"]
    for pid, s1, s2 in cl:
        print(f"clean {pid}: ref {s1:.3f}  new {s2:.3f}  d={abs(s1 - s2):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
