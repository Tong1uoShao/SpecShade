"""Verify that running the joint fallback only when neither flip candidate is aligned by the sequential searches leaves
the unified detector's output unchanged, and measure the extraction time of both variants. Uses cached inverted latents
(geo256u: single geometric edits; cmat256: compound edits; finalq/none: unwatermarked).
Usage (GPU): python -u src/proto_verify_fallback.py [--n 60]"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402


class FB:
    latent = 64


def old_extract(m, z):
    """The previous behaviour: fallback decided per candidate."""
    r0, r1 = m._extract_one(z, allow_fallback=True), m._extract_one(torch.flip(z, dims=[3]), allow_fallback=True)
    best, flipped = (r1, True) if r1.score > r0.score else (r0, False)
    return best, flipped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); a = ap.parse_args()
    m = build_method("stylering", load_config(ROOT / "configs/final256_unified.yaml"), FB())
    sets = {"geo256u": sorted(glob.glob(str(ROOT / "results/geo256u/stylering/inv/*.pt"))),
            "cmat256": sorted(glob.glob(str(ROOT / "results/cmat256/stylering/inv/*.pt"))),
            "null": sorted(glob.glob(str(ROOT / "results/finalq/none/inv/*.pt")))}
    out = {}
    for name, files in sets.items():
        rng = np.random.default_rng(0); files = [files[i] for i in rng.choice(len(files), min(a.n, len(files)), replace=False)]
        diff, t_old, t_new, fb_old, fb_new = 0, 0.0, 0.0, 0, 0
        for f in files:
            z = torch.load(f, map_location="cpu", weights_only=True)
            torch.cuda.synchronize(); t0 = time.time(); ro, fo = old_extract(m, z); torch.cuda.synchronize(); t_old += time.time() - t0
            torch.cuda.synchronize(); t0 = time.time(); rn = m.extract(z); torch.cuda.synchronize(); t_new += time.time() - t0
            fb_old += int(bool((ro.meta or {}).get("geo_fallback"))); fb_new += int(bool((rn.meta or {}).get("geo_fallback")))
            if abs(ro.score - rn.score) > 1e-9 or (ro.bits is not None and rn.bits is not None and (ro.bits != rn.bits).any()) or fo != rn.meta.get("flipped"):
                diff += 1; print(f"  DIFF {Path(f).name}: old score {ro.score:.4f} flipped={fo} fallback={bool((ro.meta or {}).get('geo_fallback'))} | new {rn.score:.4f} flipped={rn.meta.get('flipped')} fallback={bool((rn.meta or {}).get('geo_fallback'))}")
        out[name] = {"n": len(files), "differences": diff, "t_old_s": t_old / len(files), "t_new_s": t_new / len(files), "fallback_old": fb_old, "fallback_new": fb_new}
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "results/verify_fallback.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
