"""Oracle-aligned vs. blind detection on the compound attacks that defeat the blind detector (reviewer diagnosis).

For every compound image the inverted noise is realigned with the TRUE geometric parameters of the attack (known here,
because we applied the attack) and read directly; this is compared with the stored blind detection of the same image.
If the oracle read succeeds where the blind detector fails, the watermark evidence is there and the failure lies in the
search / gating; if the oracle read fails as well, the regeneration has removed the evidence.
The oracle threshold is the 99th percentile of the direct read of 1,000 unwatermarked images after the same transform.
Written to results/oracle_sync/summary.json.
Usage (GPU): python -u src/proto_oracle_sync.py --exp combo256 --n_null 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import decrypt, hex_to_bits  # noqa: E402
from core.fourier_rings import ShiftScaleSync  # noqa: E402
from core.modulation import combine_positions, hard_bits, read_cells  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import ensure_dir, load_config, read_jsonl  # noqa: E402

# True parameters (zoom s, angle theta, shift ty, tx in latent px) in the conventions of ShiftScaleSync.apply_joint.
# A PIL rotation by +30 deg is undone by theta = 330. A crop of area 0.75 resized back is undone by a zoom-out by
# sqrt(0.75) plus the shift of the crop centre; CropScale draws the crop position from default_rng(seed), so the shift
# depends only on the attack seed. The shifts are the medians found by the blind search on the plain crop_scale@0.75
# images of geo256 (100 images per seed, quarter-pixel grid), which agree with the crop boxes of the two seeds.
TRUE = {"combo_style_crop": {97: (0.866, 0.0, -1.75, 3.0), 101: (0.866, 0.0, -4.0, 1.75)},
        "combo_regen_rotate": {97: (1.0, 330.0, 0.0, 0.0), 101: (1.0, 330.0, 0.0, 0.0)}}
TOL = {"theta": 1.5, "scale": 0.03, "shift": 1.0}


def read(method, z: torch.Tensor):
    y = read_cells(z, method.alloc, method.chips, hard=False)
    yp = combine_positions(y, method.pos_of, method._n_bits, method.sizes)
    bits = decrypt(hard_bits(yp)[:method._n_bits], method.key)
    return bits, float(np.abs(yp[:method._n_bits]).mean())


def crop_box(seed: int, frac: float = 0.75, w: int = 512):
    cw = int(w * np.sqrt(frac))
    rng = np.random.default_rng(int(seed))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, w - cw + 1))
    return x0, y0, cw, ((x0 + cw / 2 - w / 2) / 8, (y0 + cw / 2 - w / 2) / 8)


def params_ok(meta: dict, true: tuple) -> bool:
    s, th, ty, tx = true
    es, eth, ety, etx = meta.get("scale", 1.0), meta.get("theta", 0.0), meta.get("ty", 0.0), meta.get("tx", 0.0)
    if meta.get("geo_fallback") and not meta.get("geo_gated", True):
        es, eth, ety, etx = meta["geo_scale"], meta["geo_theta"], meta["geo_ty"], meta["geo_tx"]
    dth = min(abs(eth - th) % 360.0, 360.0 - abs(eth - th) % 360.0)
    return dth <= TOL["theta"] and abs(es - s) <= TOL["scale"] and abs(ety - ty) <= TOL["shift"] and abs(etx - tx) <= TOL["shift"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="combo256")
    ap.add_argument("--config", default="configs/final256.yaml")
    ap.add_argument("--null_exp", default="final256", help="experiment whose null file lists the unwatermarked images")
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="results/oracle_sync")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    base = ROOT / cfg["paths"]["results"] / args.exp / "stylering"
    for seed in (97, 101):
        print("crop box seed", seed, crop_box(seed), "oracle shift (ty, tx)", TRUE["combo_style_crop"][seed][2:], flush=True)
    payload_of = {g["image"]: g["payload_hex"] for g in read_jsonl(base / "gen.jsonl")}
    blind = {r["image"]: r for r in read_jsonl(base / "detect_no_ecc.jsonl")}
    thr_blind = float(np.quantile([r["score"] for r in read_jsonl(base / "detect_null_no_ecc.jsonl")], 0.99, method="higher"))
    atk = [a for a in read_jsonl(base / "atk.jsonl") if a["attack"] in TRUE]
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    method = build_method("stylering", cfg, be)
    steps = cfg["backend"]["inv_steps"]

    def invert(paths):
        return be.invert_batch([Image.open(ROOT / p).convert("RGB") for p in paths], steps=steps).cpu().float()

    rows = []
    for b0 in range(0, len(atk), args.batch):
        batch = atk[b0:b0 + args.batch]
        Z = invert([a["image"] for a in batch])
        for a, z in zip(batch, Z):
            z = z.unsqueeze(0)
            truth = hex_to_bits(payload_of[a["src_image"]])
            true = TRUE[a["attack"]][int(a["atk_seed"])]
            bits_o, score_o = read(method, ShiftScaleSync.apply_joint(z, *true))
            bits_r, score_r = read(method, z)
            br = blind[a["image"]]
            m = br.get("meta", {})
            gated = bool(m.get("gated", True)) and bool(m.get("crop_gated", True)) and bool(m.get("geo_gated", True))
            rows.append({"image": a["image"], "attack": a["attack"], "seed": int(a["atk_seed"]),
                         "oracle_acc": float((bits_o == truth).mean()), "oracle_score": score_o,
                         "raw_acc": float((bits_r == truth).mean()), "raw_score": score_r,
                         "blind_acc": float(br["bit_acc"]), "blind_score": float(br["score"]), "blind_gated": gated,
                         "blind_params_ok": (not gated) and params_ok(m, true)})
        if (b0 // args.batch) % 10 == 0:
            r = rows[-1]
            print(f"[{len(rows)}/{len(atk)}] {r['attack']} oracle acc {r['oracle_acc']:.3f} blind acc {r['blind_acc']:.3f}", flush=True)

    null_imgs = [r["image"] for r in read_jsonl(ROOT / cfg["paths"]["results"] / args.null_exp / "stylering" / "detect_null_no_ecc.jsonl")][:args.n_null]
    null_o, null_r = [], []
    t_null = TRUE["combo_style_crop"][97]
    for b0 in range(0, len(null_imgs), args.batch):
        Z = invert(null_imgs[b0:b0 + args.batch])
        for z in Z:
            z = z.unsqueeze(0)
            null_o.append(read(method, ShiftScaleSync.apply_joint(z, *t_null))[1])
            null_r.append(read(method, z)[1])
        if (b0 // args.batch) % 50 == 0:
            print(f"null [{len(null_o)}/{len(null_imgs)}]", flush=True)
    thr_o = float(np.quantile(null_o, 0.99, method="higher")) if null_o else float("nan")
    thr_r = float(np.quantile(null_r, 0.99, method="higher")) if null_r else float("nan")

    summary = {"thr_blind": thr_blind, "thr_oracle": thr_o, "thr_raw": thr_r, "n_null": len(null_o), "attacks": {}}
    for name in TRUE:
        R = [r for r in rows if r["attack"] == name]
        f = lambda k: np.asarray([r[k] for r in R])
        g, ok = f("blind_gated"), f("blind_params_ok")
        acc_o = f("oracle_acc")
        entry = {"n": len(R),
                 "blind_tpr": float((f("blind_score") > thr_blind).mean()), "blind_acc": float(f("blind_acc").mean()),
                 "blind_frac_gated": float(g.mean()), "blind_frac_accepted_correct": float(ok.mean()),
                 "blind_frac_accepted_wrong": float((~g & ~ok).mean()),
                 "blind_acc_gated": float(f("blind_acc")[g].mean()) if g.any() else None,
                 "blind_acc_accepted_correct": float(f("blind_acc")[ok].mean()) if ok.any() else None,
                 "oracle_tpr": float((f("oracle_score") > thr_o).mean()), "oracle_acc": float(acc_o.mean()),
                 "oracle_acc_q": [float(x) for x in np.quantile(acc_o, [0.1, 0.25, 0.5, 0.75, 0.9])],
                 "oracle_frac_acc_above_0.6": float((acc_o > 0.6).mean()),
                 "raw_tpr": float((f("raw_score") > thr_r).mean()), "raw_acc": float(f("raw_acc").mean()),
                 "oracle_tpr_at_blind_thr": float((f("oracle_score") > thr_blind).mean())}
        summary["attacks"][name] = entry
        print(name, json.dumps(entry), flush=True)
    out = ensure_dir(ROOT / args.out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("ORACLE SYNC DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
