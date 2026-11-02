"""Stage 5: aggregate detect.jsonl of all methods of an experiment into tables; image quality.

Outputs results/{exp}/tables/{main,by_attack}.csv, trace.csv (--trace), quality.json (--quality).
Usage: python src/evaluate.py --exp main --methods stylering,gaussian_shading,gaussmarker,ringid,none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.ecc import hex_to_bits  # noqa: E402
from utils.io import ensure_dir, load_config, read_jsonl  # noqa: E402
from utils.metrics import clip_scores, fid, tpr_at_fpr, welch_ttest  # noqa: E402


def _load(base: Path, variant: str) -> pd.DataFrame:
    rows = read_jsonl(base / ("detect.jsonl" if variant == "soft" else f"detect_{variant}.jsonl"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cell"] = df.apply(lambda r: "clean" if r["kind"] == "clean" else f"{r['attack']}@{r['strength']}", axis=1)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--methods", required=True, help="comma list; include 'none' for FPR calibration")
    ap.add_argument("--variant", default="soft")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quality", action="store_true", help="CLIP/FID/t-test on gen images (GPU)")
    ap.add_argument("--trace", action="store_true", help="multi-user exact-match identification rate")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    res = ROOT / cfg["paths"]["results"] / args.exp
    tables = ensure_dir(res / "tables")
    methods = args.methods.split(",")

    # ---- negatives for FPR: scores of unwatermarked ('none') images decoded with each method's key ----
    # detect.py must be run with --method <m> on exp '{exp}' AND, for calibration, on the 'none'
    # generation set via results/{exp}/{m}/detect_null.jsonl (produced by scripts/main_exp.sh).
    rows = []
    for m in methods:
        if m == "none":
            continue
        df = _load(res / m, args.variant)
        if df.empty:
            print(f"[warn] no detections for {m}")
            continue
        null_name = "detect_null.jsonl" if args.variant == "soft" else f"detect_null_{args.variant}.jsonl"
        null = pd.DataFrame(read_jsonl(res / m / null_name))     # same decoder as the positives
        neg = null["score"].tolist() if not null.empty else []
        if not neg:
            print(f"[warn] no null scores for {m} ({null_name}); TPR columns will be NaN")
        for cell, g in df.groupby("cell"):
            acc = g["bit_acc"].dropna()                     # accepted (decodable) samples only
            tpr1, thr1 = tpr_at_fpr(g["score"].tolist(), neg, 0.01) if neg else (np.nan, np.nan)
            tpr01, _ = tpr_at_fpr(g["score"].tolist(), neg, 0.001) if neg else (np.nan, np.nan)
            # OURS: bit_acc is reported over accepted samples (a rejected image yields no bits);
            # exact_rate is over ALL samples (rejection = payload not recovered), so that
            # exact_rate <= accept_rate and the two together describe the reject rule.
            # bit_acc_all: rejected images scored at chance (0.5) so methods with and without a
            # reject rule share one denominator.
            rows.append({"method": m, "cell": cell, "n": len(g),
                         "bit_acc_mean": float(acc.mean()) if len(acc) else np.nan,
                         "bit_acc_std": float(acc.std(ddof=0)) if len(acc) else np.nan,
                         "bit_acc_all": float(g["bit_acc"].fillna(0.5).mean()),
                         "exact_rate": float((g["bit_acc"] == 1.0).sum() / len(g)) if len(g) else np.nan,
                         "exact_rate_acc": float((acc == 1.0).mean()) if len(acc) else np.nan,
                         "accept_rate": float(g["accepted"].dropna().astype(float).mean()) if g["accepted"].notna().any() else np.nan,
                         "tpr_1fpr": tpr1, "tpr_01fpr": tpr01, "thr_1fpr": thr1})
    main_df = pd.DataFrame(rows)
    main_df.to_csv(tables / f"main_{args.variant}.csv", index=False)
    if not main_df.empty:
        piv = main_df.pivot(index="cell", columns="method", values="bit_acc_mean")
        piv.to_csv(tables / f"by_attack_bitacc_{args.variant}.csv")
        print(piv.round(3).to_string())

    if args.trace and not main_df.empty:
        # identification = decoded payload exactly matches the image's own payload (16 bit -> 65536 ids)
        tr = main_df[["method", "cell", "exact_rate", "n"]]
        tr.to_csv(tables / "trace.csv", index=False)
        print(tr.to_string())

    if args.quality:
        q = {}
        none_gen = read_jsonl(res / "none" / "gen.jsonl")
        none_paths = [ROOT / g["image"] for g in none_gen]
        none_prompts = [g["prompt"] for g in none_gen]
        none_clip = clip_scores(none_paths, none_prompts, cfg["quality"]["clip_model"]) if none_gen else np.array([])
        q["none"] = {"clip_mean": float(none_clip.mean()) if none_clip.size else None,
                     "fid": fid(res / "none" / "gen", ROOT / cfg["paths"]["coco_fid"]) if none_gen else None}
        for m in methods:
            if m == "none":
                continue
            gen = read_jsonl(res / m / "gen.jsonl")
            if not gen:
                continue
            cs = clip_scores([ROOT / g["image"] for g in gen], [g["prompt"] for g in gen], cfg["quality"]["clip_model"])
            q[m] = {"clip_mean": float(cs.mean()), "clip_std": float(cs.std()),
                    "fid": fid(res / m / "gen", ROOT / cfg["paths"]["coco_fid"]),
                    "ttest_p_clip_vs_none": welch_ttest(cs, none_clip) if none_clip.size else None}
        (res / "quality.json").write_text(json.dumps(q, indent=1))
        print(json.dumps(q, indent=1))
    print("EVALUATE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
