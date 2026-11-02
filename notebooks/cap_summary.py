# %% [markdown]
# # Capacity sweep summary (reduced protocol, CPU only)
# img2img (anime) s=0.5, first 20 prompts x attack seed 97, for 16 / 32 / 64 / 128 / 256 bits of SpecShade and Gaussian
# Shading. 16 and 256 bits come from the main experiments (same 20 prompts, same seed, 1,000-image nulls); 32 / 64 / 128 bits
# from results/cap_*. Writes results/cap_summary.json.

# %%
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_PROMPTS, SEED, ATTACK, STRENGTH = 20, 97, "img2img_anime", 0.5


def jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def sc(x):
    return -np.inf if (x is None or not np.isfinite(x)) else float(x)


def evaluate(det, null):
    rows = [r for r in jsonl(det) if r.get("kind") != "clean" and r["attack"] == ATTACK and abs(float(r["strength"]) - STRENGTH) < 1e-6
            and r["atk_seed"] == SEED and int(re.search(r"/p(\d+)__", r["image"]).group(1)) < N_PROMPTS]
    neg = np.asarray([sc(r["score"]) for r in jsonl(null)], float)
    if not rows or not len(neg):
        return None
    thr = np.quantile(neg, 0.99, method="higher")
    pos = np.asarray([sc(r["score"]) for r in rows])
    acc = np.asarray([r["bit_acc"] if r["bit_acc"] is not None else 0.5 for r in rows])
    return {"n": len(rows), "n_null": int(len(neg)), "tpr": float((pos > thr).mean()), "acc": float(acc.mean()),
            "acc_sem": float(acc.std(ddof=1) / np.sqrt(len(acc)))}


S, G = "stylering", "gaussian_shading"
SRC = {
    "ours": {16: ("final_fr2/stylering/detect_no_ecc.jsonl", "final_fr2q/stylering/detect_null_no_ecc.jsonl"),
             32: ("cap_ours32/stylering/detect_no_ecc.jsonl", "cap_ours32/stylering/detect_null_no_ecc.jsonl"),
             64: ("cap_ours64/stylering/detect_no_ecc.jsonl", "cap_ours64/stylering/detect_null_no_ecc.jsonl"),
             128: ("cap_ours128/stylering/detect_no_ecc.jsonl", "cap_ours128/stylering/detect_null_no_ecc.jsonl"),
             256: ("final256/stylering/detect_no_ecc.jsonl", "final256/stylering/detect_null_no_ecc.jsonl")},
    "gs": {16: ("final/gaussian_shading/detect.jsonl", "final_refq/gaussian_shading/detect_null.jsonl"),
           32: ("cap_gs32/gaussian_shading/detect.jsonl", "cap_gs32/gaussian_shading/detect_null.jsonl"),
           64: ("cap_gs64/gaussian_shading/detect.jsonl", "cap_gs64/gaussian_shading/detect_null.jsonl"),
           128: ("cap_gs128/gaussian_shading/detect.jsonl", "cap_gs128/gaussian_shading/detect_null.jsonl"),
           256: ("final_gs256q/gaussian_shading/detect.jsonl", "final_gs256q/gaussian_shading/detect_null.jsonl")},
}
out = {m: {str(L): evaluate(RES / d, RES / n) for L, (d, n) in src.items()} for m, src in SRC.items()}
(RES / "cap_summary.json").write_text(json.dumps(out, indent=1))
for m in out:
    for L, v in out[m].items():
        print(m, L, v)
