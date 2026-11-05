"""Generation diversity under a FIXED payload (reviewer question): with one key and one payload, the signs of the
watermarked noise are the same for every image and only the magnitudes vary, for SpecShade and for Gaussian Shading alike.
For N_PROMPTS prompts x N_SEEDS seeds each, images are generated (i) without watermark, (ii) SpecShade with a payload per
image, (iii) SpecShade with one fixed payload, (iv) GS-256 with a payload per image and (v) GS-256 with one fixed payload.
Diversity within a prompt = mean pairwise cosine distance of OpenCLIP ViT-g image embeddings (the model of the CLIP score)
and mean pairwise RMSE of the 64x64 grey thumbnails; both averaged over prompts.
Writes results/diversity/summary.json and $TAB_OUT/numbers_diversity.tex.
Usage (GPU): python -u src/proto_diversity.py --config configs/final256_unified.yaml [--n_prompts 5 --n_seeds 20]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import ensure_dir, load_config, read_jsonl  # noqa: E402
from utils.seeds import derive_payload, per_image_seed  # noqa: E402


def clip_features(paths: list[Path], device: str) -> np.ndarray:
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s34b_b88k", device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            ims = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 16]]).to(device)
            f = model.encode_image(ims); out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


def pairwise_mean(X: np.ndarray, fn) -> float:
    return float(np.mean([fn(X[i], X[j]) for i, j in itertools.combinations(range(len(X)), 2)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final256_unified.yaml")
    ap.add_argument("--gs_config", default="configs/final_gs256.yaml")
    ap.add_argument("--n_prompts", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--out", default="results/diversity")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config); gcfg = load_config(ROOT / args.gs_config)
    out = ensure_dir(ROOT / args.out)
    prompts = read_jsonl(ROOT / cfg["paths"]["prompts"])[:args.n_prompts]
    be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
    ours, gs = build_method("stylering", cfg, be), build_method("gaussian_shading", gcfg, be)
    fixed = derive_payload("fixed-user", 0, 256)
    conds = {"none": None, "ours_random": (ours, None), "ours_fixed": (ours, fixed), "gs_random": (gs, None), "gs_fixed": (gs, fixed)}
    paths = {c: {p["prompt_id"]: [] for p in prompts} for c in conds}
    for c, spec in conds.items():
        d = ensure_dir(out / c)
        for p in prompts:
            for k in range(args.n_seeds):
                seed = per_image_seed(1000 + k, int(p["prompt_id"][1:]) if p["prompt_id"][1:].isdigit() else k)
                f = d / f"{p['prompt_id']}_{k:02d}.png"
                paths[c][p["prompt_id"]].append(f)
                if f.exists():
                    continue
                if spec is None:
                    zT = be.sample_noise(seed)
                else:
                    method, payload = spec
                    zT = method.embed(payload if payload is not None else derive_payload(p["prompt_id"], seed, 256), seed)
                be.generate(zT, p["prompt"], steps=cfg["backend"]["gen_steps"], cfg=cfg["backend"]["cfg"], seed=seed).save(f)
            print(f"[{c}] {p['prompt_id']} done", flush=True)
    del be; torch.cuda.empty_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}
    for c in conds:
        clip_d, rmse = [], []
        for pid, fs in paths[c].items():
            F = clip_features(fs, dev)
            clip_d.append(pairwise_mean(F, lambda a, b: 1.0 - float(a @ b)))
            G = np.stack([np.asarray(Image.open(f).convert("L").resize((64, 64), Image.BICUBIC), dtype=np.float32) / 255.0 for f in fs])
            rmse.append(pairwise_mean(G, lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))))
        summary[c] = {"clip_dist": float(np.mean(clip_d)), "clip_dist_sd": float(np.std(clip_d)), "rmse": float(np.mean(rmse)),
                      "n_prompts": len(paths[c]), "n_seeds": args.n_seeds}
        print(c, summary[c], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    tab = Path(os.environ.get("TAB_OUT", ROOT.parent / "docs" / "manuscripts" / "tables")); tab.mkdir(parents=True, exist_ok=True)
    tag = {"none": "None", "ours_random": "OursRand", "ours_fixed": "OursFixed", "gs_random": "GSRand", "gs_fixed": "GSFixed"}
    lines = ["% generated by code/src/proto_diversity.py -- do not edit"]
    for c, s in summary.items():
        lines += [f"\\newcommand{{\\nDivClip{tag[c]}}}{{{s['clip_dist']:.3f}}}", f"\\newcommand{{\\nDivRmse{tag[c]}}}{{{s['rmse']:.3f}}}"]
    lines.append(f"\\newcommand{{\\nDivPrompts}}{{{args.n_prompts}}}"); lines.append(f"\\newcommand{{\\nDivSeeds}}{{{args.n_seeds}}}")
    (tab / "numbers_diversity.tex").write_text("\n".join(lines) + "\n")
    print("wrote", out / "summary.json", tab / "numbers_diversity.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
