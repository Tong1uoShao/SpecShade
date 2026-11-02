"""Stage 1 (offline, once): placement-space probing -> survival map (Method 3.3, RQ2).

Every cell (channel x ring [x quadrant]) carries a known random sign; probe images are
generated, attacked with the style block (+ 'none' = clean round trip), inverted, and the
per-cell sign-survival rate S(g), mean mu(g) and std sigma(g) of y*sigma_true are aggregated
overall and per attack. Outputs results/probe/survival_map.npz + heatmaps.
Usage (GPU): python -u src/probe.py [--n 200] [--quadrants] [--seeds 97,101] [--blocks style]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.registry import AttackSpec, build_matrix, free_all, run_attacks  # noqa: E402
from core.allocation import SurvivalMap  # noqa: E402
from core.fourier_rings import enumerate_cells  # noqa: E402
from core.modulation import embed_bits, read_cells  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from utils.io import append_jsonl, done_keys, ensure_dir, load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seeds import per_image_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--quadrants", action="store_true", help="add 2x2 spatial cells as control axis")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--blocks", default="style")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attacks_config", default="configs/attacks.yaml")
    ap.add_argument("--out", default="results/probe")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    cfg["_root"] = str(ROOT)
    acfg = load_config(ROOT / args.attacks_config)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["seeds"])
    log = get_logger("probe", ROOT / "logs")
    out = ensure_dir(ROOT / args.out)
    gen_dir, atk_dir = ensure_dir(out / "gen"), ensure_dir(out / "atk")
    b = cfg["backend"]
    cells = enumerate_cells(latent=b["latent"])
    if args.quadrants:
        cells = cells + enumerate_cells(quadrants=True, latent=b["latent"])
    keys = np.array([c.key() for c in cells], dtype=np.int64)

    # ---- S1: probe images (all cells lit with known random signs) ----
    prompts = read_jsonl(ROOT / cfg["paths"]["prompts"])[:args.n]
    gen_meta = out / "gen.jsonl"
    done = done_keys(gen_meta, "prompt_id")
    be = None
    for i, p in enumerate(prompts):
        if p["prompt_id"] in done:
            continue
        if be is None:
            be = SDBackend(model_id=b["model_id"], prefer_fp16=b["prefer_fp16"], size=b["size"])
        seed = per_image_seed(4242, i)
        signs = np.random.default_rng(seed).integers(0, 2, len(cells))     # known bits per image
        zT = SDBackend.sample_noise(seed, latent=b["latent"])
        zw = embed_bits(zT, cells, signs)
        t0 = time.time()
        img = be.generate(zw, p["prompt"], steps=b["gen_steps"], cfg=b["cfg"], seed=seed)
        path = gen_dir / f"{p['prompt_id']}.png"
        img.save(path)
        append_jsonl(gen_meta, {"prompt_id": p["prompt_id"], "prompt": p["prompt"], "seed": seed,
                                "image": str(path.relative_to(ROOT)), "t_gen_s": round(time.time() - t0, 2)})
        if i % 10 == 0:
            log.info(f"[gen {i+1}/{len(prompts)}] {time.time()-t0:.1f}s")
    if be is not None:
        be.free()
        be = None
    gen = read_jsonl(gen_meta)
    log.info(f"probe images: {len(gen)}")

    # ---- S2: attacks (style block + none) ----
    specs = [AttackSpec("none", "none", None, (0.0,))] + build_matrix(acfg, args.blocks.split(","), args.names)
    images = [{"image": g["image"], "prompt": g["prompt"], "src_key": g["prompt_id"]} for g in gen]
    run_attacks(images, specs, seeds, cfg, atk_dir, out / "atk.jsonl", log)
    free_all()
    atk = list({a["image"]: a for a in read_jsonl(out / "atk.jsonl")}.values())   # dedup by image (resume safety)
    log.info(f"attacked images: {len(atk)}")

    # ---- S3: invert + read every cell ----
    read_meta = out / "reads.jsonl"
    done_r = done_keys(read_meta, "image")
    todo = [a for a in atk if a["image"] not in done_r]
    if todo:
        be = SDBackend(model_id=b["model_id"], prefer_fp16=b["prefer_fp16"], size=b["size"])
        B = 8   # batched inversion (numerically identical to per-image, ~2-3x throughput)
        for b0 in range(0, len(todo), B):
            batch = todo[b0:b0 + B]
            Z = be.invert_batch([Image.open(ROOT / a["image"]).convert("RGB") for a in batch],
                                steps=b["inv_steps"]).cpu()
            for a, z in zip(batch, Z):
                y = read_cells(z.unsqueeze(0), cells)
                append_jsonl(read_meta, {"image": a["image"], "src_key": a["src_key"], "attack": a["attack"],
                                         "strength": a["strength"], "atk_seed": a["atk_seed"], "y": np.round(y, 4).tolist()})
            if (b0 // B) % 25 == 0:
                log.info(f"[read {min(b0+B, len(todo))}/{len(todo)}]")
        be.free()

    # ---- S4: aggregate survival map ----
    reads = list({r["image"]: r for r in read_jsonl(read_meta)}.values())          # dedup by image
    sign_of = {}
    for g in gen:
        sign_of[g["prompt_id"]] = 2 * np.random.default_rng(g["seed"]).integers(0, 2, len(cells)) - 1
    Y, A = [], []
    for r in reads:
        Y.append(np.array(r["y"]) * sign_of[r["src_key"]])     # y * true sign -> positive = survived
        A.append(f"{r['attack']}" if r["attack"] == "none" else f"{r['attack']}@{r['strength']}")
    Y, A = np.array(Y), np.array(A)
    S_all, mu_all, sigma_all = (Y > 0).mean(0), Y.mean(0), Y.std(0)
    per_attack = {a: (Y[A == a] > 0).mean(0) for a in sorted(set(A))}
    smap = SurvivalMap(cells, S_all, mu_all, sigma_all, per_attack,
                       {"n_images": len(gen), "n_reads": len(reads), "seeds": seeds, "attacks": sorted(set(A)),
                        "note": "S=sign survival; mu/sigma of y*true_sign over all attacks incl. none"})
    smap.save(str(out / "survival_map.npz"))
    # heatmaps (channel x ring) overall + per attack, full-image cells only
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        full = [i for i, c in enumerate(cells) if c.q is None]
        n_ch, r_min = 4, min(cells[i].r for i in full)
        r_max = max(cells[i].r for i in full)
        def grid(v):
            G = np.full((n_ch, r_max - r_min + 1), np.nan)
            for i in full:
                G[cells[i].c, cells[i].r - r_min] = v[i]
            return G
        names = ["ALL"] + sorted(set(A))
        fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.4 * len(names) + 1))
        for ax, nm in zip(np.atleast_1d(axes), names):
            v = S_all if nm == "ALL" else per_attack[nm]
            im = ax.imshow(grid(v), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_ylabel(nm, fontsize=7); ax.set_yticks(range(n_ch)); ax.set_xticks(range(0, r_max - r_min + 1, 5))
            ax.set_xticklabels(range(r_min, r_max + 1, 5))
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.6, label="sign survival S(g)")
        fig.savefig(out / "survival_heatmap.png", dpi=150, bbox_inches="tight")
    except Exception as e:  # heatmap is cosmetic
        log.info(f"heatmap skipped: {e}")
    summary = {a: {"S_low(1-8)": float(np.mean([per_attack[a][i] for i in range(len(cells)) if cells[i].q is None and cells[i].r <= 8])),
                   "S_mid(9-16)": float(np.mean([per_attack[a][i] for i in range(len(cells)) if cells[i].q is None and 9 <= cells[i].r <= 16])),
                   "S_high(17+)": float(np.mean([per_attack[a][i] for i in range(len(cells)) if cells[i].q is None and cells[i].r >= 17]))}
               for a in per_attack}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("survival by band per attack:\n" + json.dumps(summary, indent=1))
    log.info("PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
