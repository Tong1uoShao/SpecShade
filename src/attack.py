"""Stage 3: attack the generated images of one method with a block of the attack matrix.

results/{exp}/{method}/atk/*.png + atk.jsonl (resume by output path).
Usage (GPU): python -u src/attack.py --exp main --method stylering --blocks style [--names img2img_anime ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.registry import build_matrix, free_all, run_attacks  # noqa: E402
from utils.io import load_config, read_jsonl  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--blocks", default="style", help="comma list of attacks.yaml blocks")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--seeds", default=None, help="comma list; default cfg.seeds")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--attacks_config", default="configs/attacks.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only first N generated images")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    cfg["_root"] = str(ROOT)
    acfg = load_config(ROOT / args.attacks_config)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["seeds"])
    log = get_logger(f"attack_{args.exp}_{args.method}", ROOT / "logs")
    base = ROOT / cfg["paths"]["results"] / args.exp / args.method
    gen = read_jsonl(base / "gen.jsonl")
    if args.limit:
        gen = gen[:args.limit]
    images = [{"image": g["image"], "prompt": g["prompt"], "src_key": g["prompt_id"]} for g in gen]
    specs = build_matrix(acfg, args.blocks.split(","), args.names)
    log.info(f"exp={args.exp} method={args.method} images={len(images)} specs={[s.name for s in specs]} seeds={seeds}")
    run_attacks(images, specs, seeds, cfg, base / "atk", base / "atk.jsonl", log)
    free_all()
    log.info("ATTACK DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
