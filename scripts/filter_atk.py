"""Create an experiment directory that re-uses already attacked images but lists only the chosen cells.

detect.py then visits just these images, so ablated detectors can be run on a subset of an existing attack set.
Usage: python scripts/filter_atk.py --src geo256 --dst abl_sync_norot --cells rotate:75 crop_scale:0.75 --n_prompts 50
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--method", default="stylering")
ap.add_argument("--cells", nargs="+", required=True, help="attack:strength")
ap.add_argument("--n_prompts", type=int, default=50)
args = ap.parse_args()

cells = [(c.split(":")[0], float(c.split(":")[1])) for c in args.cells]
src, dst = ROOT / "results" / args.src / args.method, ROOT / "results" / args.dst / args.method
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "gen.jsonl", dst / "gen.jsonl")
keep = []
for line in open(src / "atk.jsonl"):
    a = json.loads(line)
    if int(a["src_key"][1:]) < args.n_prompts and any(a["attack"] == n and abs(float(a["strength"]) - s) < 1e-6 for n, s in cells):
        keep.append(line if line.endswith("\n") else line + "\n")
(dst / "atk.jsonl").write_text("".join(keep))
print(f"{args.dst}: {len(keep)} attacked images from {args.src}")
