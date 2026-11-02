# %% [markdown]
# # Fig. 1 -- teaser: the same prompt watermarked by GS-256 and SpecShade-256, stylized (IP2P) and mirrored
# Reads real detections (exps combo256, combo_gs256) and thresholds from their null runs. Same visual style as the
# framework figure (handwritten italic labels, pastel dashed boxes, drop shadows); the result boxes are sized from the
# rendered text extents, so no line can leave its box. Runs on the GPU node:
#   FIG_OUT=$FIG_OUT FONT_DIR=$FONT_DIR PID=p0014 python notebooks/fig_teaser.py

# %%
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("FIG_OUT", ROOT.parent / "docs" / "manuscripts" / "figs"))
FONTS = Path(os.environ.get("FONT_DIR", "$FONT_DIR"))
PID = os.environ.get("PID", "p0014")
for f in FONTS.glob("ComicNeue-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Comic Neue", "mathtext.fontset": "stix", "font.size": 7.5, "pdf.fonttype": 42})
INK = "#222222"


def find(det_path, image_suffix):
    for l in open(det_path):
        r = json.loads(l)
        if r["image"].endswith(image_suffix):
            return r
    raise KeyError(image_suffix)


def tau(null_path):
    return float(np.quantile([json.loads(l)["score"] for l in open(null_path)], 0.99))


rows = []
for name, exp, m, det, null in (("Gaussian Shading-256", "combo_gs256", "gaussian_shading", "detect.jsonl", "detect_null.jsonl"),
                                ("SpecShade-256 (ours)", "combo256", "stylering", "detect_no_ecc.jsonl", "detect_null_no_ecc.jsonl")):
    base = RES / exp / m
    atk = f"{PID}__combo_style_mirror_1_97.png"
    r = find(base / det, atk)
    gen_dir = {"combo_gs256": "final_gs256/gaussian_shading", "combo256": "final_fr256/stylering"}[exp]
    rows.append(dict(name=name, src=RES / gen_dir / "gen" / f"{PID}.png", atk=base / "atk" / atk,
                     score=float(r["score"]), acc=float(r["bit_acc"]), tau=tau(base / null)))

W, H = 3.5, 2.42                                   # inches = data units
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
S = 0.93
CX = [0.74, 1.76]                                  # image columns
BX = 2.86                                          # result column
RY = [1.75, 0.60]                                  # row centres


def text(x, y, s, size=7.5, color=INK, style="italic", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha="center", va="center", fontstyle=style, zorder=8, **kw)


def img(path, cx, cy, s):
    ax.add_patch(Rectangle((cx - s / 2 + 0.025, cy - s / 2 - 0.025), s, s, fc="#9A9A9A", ec="none", alpha=0.5, zorder=1))
    ax.imshow(np.asarray(Image.open(path).convert("RGB")), extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2), zorder=2)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec="#3A3A3A", lw=0.5, zorder=3))


text(CX[0], RY[0] + S / 2 + 0.11, "watermarked")
text(CX[1], RY[0] + S / 2 + 0.11, "stylized + mirrored")
text(BX, RY[0] + S / 2 + 0.11, "detection")
renderer = fig.canvas.get_renderer()
for r, cy in zip(rows, RY):
    ok = r["score"] >= r["tau"]
    col, fc = ("#2E8B57", "#E8F5EC") if ok else ("#C0392B", "#FBE8E6")
    ax.text(0.13, cy, r["name"].replace(" (ours)", "\n(ours)"), rotation=90, fontsize=7.5, fontweight="bold",
            fontstyle="italic", ha="center", va="center", color=INK, linespacing=1.1)
    img(r["src"], CX[0], cy, S)
    img(r["atk"], CX[1], cy, S)
    t = [text(BX, cy + 0.17, "detected" if ok else "not detected", size=8.5, color=col, fontweight="bold"),
         text(BX, cy, rf"$S={r['score']:.2f}\,{'\\geq' if ok else '<'}\,\tau={r['tau']:.2f}$", size=8),
         text(BX, cy - 0.16, f"bit accuracy {r['acc']:.2f}")]
    bb = [x.get_window_extent(renderer).transformed(ax.transData.inverted()) for x in t]
    x0, x1 = min(b.x0 for b in bb) - 0.07, max(b.x1 for b in bb) + 0.07
    y0, y1 = min(b.y0 for b in bb) - 0.07, max(b.y1 for b in bb) + 0.07
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=col,
                                lw=0.9, ls=(0, (3, 2)), zorder=5))
    assert x0 > CX[1] + S / 2 + 0.05 and x1 < W, (x0, x1)
    ax.add_patch(FancyArrowPatch((CX[1] + S / 2 + 0.03, cy), (x0 - 0.02, cy), arrowstyle="-|>", mutation_scale=7,
                                 color=INK, lw=0.8, shrinkA=0, shrinkB=0, zorder=4))
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "teaser.pdf"); fig.savefig(OUT / "teaser.png", dpi=250)
print(json.dumps([{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in rows]))
