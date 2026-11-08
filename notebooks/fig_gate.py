# %% [markdown]
# # Supplementary Fig. S1: sweep of the joint-search gate (CPU only)
# Fraction of images whose joint-search peak passes a gate kappa: unwatermarked images (false alignments) vs. the two
# compound attacks that defeat the detector (true peaks), and the subset of those whose watermark is detectable once
# aligned (oracle alignment, Sec. IV-D). Reads results/gate_sweep.json (written by the gate-sweep analysis) and
# results/oracle_sync/rows.jsonl. Usage: FIG_OUT=/path/to/paper/figs python notebooks/fig_gate.py

# %%
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("FIG_OUT", ROOT.parent / "docs" / "manuscripts" / "figs"))
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.bbox": "tight"})
d = json.load(open(RES / "gate_sweep.json"))
orc = {r["image"]: r for r in (json.loads(l) for l in open(RES / "oracle_sync" / "rows.jsonl"))}
thr_o = json.load(open(RES / "oracle_sync" / "summary.json"))["thr_oracle"]
rows = [json.loads(l) for l in open(RES / "combo256u" / "stylering" / "detect_no_ecc.jsonl")]
ks = np.arange(3.0, 10.01, 0.25)
fig, ax = plt.subplots(figsize=(3.32, 2.1))
zn = np.asarray(d["null_z"])
ax.plot(ks, [(zn >= k).mean() for k in ks], color="k", ls=":", label="unwatermarked")
for a, lab, c in (("combo_regen_rotate", "regen.+rot.", "#1f5fbf"), ("combo_style_crop", "img2img+crop", "#2e8b57")):
    R = [r for r in rows if r["attack"] == a and "geo_z" in r["meta"]]
    z = np.asarray([r["meta"]["geo_z"] for r in R])
    det = np.asarray([orc.get(r["image"], {}).get("oracle_score", 0.0) > thr_o for r in R])
    ax.plot(ks, [(z >= k).mean() for k in ks], color=c, label=f"{lab}")
    ax.plot(ks, [((z >= k) & det).mean() for k in ks], color=c, ls="--", label=f"{lab}, detectable")
ax.axvline(8.0, color="#b54708", lw=0.8); ax.text(8.1, 0.95, r"$\kappa_{\mathrm{crop}}$", color="#b54708", fontsize=8, va="top")
ax.set_xlabel(r"gate $\kappa$ of the joint search (robust $z$-score)"); ax.set_ylabel("fraction passing the gate")
ax.set_ylim(0, 1.0); ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, loc="center right", fontsize=8, handlelength=1.6, labelspacing=0.3)
fig.savefig(OUT / "gate.pdf"); plt.close(fig)
print("wrote", OUT / "gate.pdf")
