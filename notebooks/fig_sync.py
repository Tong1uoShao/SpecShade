# %% [markdown]
# # Fig. 4 -- self-synchronization on real data (GPU)
# One SpecShade-256 image (exp final_fr256, p0000) and its rotated (30 deg) and cropped (a=0.75) versions from
# exp geo256 are inverted with DDIM; we plot (a) the images with the score read directly vs. after
# synchronization, (b) the rotation score S(theta) of the rotated image, (c) the translation map S~(gamma*, t) of
# the cropped image at the best zoom. Usage (GPU node):
#   FIG_OUT=$FIG_OUT python notebooks/fig_sync.py

# %%
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.fourier_rings import ShiftScaleSync, rotate_latents, zoom_latents  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", ROOT.parent / "docs" / "manuscripts" / "figs"))
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
PID = os.environ.get("PID", "p0000")
IMGS = {"watermarked": ROOT / f"results/final_fr256/stylering/gen/{PID}.png",
        "rotated $30^\\circ$": ROOT / f"results/geo256/stylering/atk/{PID}__rotate_30_97.png",
        "cropped $a{=}0.75$": ROOT / f"results/geo256/stylering/atk/{PID}__crop_scale_0.75_97.png"}

# %%
cfg = load_config(ROOT / "configs/final256.yaml")
be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
m = build_method("stylering", cfg, be)
ims = {k: Image.open(p).convert("RGB") for k, p in IMGS.items()}
Z = be.invert_batch(list(ims.values()), steps=cfg["backend"]["inv_steps"]).cpu().float()
z = {k: Z[i:i + 1] for i, k in enumerate(ims)}

direct = {k: float(m.sync_stat(v)[0]) for k, v in z.items()}
synced = {k: float(m.extract(v).score) for k, v in z.items()}
angles = np.arange(0.0, 360.0, 1.0)
s_theta = m.sync_stat(rotate_latents(z["rotated $30^\\circ$"], list(angles)))
s_star, ty, tx, peak, _, zsc = m.sync.search(z["cropped $a{=}0.75$"], m.scales)
M = m.sync.stat_maps(zoom_latents(z["cropped $a{=}0.75$"], [s_star]))[0].numpy()
M = np.fft.fftshift(M)                           # centre = zero shift
U = M.shape[0]; up = m.sync.up; half = int(32 * up)
c = U // 2
M = M[c - half:c + half, c - half:c + half]
print({"direct": direct, "synced": synced, "theta_peak": float(angles[int(np.argmax(s_theta))]),
       "crop": {"s": s_star, "ty": ty, "tx": tx, "peak": peak, "z": zsc}})

# %%
fig = plt.figure(figsize=(2.94, 2.85))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.7, wspace=1.8)
for i, (k, im) in enumerate(ims.items()):
    ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(k, fontsize=8)
    ax.set_xlabel(f"direct {direct[k]:.2f}\nsynced {synced[k]:.2f}", fontsize=7.5)
ax = fig.add_subplot(gs[1, 0:3])
ax.plot(angles, s_theta, lw=0.7, color="#1f5fbf")
ax.set_xlim(0, 360); ax.set_xticks([0, 120, 240, 360]); ax.set_xlabel(r"angle $\theta$ (deg)", fontsize=8)
ax.set_ylabel(r"$S(\theta)$", fontsize=8); ax.set_title("(b) rotation", fontsize=8.5)
k = int(np.argmax(s_theta))
ax.annotate(f"{angles[k]:.0f}$^\\circ$", (angles[k], s_theta[k]), xytext=(150, s_theta[k] * 0.97), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", lw=0.5))
ax = fig.add_subplot(gs[1, 3:6])
ax.imshow(M, cmap="magma", extent=[-32, 32, 32, -32]); ax.set_xticks([-32, 0, 32]); ax.set_yticks([-32, 0, 32])
ax.set_xlabel("$t_x$ (latent px)", fontsize=8); ax.set_ylabel("$t_y$", fontsize=8)
ax.set_title("(c) shift map", fontsize=8.5)
print("gamma* =", round(float(s_star), 3))
fig.text(0.0, 0.985, "(a)", fontsize=8.5)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "sync.pdf"); fig.savefig(OUT / "sync.png", dpi=170)
print("wrote", OUT / "sync.pdf")
