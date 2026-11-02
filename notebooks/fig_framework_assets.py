# %% [markdown]
# # Fig. 3 assets -- real intermediate results for the framework figure (GPU)
# Rebuilds the initial noise of one SpecShade-256 image (exp final_fr256, PID) from its seed and payload, and
# renders: noise, spectrum, carrier signs before/after embedding, allocation of the carrier to bits, value
# histograms, the image and its edits, the inverted noise of an edited image, the rotation score curve, the
# translation map, the per-bit votes after synchronization, and the null score distribution with the threshold.
# Usage (GPU node): ASSET_OUT=$FIG_OUT/fw_assets PID=p0014 python notebooks/fig_framework_assets.py

# %%
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.ecc import hex_to_bits  # noqa: E402
from core.fourier_rings import ShiftScaleSync, ring_mask, rotate_latents, zoom_latents  # noqa: E402
from core.sd_backend import SDBackend  # noqa: E402
from models import build_method  # noqa: E402
from utils.io import load_config  # noqa: E402

OUT = Path(os.environ.get("ASSET_OUT", ROOT.parent / "docs" / "manuscripts" / "figs" / "fw_assets"))
OUT.mkdir(parents=True, exist_ok=True)
PID = os.environ.get("PID", "p0014")
RES = ROOT / "results"
BLUE, RED, BG = "#2f6fd6", "#d9534f", "#20242c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})


def save_map(a, name, cmap="gray", vmin=None, vmax=None, px=256):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig)


def save_img(path, name, px=320):
    Image.open(path).convert("RGB").resize((px, px), Image.LANCZOS).save(OUT / f"{name}.png")


# %% embedding side
cfg = load_config(ROOT / "configs/final256.yaml")
be = SDBackend(model_id=cfg["backend"]["model_id"], prefer_fp16=cfg["backend"]["prefer_fp16"], size=cfg["backend"]["size"])
m = build_method("stylering", cfg, be)
gen = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(RES / "final_fr256/stylering/gen.jsonl")}[PID]
payload = hex_to_bits(gen["payload_hex"]); seed = int(gen["gen_seed"])
z = be.sample_noise(seed, latent=64).float()
zw = m.embed(payload, seed).float()
code = np.asarray(m._code(payload)).astype(int)                       # encrypted bits written into the noise
F = torch.fft.fftshift(torch.fft.fft2(z[0, 0])).numpy()
Fw = torch.fft.fftshift(torch.fft.fft2(zw[0, 0])).numpy()

save_map(z[0, 0].numpy(), "noise", vmin=-3, vmax=3)
save_map(zw[0, 0].numpy(), "noise_w", vmin=-3, vmax=3)
save_map(np.log1p(np.abs(F)), "spectrum", cmap="magma")

carrier = np.zeros((64, 64), bool); pos = np.full((64, 64), -1)
for cell, p in zip(m.alloc, m.pos_of):
    if cell.c == 0 and cell.part == 0:
        mk = ring_mask(64, 64, cell.r); mk = mk.numpy() if hasattr(mk, "numpy") else np.asarray(mk)
        carrier |= mk; pos[mk] = p
signs = ListedColormap([RED, BG, BLUE])
before = np.where(carrier, np.sign(F.real), 0); after = np.where(carrier, np.sign(Fw.real), 0)
W = slice(20, 44)                                                                   # central 24x24 block, readable pixels
save_map(before[W, W], "sign_before", cmap=signs, vmin=-1, vmax=1)
save_map(after[W, W], "sign_after", cmap=signs, vmin=-1, vmax=1)
rng = np.random.default_rng(3); hue = rng.permutation(256) / 256.0                  # random hue per bit position
alloc_rgb = np.zeros((64, 64, 3)); alloc_rgb[:] = matplotlib.colors.to_rgb(BG)
alloc_rgb[carrier] = plt.get_cmap("twilight_shifted")(hue[pos[carrier]])[:, :3] * 0.75 + 0.2
save_map(alloc_rgb, "alloc")

fig, ax = plt.subplots(figsize=(1.6, 1.0), dpi=200)
bins = np.linspace(-4, 4, 41)
ax.hist(z.numpy().ravel(), bins=bins, density=True, color="#9aa3b2", alpha=0.9, label="$z$")
ax.hist(zw.numpy().ravel(), bins=bins, density=True, histtype="step", color=BLUE, lw=1.2, label="$z_w$")
xs = np.linspace(-4, 4, 200); ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="k", lw=0.7, ls="--", label="$\\mathcal{N}(0,1)$")
ax.set_yticks([]); ax.set_xticks([-3, 0, 3]); ax.tick_params(labelsize=6, length=2)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.2)
fig.savefig(OUT / "hist.png", dpi=300, bbox_inches="tight", pad_inches=0.01); plt.close(fig)

# %% image and edits
A = RES / "final_fr256/stylering/atk"; G = RES / "geo256/stylering/atk"
save_img(RES / f"final_fr256/stylering/gen/{PID}.png", "image")
edits = {"edit_style": A / f"{PID}__img2img_anime_0.5_97.png", "edit_regen": A / f"{PID}__regen_sd_2_97.png",
         "edit_crop": G / f"{PID}__crop_scale_0.75_97.png", "edit_rot": G / f"{PID}__rotate_30_97.png",
         "edit_mirror": G / f"{PID}__mirror_0_97.png"}
combo = RES / f"combo256/stylering/atk/{PID}__combo_style_mirror_1_97.png"
if combo.exists():
    edits["edit_combo"] = combo
for k, p in edits.items():
    save_img(p, k, px=220)

# %% detection side
ims = [Image.open(edits[k]).convert("RGB") for k in ("edit_crop", "edit_rot")]
Z = be.invert_batch(ims, steps=cfg["backend"]["inv_steps"]).cpu().float()
zc, zr = Z[0:1], Z[1:2]
save_map(zc[0, 0].numpy(), "zhat", vmin=-3, vmax=3)
angles = np.arange(0.0, 360.0, 1.0)
s_theta = m.sync_stat(rotate_latents(zr, list(angles)))
fig, ax = plt.subplots(figsize=(1.7, 0.95), dpi=200)
ax.plot(angles, s_theta, color="#7c4ddb", lw=0.8)
ax.set_xticks([0, 180, 360]); ax.set_yticks([]); ax.tick_params(labelsize=6, length=2)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
k = int(np.argmax(s_theta)); ax.plot([angles[k]], [s_theta[k]], "o", ms=2.5, color=RED)
fig.savefig(OUT / "rot_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.01); plt.close(fig)
s_star, ty, tx, peak, _, zsc = m.sync.search(zc, m.scales)
M = np.fft.fftshift(m.sync.stat_maps(zoom_latents(zc, [s_star]))[0].numpy())
c, h = M.shape[0] // 2, 16 * m.sync.up
save_map(M[c - h:c + h, c - h:c + h], "shift_map", cmap="magma")

res_c, res_r, res_w = m.extract(zc), m.extract(zr), None
y = np.asarray(res_c.llr, float)[:256] / 2.0
agree = y * (2 * code[:256] - 1)
save_map(agree.reshape(16, 16), "votes", cmap="RdBu", vmin=-0.8, vmax=0.8)

null_file = RES / "geo256/stylering/detect_null_no_ecc.jsonl"
if not null_file.exists():
    null_file = RES / "final_fr256/stylering/detect_null_no_ecc.jsonl"
null = np.array([json.loads(l)["score"] for l in open(null_file)])
tau = float(np.quantile(null, 0.99))
fig, ax = plt.subplots(figsize=(1.8, 1.0), dpi=200)
ax.hist(null, bins=30, color="#9aa3b2", density=True)
ax.axvline(tau, color="k", lw=0.8, ls="--"); ax.text(tau, ax.get_ylim()[1] * 0.92, " $\\tau$", fontsize=6.5)
for sc, col in ((res_c.score, BLUE), (res_r.score, "#7c4ddb")):
    ax.axvline(sc, color=col, lw=1.2)
ax.set_yticks([]); ax.tick_params(labelsize=6, length=2); ax.set_xlim(0.05, max(0.45, res_c.score + 0.05))
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.savefig(OUT / "scores.png", dpi=300, bbox_inches="tight", pad_inches=0.01); plt.close(fig)

# %% extra assets for the v3 framework figure (fig_framework.py): per-channel latents, aligned noise, key maps, raw curves
for c in range(4):
    save_map(z[0, c].numpy(), f"noise_c{c}", vmin=-3, vmax=3)
    save_map(zw[0, c].numpy(), f"noise_w_c{c}", vmin=-3, vmax=3)
    save_map(zc[0, c].numpy(), f"zhat_c{c}", vmin=-3, vmax=3)
z_al = ShiftScaleSync.apply(zc, s_star, ty, tx)
for c in range(4):
    save_map(z_al[0, c].numpy(), f"zhat_al_c{c}", vmin=-3, vmax=3)
bvec = 2 * code - 1                                                   # encrypted payload as +-1
chips = np.where(carrier, np.sign(Fw.real) * bvec[np.clip(pos, 0, None)], 0)  # chi = sign / b on the real-part carrier
save_map(chips[W, W], "chips", cmap=ListedColormap(["#3a3a3a", "#f4f1e8", "#d4a017"]), vmin=-1, vmax=1)
save_map(np.where(carrier, 1, 0)[W, W] * np.sign(Fw.real)[W, W], "signs_new", cmap=signs, vmin=-1, vmax=1)
save_map(payload[:256].reshape(16, 16), "payload", cmap=ListedColormap(["#f4f1e8", "#2b2b2b"]), vmin=0, vmax=1)
save_map(np.log1p(np.abs(F))[W, W], "spectrum_zoom", cmap="magma")
hz, edges = np.histogram(z.numpy().ravel(), bins=np.linspace(-4, 4, 41), density=True)
hw, _ = np.histogram(zw.numpy().ravel(), bins=edges, density=True)
mc, mh = M.shape[0] // 2, 16 * m.sync.up
Mc = M[mc - mh:mc + mh, mc - mh:mc + mh]                            # same window as shift_map.png
pk = np.unravel_index(int(np.argmax(Mc)), Mc.shape)
np.savez(OUT / "raw.npz", angles=angles, s_theta=np.asarray(s_theta), k=k, null=null, tau=tau,
         score_crop=res_c.score, score_rot=res_r.score, hist_edges=edges, hist_z=hz, hist_zw=hw, agree=agree,
         shift_peak=np.asarray(pk), shift_size=np.asarray(Mc.shape), crop_s=s_star, crop_z=zsc,
         acc_crop=float((res_c.bits == payload).mean()))

info = {"pid": PID, "theta_peak": float(angles[k]), "crop_s": s_star, "crop_t": [ty, tx], "crop_z": zsc,
        "score_crop": res_c.score, "score_rot": res_r.score, "tau": tau, "n_null": int(len(null)),
        "acc_crop": float((res_c.bits == payload).mean()), "acc_rot": float((res_r.bits == payload).mean()),
        "edits": sorted(edits)}
json.dump(info, open(OUT / "info.json", "w"), indent=1)
print(json.dumps(info))
