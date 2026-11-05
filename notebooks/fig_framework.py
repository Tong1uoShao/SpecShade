# %% [markdown]
# # Fig. 2 -- framework figure (v6): layout of v3/v5 (embedding | attacks | detection with four stages) in the visual
# style of common watermarking framework figures: light-gray canvas, coloured dashed module boxes, pastel fills, drop
# shadows, 3-D slab drawings for the diffusion U-Net, bit strings for payloads, and a handwritten italic label font
# (Comic Neue, SIL OFL; math in STIX). Real intermediate results of one image come from fig_framework_assets.py.
# Usage (GPU node, CPU only): ASSET_OUT=.../figs/fw_assets FONT_DIR=.../env/fonts FIG_OUT=.../figs python notebooks/fig_framework.py

# %%
import colorsys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors, font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
A = Path(os.environ.get("ASSET_OUT", ROOT.parent / "docs" / "manuscripts" / "figs" / "fw_assets"))
OUT = Path(os.environ.get("FIG_OUT", ROOT.parent / "docs" / "manuscripts" / "figs"))
FONTS = Path(os.environ.get("FONT_DIR", "fonts"))
RAW = np.load(A / "raw.npz")
for f in FONTS.glob("ComicNeue-*.ttf"):
    font_manager.fontManager.addfont(str(f))

# labels are set in italic per text() call; a global italic style would also turn the STIX "rm" math font italic,
# which lacks operators such as \odot, \oplus, \cdots and \geq
plt.rcParams.update({"font.family": "Comic Neue", "mathtext.fontset": "stix",
                     "font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.5})
INK = "#222222"
C_EMB, C_DET, C_ATK, C_ATK_F = "#E03A3A", "#2F7CC4", "#3E9CC2", "#DCEFF7"
C_KEY, C_KEY_F = "#C69214", "#FFF5D9"
STAGE = [("Inversion", "#7A7A7A", "#F4F4F4"), ("Synchronization", "#4E9E5F", "#E6F3E6"),
         ("Voting", "#8C6BC2", "#EFEAF8"), ("Decision", "#D05A5A", "#F9E4E4")]
FL, FT = 7.5, 8.8                                  # label / title font sizes (pt)

W, H = 716, 336                                   # canvas units (1 unit = 0.01 in)
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


# %% drawing helpers
def shade(c, f):
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(h, max(0, min(1, l * f)), s)


def rbox(x0, y0, w, h, fc="none", ec=INK, lw=0.8, ls="-", r=4, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=FL, color=INK, ha="center", va="center", style="italic", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=8, fontstyle=style, **kw)


def shadow_rect(x0, y0, w, h, z):
    ax.add_patch(Rectangle((x0 + 1.6, y0 - 1.6), w, h, fc="#9A9A9A", ec="none", alpha=0.5, zorder=z - 0.05))


def img(name, cx, cy, s, interp="nearest", z=3):
    shadow_rect(cx - s / 2, cy - s / 2, s, s, z)
    ax.imshow(plt.imread(A / f"{name}.png"), extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2), interpolation=interp, zorder=z)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec="#3A3A3A", lw=0.5, zorder=z + 0.02))


def stack(prefix, cx, cy, s, off=3.5, z=3):
    """Four latent channels as a tensor; back channels shifted up-right, each with a thin shadow."""
    for i in (3, 2, 1, 0):
        x0, y0 = cx - s / 2 + (i - 1.5) * off, cy - s / 2 + (i - 1.5) * off
        zz = z + (3 - i) * 0.2
        shadow_rect(x0, y0, s, s, zz)
        ax.imshow(plt.imread(A / f"{prefix}_c{i}.png"), extent=(x0, x0 + s, y0, y0 + s), interpolation="nearest", zorder=zz)
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec="#3A3A3A", lw=0.45, zorder=zz + 0.02))


def arrow(p0, p1, color=INK, lw=0.9, ls="-", conn="arc3,rad=0", ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=conn, shrinkA=0, shrinkB=0, zorder=6))


def slabs(x0, cy, heights, cols, t=4.6, gap=1.6, d=4.5, z=4):
    """U-Net drawn as 3-D slabs (front, top and side faces), left to right."""
    x = x0
    for k, (h, c) in enumerate(zip(heights, cols)):
        zz = z + k * 0.01
        y0, y1 = cy - h / 2, cy + h / 2
        ax.add_patch(Polygon([(x, y1), (x + d, y1 + d), (x + t + d, y1 + d), (x + t, y1)], closed=True,
                             fc=shade(c, 1.12), ec=shade(c, 0.7), lw=0.4, zorder=zz))
        ax.add_patch(Polygon([(x + t, y0), (x + t + d, y0 + d), (x + t + d, y1 + d), (x + t, y1)], closed=True,
                             fc=shade(c, 0.88), ec=shade(c, 0.7), lw=0.4, zorder=zz))
        ax.add_patch(Rectangle((x, y0), t, h, fc=c, ec=shade(c, 0.7), lw=0.4, zorder=zz))
        x += t + gap
    return x - gap + d


def unet(cx, cy, scale=1.0, palette=("#F7B58A", "#CFE5F6", "#E8817F")):
    hs = [h * scale for h in (42, 33, 24, 24, 33, 42)]
    cols = [palette[0], palette[1], palette[2], palette[2], palette[1], palette[0]]
    width = 6 * 4.6 + 5 * 1.6 + 4.5
    return slabs(cx - width / 2, cy - 2, hs, cols)


def bits(cx, cy, s=r"$\mathbf{1011\cdots01}$", w=56, h=17):
    shadow_rect(cx - w / 2, cy - h / 2, w, h, 4)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec="#3A3A3A", lw=0.6, zorder=4))
    ax.text(cx, cy, s, fontsize=8, color=INK, ha="center", va="center", zorder=8)


def key_icon(cx, cy, s=1.0, color=C_KEY):
    ax.add_patch(Circle((cx - 7 * s, cy), 4.2 * s, fc="white", ec=color, lw=1.3, zorder=8))
    ax.add_patch(Rectangle((cx - 2.8 * s, cy - 0.9 * s), 12 * s, 1.8 * s, fc=color, ec=color, lw=0, zorder=8))
    for dx in (5.5, 8.2):
        ax.add_patch(Rectangle((cx + dx * s, cy - 4 * s), 1.6 * s, 3.2 * s, fc=color, ec=color, lw=0, zorder=8))


def inset(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.set_yticks([]); a.set_xticks([]); a.patch.set_alpha(0)
    return a


# %% canvas and modules
ax.add_patch(FancyBboxPatch((6, 2), W - 10, H - 8, boxstyle="round,pad=0,rounding_size=6", fc="#BDBDBD", ec="none", alpha=0.6, zorder=-2))
ax.add_patch(FancyBboxPatch((3, 5), W - 10, H - 8, boxstyle="round,pad=0,rounding_size=6", fc="#F4F4F4", ec="#CFCFCF", lw=0.8, zorder=-1))
rbox(12, 13, 286, 312, ec=C_EMB, lw=1.0, ls=(0, (5, 3)), r=6, z=0)
text(155, 313, "(a) Watermark embedding", size=FT, color=C_EMB)
rbox(306, 13, 72, 312, fc=C_ATK_F, ec=C_ATK, lw=1.0, ls=(0, (4, 2.5)), r=6, z=0)
text(342, 313, "(b) Attacks", size=FT, color=C_ATK)
rbox(386, 13, 321, 312, ec=C_DET, lw=1.0, ls=(0, (5, 3)), r=6, z=0)
text(546, 313, "(c) Blind detection", size=FT, color=C_DET)

# %% (a) embedding
C1, C2, C3, C4 = 54, 126, 194, 244
R1, R2, R3 = 252, 166, 76
stack("noise", C1, R1, 38)
text(C1, R1 - 33, r"initial noise $\mathbf{z}_T$")
arrow((C1 + 25, R1), (C2 - 27, R1)); text((C1 + C2) / 2 + 1, R1 + 8, "FFT")
img("spectrum", C2, R1, 46, interp="bilinear")
for r in (8, 16, 24, 31):
    ax.add_patch(Circle((C2, R1), r * 46 / 64, fill=False, ec="white", lw=0.45, alpha=0.75, zorder=4))
text(C2, R1 - 33, r"spectrum $\mathbf{F}$")
ax.add_patch(Circle((C3, R1), 7, fc="white", ec=INK, lw=0.9, zorder=7)); text(C3, R1 - 0.4, r"$\odot$", size=11)
arrow((C2 + 25, R1), (C3 - 8, R1)); text((C2 + C3) / 2 + 2, R1 + 8, r"$|\mathbf{F}|$")
arrow((C3 + 8, R1), (C4 - 20, R1)); text(C3 + 20, R1 + 8, "IFFT")
stack("noise_w", C4 + 8, R1, 34)
text(C4 + 8, R1 + 41, "watermarked"); text(C4 + 8, R1 + 31, r"noise $\mathbf{z}_T^{w}$")
bits(C1, R2)
text(C1, R2 - 18, r"payload $\mathbf{u}$")
arrow((C1 + 29, R2), (C3 - 26, R2))
text(C2 - 2, R2 + 8, r"$\oplus$ key", color=C_KEY); text(C2 - 2, R2 - 8, r"$\mathbf{b}$", size=8.5)
img("signs_new", C3, R2, 46)
text(C3 + 6, (R1 + R2) / 2 + 3, r"signs $\boldsymbol{\eta}$", ha="left")
arrow((C3, R2 + 25), (C3, R1 - 8))
arrow((C4 + 8, R1 - 26), (C4 + 8, R2 + 26))
unet(C4 + 8, R2, scale=0.95)
text(C4 + 8, R2 - 30, "diffusion model")
arrow((C4 + 8, R2 - 38), (C4 + 8, R3 + 25))
img("alloc", C1, R3, 38)
text(C1, R3 - 29, r"allocation $\psi$")
rbox(C2 - 25, R3 - 15, 50, 30, fc=C_KEY_F, ec=C_KEY, lw=0.9, ls=(0, (3, 2)), r=4, z=5)
key_icon(C2 - 1, R3 + 5, 0.95)
text(C2, R3 - 7, r"key $K$", color=C_KEY)
img("chips", C3, R3, 38)
text(C3, R3 - 29, r"chips $\boldsymbol{\chi}$")
arrow((C2 - 26, R3), (C1 + 21, R3), color=C_KEY)
arrow((C2 + 26, R3), (C3 - 21, R3), color=C_KEY)
arrow((C2 - 2, R3 + 16), (C2 - 2, R2 - 13), color=C_KEY, ls=(0, (2.5, 1.8)), lw=0.8)
arrow((C1 + 14, R3 + 21), (C2 - 16, R2 - 3), color=C_KEY, conn="arc3,rad=-0.2")
arrow((C3, R3 + 21), (C3, R2 - 25), color=C_KEY)
img("image", C4 + 8, R3, 44, interp="bilinear")
text(C4 + 8, R3 - 31, "watermarked"); text(C4 + 8, R3 - 41, "image")

# %% (b) attacks
AX = 342
for (n, t), y in zip([("edit_regen", "regeneration"), ("edit_style", "style transfer"), ("edit_crop", "crop"), ("edit_rot", "rotation")],
                     [254, 186, 118, 50]):
    img(n, AX, y, 42, interp="bilinear")
    text(AX, y - 28, t, size=7.2)
arrow((C4 + 33, R3), (306, R3), color=C_ATK, lw=1.1)

# %% (c) detection: four stages
SX = [(392, 452), (458, 562), (568, 634), (640, 701)]
SY0, SY1 = 58, 298
MID = 188
for (x0, x1), (t, ec, fc) in zip(SX, STAGE):
    rbox(x0, SY0, x1 - x0, SY1 - SY0, fc=fc, ec=ec, lw=0.8, ls=(0, (3, 2)), r=4, z=1)
    text((x0 + x1) / 2, SY1 - 10, t, color=ec)
for (a0, a1), (b0, b1) in zip(SX[:-1], SX[1:]):
    arrow((a1 + 0.3, MID), (b0 - 0.3, MID), lw=1.0)
arrow((378, MID), (SX[0][0], MID), color=C_ATK, lw=1.1)

# (1) inversion
cx = (SX[0][0] + SX[0][1]) / 2
img("edit_crop", cx, 252, 40, interp="bilinear")
arrow((cx, 230), (cx, 214))
unet(cx, 192, scale=0.72, palette=("#8FD3C6", "#D7D2EF", "#76B7E5"))
text(cx, 166, "DDIM", size=7.2)
arrow((cx, 159), (cx, 143))
stack("zhat", cx, 120, 30, off=3)
text(cx, 92, r"$\hat{\mathbf{z}}_T$"); text(cx, 80, "misaligned", size=7.2)

# (2) synchronization
x0, x1 = SX[1]; cx = (x0 + x1) / 2
ra = inset(x0 + 10, 240, x1 - x0 - 20, 32)
ang, st = RAW["angles"], RAW["s_theta"]; k = int(RAW["k"])
ra.plot(ang, st, color=STAGE[1][1], lw=0.8); ra.plot([ang[k]], [st[k]], "o", ms=3, color="#C0392B"); ra.set_xlim(0, 360)
text(cx, 232, "rotation search")
img("shift_map", cx, 192, 40, interp="bilinear")
sz = RAW["shift_size"]; pk = RAW["shift_peak"]
px, py = cx - 20 + (pk[1] + 0.5) / sz[1] * 40, 172 + 40 - (pk[0] + 0.5) / sz[0] * 40
ax.add_patch(Circle((px, py), 3.4, fill=False, ec="white", lw=0.8, zorder=6))
text(cx, 163, "scale + shift search")
arrow((cx, 155), (cx, 137))
stack("zhat_al", cx, 112, 30, off=3)
text(cx, 84, r"aligned $\hat{\mathbf{z}}_T$")

# (3) voting
x0, x1 = SX[2]; cx = (x0 + x1) / 2
img("votes", cx, 234, 48)
text(cx, 201, r"votes $y_j$")
arrow((cx, 192), (cx, 174))
text(cx, 161, r"$\hat{b}_j=\mathrm{sign}(y_j)$", size=8)
bits(cx, 120, w=54)
text(cx, 102, r"bits $\hat{\mathbf{b}}$")

# (4) decision
x0, x1 = SX[3]; cx = (x0 + x1) / 2
da = inset(x0 + 7, 216, x1 - x0 - 14, 52)
null = RAW["null"]; tau = float(RAW["tau"]); sc = float(RAW["score_crop"])
da.hist(null, bins=28, color="#BDBDBD", density=True)
da.axvline(tau, color=INK, lw=0.7, ls="--"); da.axvline(sc, color=STAGE[3][1], lw=1.2)
da.set_xlim(0.08, max(0.42, sc + 0.04))
yl = da.get_ylim()[1]
da.text(tau, yl * 0.97, r" $\tau$", fontsize=8, va="top"); da.text(sc, yl * 0.97, r" $S$", fontsize=8, color=STAGE[3][1], va="top")
text(cx, 205, r"score $S$ vs. $\tau$")
text(cx, 184, r"$S\geq\tau$", color=STAGE[3][1]); text(cx, 172, "detected", color=STAGE[3][1])
arrow((cx, 162), (cx, 132))
bits(cx, 120, w=54)
text(cx, 102, r"payload $\hat{\mathbf{u}}$")

# key feeding synchronization and voting
ky = 36
rbox(SX[0][0], ky - 11, 56, 22, fc=C_KEY_F, ec=C_KEY, lw=0.8, ls=(0, (3, 2)), r=4, z=5)
key_icon(SX[0][0] + 13, ky, 0.85)
text(SX[0][0] + 23, ky, r"key $K$", color=C_KEY, ha="left")
ax.plot([SX[0][0] + 56, (SX[2][0] + SX[2][1]) / 2], [ky, ky], color=C_KEY, lw=0.8, ls=(0, (3, 2)), zorder=4)
arrow(((SX[1][0] + SX[1][1]) / 2, ky), ((SX[1][0] + SX[1][1]) / 2, SY0), color=C_KEY, lw=0.8)
arrow(((SX[2][0] + SX[2][1]) / 2, ky), ((SX[2][0] + SX[2][1]) / 2, SY0), color=C_KEY, lw=0.8)

fig.savefig(OUT / "framework.pdf")
fig.savefig(OUT / "framework.png", dpi=220)
print("wrote", OUT / "framework.pdf")
