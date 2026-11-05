# %% [markdown]
# # Paper figures (SpecShade, IEEE Transactions draft)
# Every figure reads the result tables under code/results/<exp>/tables and writes a PDF to $FIG_OUT
# (default docs/manuscripts/figs). Run on the GPU node (needs matplotlib/pandas):
#   FIG_OUT=$FIG_OUT python notebooks/figures.py [law croplimit ablation ...]
# Cells are marked with "# %%" (VS Code / Jupyter); figures.ipynb is generated from this file.

# %%
import os
import sys
from math import erf, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
RES = ROOT / "results"
OUT = Path(os.environ.get("FIG_OUT", ROOT.parent / "docs" / "manuscripts" / "figs"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = {"ours": "#1f5fbf", "gs": "#7a7a7a", "gm": "#b54708", "accent": "#2e8b57", "bad": "#c0392b"}
ONE_COL, TWO_COL = 3.5, 7.16      # IEEE column / page width in inches


def table(exp: str, method: str, variant: str | None = None) -> pd.DataFrame:
    for name in ((f"main_{variant}.csv",) if variant else ("main_no_ecc.csv", "main_soft.csv")):
        p = RES / exp / "tables" / name
        if p.exists():
            df = pd.read_csv(p)
            df = df[df["method"] == method]
            if len(df):
                return df.set_index("cell")
    raise FileNotFoundError(f"{exp}/{method}")


def _first(*exps, variant=None):
    """The first experiment whose tables exist (unified-detector re-runs first, the original runs as fallback)."""
    for e in exps:
        if (RES / e / "tables" / (f"main_{variant}.csv" if variant else "main_no_ecc.csv")).exists():
            return e
    return exps[-1]


def fam(df: pd.DataFrame, attack: str, strength: str, col: str) -> float:
    """Mean over the style variants of an attack family (e.g. img2img_anime/comic/watercolor@0.5)."""
    rows = [c for c in df.index if (c.startswith(attack + "_") or c.split("@")[0] == attack) and c.endswith("@" + strength)]
    return float(df.loc[rows, col].mean()) if rows else float("nan")


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# %% [markdown]
# ## Fig. 2 -- votes x survival law (Sec. III-B)
# Per-element sign survival p of Gaussian Shading measured by src/proto_gs_survival.py (20 images per cell,
# dev_log 2026-09-13); predicted bit accuracy Phi((2p-1) sqrt(n)) vs measured bit accuracy of GS-16 (n=1024,
# exp final) and GS-256 (n=64, exp final_gs256).

# %%
SURVIVAL = {  # cell -> per-element sign survival p (measured on GS images, anime style for img2img)
    "img2img_anime@0.3": 0.549, "img2img_anime@0.5": 0.527, "img2img_anime@0.7": 0.511,
    "regen_sd@1.0": 0.550, "regen_sd@2.0": 0.527, "regen_sd@4.0": 0.515,
}


def fig_law():
    gs16, gs256 = table("final", "gaussian_shading"), table("final_gs256", "gaussian_shading")
    fig, (a, b) = plt.subplots(1, 2, figsize=(ONE_COL * 2, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
    errs = []
    for df, n, mk, lab in ((gs16, 1024, "o", "GS-16 ($n{=}1024$)"), (gs256, 64, "s", "GS-256 ($n{=}64$)")):
        xs, ys = [], []
        for cell, p in SURVIVAL.items():
            if cell in df.index:
                xs.append(phi((2 * p - 1) * sqrt(n)))
                ys.append(float(df.loc[cell, "bit_acc_all"]))
        errs += [abs(x - y) for x, y in zip(xs, ys)]
        a.scatter(xs, ys, marker=mk, s=16, label=lab, color=COL["gs"] if n == 64 else COL["ours"], zorder=3)
    a.plot([0.5, 1], [0.5, 1], ls="--", lw=0.8, color="k")
    a.set_xlim(0.5, 1.0); a.set_ylim(0.5, 1.0)
    a.set_xlabel(r"predicted bit acc. $\Phi((2p-1)\sqrt{n})$"); a.set_ylabel("measured bit accuracy")
    a.set_title(f"(a) prediction vs. measurement (max err {max(errs):.3f})")
    a.legend(loc="lower right", frameon=False)
    n = np.logspace(0, 4.3, 200)
    for p, ls in ((0.549, "-"), (0.527, "--"), (0.511, ":")):
        b.plot(n, [phi((2 * p - 1) * sqrt(v)) for v in n], ls=ls, color="k", label=f"$p={p}$")
    for v, txt in ((64, "256 bit\n(64 votes)"), (1024, "16 bit\n(1024 votes)")):
        b.axvline(v, color=COL["bad"] if v == 64 else COL["accent"], lw=0.8)
        b.text(v * 1.08, 0.52, txt, fontsize=6.5, color=COL["bad"] if v == 64 else COL["accent"])
    b.set_xscale("log"); b.set_ylim(0.5, 1.0); b.set_xlabel("votes per bit $n$"); b.set_ylabel("predicted bit accuracy")
    b.set_title("(b) survival after img2img @0.3 / 0.5 / 0.7")
    b.legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "law.pdf"); plt.close(fig)
    print("law: max |pred-meas| =", round(max(errs), 4))


# %% [markdown]
# ## Fig. 5 -- crop limit (Sec. V-C)
# Random-position crop of area fraction a, resized back; TPR@1%FPR and bit accuracy vs a, for SpecShade at
# 256 and 16 bits (exps fr256_cropmin, fr2_cropmin2: 50 images x 2 seeds).

# %%
def fig_croplimit():
    ours256, ours16 = table("fr256_cropmin", "stylering"), table("fr2_cropmin2", "stylering")
    areas = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ONE_COL * 2, 2.0))
    for df, lab, c, mk in ((ours256, "SpecShade-256", COL["ours"], "o"), (ours16, "SpecShade-16", COL["accent"], "^")):
        xs = [x for x in areas if f"crop_scale@{x}" in df.index]
        a.plot(xs, [df.loc[f"crop_scale@{x}", "tpr_1fpr"] for x in xs], marker=mk, ms=3.5, color=c, label=lab)
        b.plot(xs, [df.loc[f"crop_scale@{x}", "bit_acc_all"] for x in xs], marker=mk, ms=3.5, color=c, label=lab)
    for ax, yl in ((a, "TPR@1%FPR"), (b, "bit accuracy")):
        ax.set_xlabel("retained area fraction $a$"); ax.set_ylabel(yl); ax.invert_xaxis(); ax.grid(alpha=0.25, lw=0.5)
    a.set_ylim(-0.03, 1.03); b.set_ylim(0.45, 1.01)
    a.set_title("(a) detection"); b.set_title("(b) decoding")
    b.legend(loc="lower left", frameon=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "croplimit.pdf"); plt.close(fig)


# %% [markdown]
# ## Fig. 9 -- ablation of the embedding design (Sec. IV-F)
# (a) 2x2 at an equal sign budget (16-bit payload, BCH(63,16) codeword repeated over the full budget):
#     spatial/Fourier carrier x hard/magnitude-weighted votes; bit accuracy under img2img@0.5 and regen x2
#     (exps full_spatial, full_spatial_soft, full_chip_hard, full_chip; 50 images x 2 seeds).
# (b) coherence length: sector chips per ring K (63 cells, 16 bit), TPR@1%FPR under img2img@0.5 vs the share
#     of images with visible artifacts (exps main=K1, abl_sec4/8/16/32, abl_sec0=per-pair chips).

# %%
def fig_ablation():
    # (a): real embeddings, 256 bits each, generated / attacked / inverted separately (15 key attacks): GS-256 with the hard
    #   majority (final_gs256q) and with a magnitude-weighted vote (final_gs256s), SpecShade-256 with hard cell reads
    #   (final256h) and with magnitude weighting (final256). (b): sign survival per ring of the same inverted noise of 100
    #   SpecShade-256 images (src/proto_freq_survival.py, results/freq_survival2).
    import json
    S = json.load(open(RES / "freq_survival2" / "summary.json"))["attacks"]
    cells = [(["regen_sd@2.0", "regen_sd@2"], "regen.\n$\\times$2"), (["regen_sd@4.0", "regen_sd@4"], "regen.\n$\\times$4"),
             (["img2img_anime@0.3"], "img2img\n0.3"), (["img2img_anime@0.5"], "img2img\n0.5")]
    variants = [(_opt("final_gs256q", "gaussian_shading", "soft"), "spatial, hard"), (_opt("final_gs256s", "gaussian_shading", "soft"), "spatial, soft"),
                (_opt("final256h", "stylering", "no_ecc"), "Fourier, hard"), (_opt(_first("final256ua", "final256"), "stylering", "no_ecc"), "Fourier, soft")]
    fig = plt.figure(figsize=(TWO_COL, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.15, 0.85, 1.2], wspace=0.4, left=0.05, right=0.995, bottom=0.2, top=0.88)
    a, b, c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    w = 0.2
    for i, (df, lab) in enumerate(variants):
        ys = [_pick([df], ks, "tpr_1fpr") for ks, _ in cells]
        shade = ["#c7c7c7", "#8f8f8f", "#8fb3e6", COL["ours"]][i]
        a.bar(np.arange(len(cells)) + (i - 1.5) * w, ys, w, color=shade, label=lab, edgecolor="k", lw=0.3)
    a.set_xticks(range(len(cells))); a.set_xticklabels([lab for _, lab in cells], fontsize=7); a.set_xlim(-0.6, len(cells) - 0.4)
    a.set_ylim(0, 1.3); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a.set_ylabel("TPR@1%FPR (256 bits)")
    a.set_title("(a) embedding domain $\\times$ vote type")
    a.legend(loc="upper right", frameon=False, ncol=2, fontsize=7, handlelength=1.0, labelspacing=0.3, columnspacing=0.8)
    rings = np.arange(1, 45)
    for k, lab, col in (("regen_sd:2.0", "regen. $\\times$2", COL["ours"]), ("img2img_anime:0.5", "img2img 0.5", COL["accent"])):
        b.plot(rings, S[k]["hard"], color=col, lw=1.0, label=lab)
        b.plot(rings, S[k]["weighted"], color=col, lw=1.0, ls="--")
    sp = S["regen_sd:2.0"]["spatial_hard"]
    b.axhline(sp, color="k", lw=0.6, ls=":")
    b.text(2, sp - 0.012, "spatial, hard", ha="left", va="top", fontsize=7)
    b.set_xlabel("ring $r$"); b.set_ylabel("sign survival"); b.set_ylim(0.48, 0.8); b.set_xlim(1, 44)
    b.set_title("(b) survival per ring"); b.legend(frameon=False, fontsize=7, handlelength=1.4, loc="upper right", labelspacing=0.3)
    b.grid(alpha=0.25, lw=0.5)
    ks = [("main", "1"), ("abl_sec4", "4"), ("abl_sec8", "8"), ("abl_sec16", "16"), ("abl_sec32", "32"), ("abl_sec0", "pair")]
    tpr = [fam(table(e, "stylering"), "img2img", "0.5", "tpr_1fpr") for e, _ in ks]
    c.plot(range(len(ks)), tpr, marker="o", ms=3.5, color=COL["ours"])
    for i, t in enumerate(tpr):
        c.text(i, t + 0.02, f"{t:.2f}", ha="center", fontsize=7)
    c.set_xticks(range(len(ks))); c.set_xticklabels([k for _, k in ks])
    c.set_xlabel("sectors per ring $n_s$"); c.set_ylabel("TPR@1%FPR, img2img 0.5")
    c.set_ylim(0, 0.7); c.set_title("(c) coherent chips: robust")
    c.grid(alpha=0.25, lw=0.5)
    # (d) ... but visible: top-left quarters of the same prompts generated with each design (16-bit ablation)
    from PIL import Image
    designs = [("main", "$n_s{=}1$"), ("abl_sec8", "$n_s{=}8$"), ("abl_sec0", "per pair")]
    pids = ["p0000", "p0003"]                              # two prompts with flat backgrounds
    t = fig.add_subplot(gs[3]); t.axis("off")
    t.set_title("(d) coherent chips: visible", pad=11)
    sub = gs[3].subgridspec(len(pids), len(designs), wspace=0.04, hspace=0.04)
    for r, pid in enumerate(pids):
        for c, (exp, lab) in enumerate(designs):
            ax = fig.add_subplot(sub[r, c])
            im = Image.open(RES / exp / "stylering" / "gen" / f"{pid}.png").convert("RGB").crop((0, 0, 256, 256))
            ax.imshow(np.asarray(im), interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_anchor("N" if r == 0 else "S")
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(lab, fontsize=7.5, pad=2)
    fig.savefig(OUT / "ablation.pdf"); plt.close(fig)


# %% [markdown]
# ## Fig. -- comparison with the full GaussMarker on the 15 key attacks (replaces the former Table II)
# TPR@1%FPR (bars) and bit accuracy (markers) of SpecShade-256, GS-256 and GaussMarker-256 (trained GNR, official
# single-message protocol). GaussMarker is scored with the stronger of its two score fusions (gm256a / gm256).

# %%
GM_KEYS = [("Regen. $\\times1$", ["regen_sd@1.0"]), ("Regen. $\\times2$", ["regen_sd@2.0"]), ("Regen. $\\times4$", ["regen_sd@4.0"]),
           ("VAE-Cheng", ["vae_cheng@3.0", "vae_cheng@3"]), ("img2img $0.3$", ["img2img_anime@0.3"]),
           ("img2img $0.5$", ["img2img_anime@0.5"]), ("AdaIN $0.6$", ["adain_anime@0.6"]), ("AdaIN $1.0$", ["adain_anime@1.0"]),
           ("IP2P $1.0$", ["ip2p_anime@1.0"]), ("JPEG $25$", ["jpeg@25.0", "jpeg@25"]), ("Blur $r{=}8$", ["blur@8.0", "blur@8"]),
           ("Crop $0.75$", ["crop_scale@0.75"]), ("Rotation $75^\\circ$", ["rotate@75.0", "rotate@75"]),
           ("Mirror", ["mirror@0.0", "mirror@0"]), ("img2img + crop", ["combo_style_crop@1.0", "combo_style_crop@1"])]


def _opt(exp, method, variant=None):
    try:
        return table(exp, method, variant)
    except FileNotFoundError:
        return None


def _pick(dfs, cells, col):
    for df in dfs:
        if df is None:
            continue
        for c in cells:
            if c in df.index:
                return float(df.loc[c, col])
    return float("nan")


def fig_gm():
    ours = [_opt(_first("final256ua", "final256"), "stylering"), _opt(_first("geo256u", "geo256"), "stylering"), _opt(_first("combo256u", "combo256"), "stylering")]
    gs = [_opt("final_gs256q", "gaussian_shading"), _opt("combo_gs256", "gaussian_shading")]
    cand = [(_opt(e, "gaussmarker_full"), e) for e in ("gm256a", "gm256")]
    cand = [(df, e) for df, e in cand if df is not None]
    gm, gm_e = max(cand, key=lambda de: np.nanmean([_pick([de[0]], c, "tpr_1fpr") for _, c in GM_KEYS]))
    import json
    nulls = sorted(json.loads(l)["score"] for l in open(RES / gm_e / "gaussmarker_full/detect_null.jsonl"))
    gm_tau = nulls[int(0.99 * len(nulls)) - 1]       # its 1%-FPR threshold on the bit agreement (as in tables.py)
    # SFW (HSQR, verification protocol: the detector is given the key) and MaXsive (one fixed watermark), re-implemented
    # from the official code (src/models/baseline/sfw.py, maxsive.py); neither carries a comparable 256-bit payload
    sfw, mx = [_opt("final_sfw", "sfw_hsqr", "known")], [_opt("final_maxsive", "maxsive")]
    series = [("\\method-256 (ours)".replace("\\method", "SpecShade"), ours, COL["ours"], "o"), ("GS-256", gs, COL["gs"], "s"),
              ("GaussMarker-256", [gm], COL["gm"], "D"), ("SFW (HSQR)", sfw, COL["accent"], None), ("MaXsive", mx, "#8e44ad", None)]
    x = np.arange(len(GM_KEYS)); w = 0.16
    fig, (a, b) = plt.subplots(2, 1, figsize=(TWO_COL, 2.8), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
    for k, (lab, dfs, c, mk) in enumerate(series):
        tpr = [_pick(dfs, cells, "tpr_1fpr") for _, cells in GM_KEYS]
        a.bar(x + (k - 2) * w, tpr, w, color=c, label=lab, edgecolor="none")
        if mk is not None:
            acc = [_pick(dfs, cells, "bit_acc_all") for _, cells in GM_KEYS]
            b.plot(x + (k - 2) * w, acc, mk, ms=4, color=c, label=lab, ls="none")
    a.set_ylabel("TPR@1%FPR"); a.set_ylim(0, 1.05); a.grid(axis="y", alpha=0.25, lw=0.5)
    a.legend(ncol=5, loc="lower left", frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=7, handlelength=1.2, columnspacing=1.0)
    b.axhline(0.5, color="k", lw=0.6, ls=":"); b.text(-0.45, 0.505, "chance", fontsize=6.5, ha="left", va="bottom")
    b.axhline(gm_tau, color=COL["gm"], lw=0.9, ls="--")
    b.text(-0.45, gm_tau + 0.006, rf"GaussMarker $\tau$ ({gm_tau:.3f})", fontsize=6.5, color=COL["gm"], ha="left", va="bottom")
    b.set_ylabel("bit accuracy"); b.set_ylim(0.45, 1.02); b.grid(axis="y", alpha=0.25, lw=0.5)
    for ax in (a, b):
        for g in (3.5, 8.5, 10.5):                    # regeneration | style transfer | classical | geometric
            ax.axvline(g, color="k", lw=0.4, alpha=0.4)
    b.set_xticks(x); b.set_xticklabels([l for l, _ in GM_KEYS], fontsize=6.5, rotation=30, ha="right", rotation_mode="anchor")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(OUT / "gm.pdf"); plt.close(fig)


# %% [markdown]
# ## Fig. -- geometric robustness (replaces the former Table III and the crop-limit figure)
# (a) rotation angle: TPR and bit accuracy of SpecShade-256 (integer angles 15-90 from geo256, half-way angles
#     7.5/22.5/37.5/67.5 from geo256f, hollow markers) and of GS-256 with its original decoder (geo_gs256).
# (b,c) crop area fraction a: SpecShade-256 (a >= 0.5 from geo256, 100 images x 2 seeds, filled; a < 0.5 from the
#     sweep fr256_cropmin, 50 images x 2 seeds, hollow), SpecShade-16 (sweep fr2_cropmin2) and GS-256 (geo_gs256).

# %%
def fig_geo():
    geo, geof, gsg = table("geo256", "stylering"), table("geo256f", "stylering"), table("geo_gs256", "gaussian_shading")
    sw256, sw16 = table("fr256_cropmin", "stylering"), table("fr2_cropmin2", "stylering")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TWO_COL, 1.75))
    ints, halves = [15, 30, 45, 75, 90], [7.5, 22.5, 37.5, 67.5]
    allang = sorted(ints + halves)
    def rot(df, ang, col):
        for key in (f"rotate@{float(ang)}", f"rotate@{ang}"):
            if key in df.index:
                return float(df.loc[key, col])
        return float("nan")
    for col, ls, lab in (("tpr_1fpr", "-", "TPR"), ("bit_acc_all", "--", "Acc")):
        ys = [rot(geo if ang in ints else geof, ang, col) for ang in allang]
        a.plot(allang, ys, ls=ls, color=COL["ours"], lw=1.0, label=f"SpecShade-256 {lab}")
        a.plot(ints, [rot(geo, g, col) for g in ints], "o", ms=3.5, color=COL["ours"])
        a.plot(halves, [rot(geof, g, col) for g in halves], "o", ms=3.5, mfc="white", color=COL["ours"])
        a.plot(ints, [rot(gsg, g, col) for g in ints], ls=ls, marker="s", ms=3, color=COL["gs"], lw=1.0, label=f"GS-256 {lab}")
    a.set_xlabel("rotation angle (deg)"); a.set_ylim(-0.03, 1.03); a.set_xticks([0, 15, 30, 45, 60, 75, 90])
    a.set_title("(a) rotation"); a.grid(alpha=0.25, lw=0.5); a.legend(loc="center right", frameon=False, fontsize=6)
    big, small = [0.9, 0.75, 0.5], [0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    def crop(df, x, col):
        k = f"crop_scale@{x}"
        return float(df.loc[k, col]) if k in df.index else float("nan")
    for ax, col, yl in ((b, "tpr_1fpr", "TPR@1%FPR"), (c, "bit_acc_all", "bit accuracy")):
        ax.plot(big + small, [crop(geo, x, col) for x in big] + [crop(sw256, x, col) for x in small], color=COL["ours"], lw=1.0,
                label="SpecShade-256")
        ax.plot(big, [crop(geo, x, col) for x in big], "o", ms=3.5, color=COL["ours"])
        ax.plot(small, [crop(sw256, x, col) for x in small], "o", ms=3.5, mfc="white", color=COL["ours"])
        xs16 = [0.5] + small
        ax.plot(xs16, [crop(sw16, x, col) for x in xs16], marker="^", ms=3.5, mfc="white", color=COL["accent"], lw=1.0, label="SpecShade-16")
        ax.plot(big, [crop(gsg, x, col) for x in big], marker="s", ms=3, color=COL["gs"], lw=1.0, label="GS-256")
        ax.set_xlabel("retained area fraction $a$"); ax.set_ylabel(yl); ax.invert_xaxis(); ax.grid(alpha=0.25, lw=0.5)
    b.set_ylim(-0.03, 1.03); c.set_ylim(0.45, 1.01)
    b.set_title("(b) crop: detection"); c.set_title("(c) crop: decoding")
    b.legend(loc="center left", frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "geo.pdf"); plt.close(fig)



# %% [markdown]
# ## Fig. -- capacity sweep (reduced protocol: 20 prompts x 1 seed, img2img anime s=0.5)
# Bit accuracy and TPR@1%FPR of SpecShade and Gaussian Shading at 16-256 bits (results/cap_summary.json from
# notebooks/cap_summary.py); dashed: the votes x survival law Phi((2p-1) sqrt(n)) with the measured GS survival p=0.527.

# %%
def fig_capacity():
    import json
    d = json.load(open(RES / "cap_summary.json"))
    Ls = [16, 32, 64, 128, 256]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ONE_COL, 1.75))
    for key, lab, c, mk in (("ours", "SpecShade", COL["ours"], "o"), ("gs", "GS", COL["gs"], "s")):
        pts = [(i, d[key][str(L)]) for i, L in enumerate(Ls) if d[key].get(str(L))]
        xs = [i for i, _ in pts]
        a.errorbar(xs, [v["acc"] for _, v in pts], yerr=[v["acc_sem"] for _, v in pts], marker=mk, ms=3.5, color=c, lw=1.0, capsize=1.5, label=lab)
        b.plot(xs, [v["tpr"] for _, v in pts], marker=mk, ms=3.5, color=c, lw=1.0, label=lab)
    law = [phi((2 * 0.527 - 1) * sqrt(16384 / L)) for L in Ls]
    a.plot(range(len(Ls)), law, ls="--", color="k", lw=0.7, label="law, $p{=}0.527$")
    for ax, yl in ((a, "bit accuracy"), (b, "TPR@1%FPR")):
        ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls); ax.set_xlabel("payload (bits)"); ax.set_ylabel(yl); ax.grid(alpha=0.25, lw=0.5)
    a.set_ylim(0.5, 1.02); b.set_ylim(-0.03, 1.03)
    a.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "capacity.pdf"); plt.close(fig)


# %%
FIGS = {"capacity": fig_capacity, "law": fig_law, "croplimit": fig_croplimit, "ablation": fig_ablation, "gm": fig_gm, "geo": fig_geo}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
        if os.environ.get("FIG_PNG"):                     # preview copies for checking the layout
            import subprocess
            subprocess.run(["pdftoppm", "-png", "-r", "160", "-singlefile", str(OUT / f"{name}.pdf"), str(OUT / name)], check=False)
        print("wrote", OUT / f"{name}.pdf")
