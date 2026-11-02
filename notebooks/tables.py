# %% [markdown]
# # Paper tables (LaTeX, booktabs) generated from code/results/<exp>/tables
# Run on the GPU node:  TAB_OUT=$TAB_OUT python notebooks/tables.py
# Experiments still running fall back to the previous run (noted in a LaTeX comment) or print "--".

# %%
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
RES = ROOT / "results"
OUT = Path(os.environ.get("TAB_OUT", ROOT.parent / "docs" / "manuscripts" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)
NOTES: list[str] = []


def table(exp: str, method: str, fallback: str | None = None) -> pd.DataFrame | None:
    for e in ([exp] + ([fallback] if fallback else [])):
        for name in ("main_no_ecc.csv", "main_soft.csv"):
            p = RES / e / "tables" / name
            if p.exists():
                df = pd.read_csv(p)
                df = df[df["method"] == method]
                if len(df):
                    if e != exp:
                        NOTES.append(f"{exp}/{method} not ready -> using {e}")
                    return df.set_index("cell")
    NOTES.append(f"{exp}/{method} missing")
    return None


def val(df, cells, col):
    if df is None:
        return None
    cells = [c for c in cells if c in df.index]
    return float(df.loc[cells, col].mean()) if cells else None


def fam_cells(attack: str, strength: str, styles=("anime", "comic", "watercolor")) -> list[str]:
    return [f"{attack}_{s}@{strength}" for s in styles]


def fmt(x, nd=2):
    return "--" if x is None or x != x else f"{x:.{nd}f}"


def bold_best(vals, nd=2, floor=None):
    """format a row of numbers: best bold, second best underlined (only when >= 3 values compete); ties share.
    A second best at chance level is not underlined: floor = 0.10 for TPR rows (nd=2), 0.55 for bit accuracy (nd=3)."""
    floor = (0.10 if nd == 2 else 0.55) if floor is None else floor
    good = sorted({round(v, nd) for v in vals if v is not None and v == v}, reverse=True)
    best = good[0] if good else None
    second = good[1] if len(good) > 1 and sum(v is not None and v == v for v in vals) >= 3 else None
    if second is not None and second <= floor:
        second = None
    out = []
    for v in vals:
        s = fmt(v, nd)
        if v is not None and v == v and round(v, nd) == best:
            s = f"\\textbf{{{s}}}"
        elif v is not None and v == v and second is not None and round(v, nd) == second:
            s = f"\\underline{{{s}}}"
        out.append(s)
    return out


MARK_NOTE = r"Best value in bold; second best underlined unless it is at chance level (TPR $\le 0.10$, Acc $\le 0.55$)."


def write(name: str, body: str):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


# %% [markdown]
# ## Table I -- regeneration, style-transfer and classical attacks (256-bit main comparison)

# %%
ROWS_MAIN = [
    ("No attack", ["clean"]),
    (None, "Regeneration / learned compression"),
    ("Regen.\\ $\\times1$", ["regen_sd@1.0"]), ("Regen.\\ $\\times2$", ["regen_sd@2.0"]),
    ("Regen.\\ $\\times4$", ["regen_sd@4.0"]),
    ("VAE-Cheng $q{=}3$", ["vae_cheng@3.0", "vae_cheng@3"]), ("VAE-Ball\\'e $q{=}3$", ["vae_bmshj@3.0", "vae_bmshj@3"]),
    (None, "Style transfer"),
    ("img2img $s{=}0.3$", fam_cells("img2img", "0.3")), ("img2img $s{=}0.5$", fam_cells("img2img", "0.5")),
    ("img2img $s{=}0.7$", fam_cells("img2img", "0.7")),
    ("AdaIN $\\alpha{=}0.3$", fam_cells("adain", "0.3")), ("AdaIN $\\alpha{=}0.6$", fam_cells("adain", "0.6")),
    ("AdaIN $\\alpha{=}1.0$", fam_cells("adain", "1.0")),
    ("IP2P $g_I{=}2.0$", fam_cells("ip2p", "2.0")), ("IP2P $g_I{=}1.5$", fam_cells("ip2p", "1.5")),
    ("IP2P $g_I{=}1.0$", fam_cells("ip2p", "1.0")),
    (None, "Classical distortions"),
    ("JPEG $Q{=}25$", ["jpeg@25.0", "jpeg@25"]), ("Gaussian blur $r{=}8$", ["blur@8.0", "blur@8"]),
    ("Gaussian noise $\\sigma{=}0.1$", ["noise@0.1"]), ("Median $k{=}7$", ["median@7.0", "median@7"]),
    ("Resize $0.25$", ["resize@0.25"]), ("Brightness $\\times6$", ["brightness@6.0", "brightness@6"]),
    ("Random drop $0.8$", ["rand_drop@0.8"]),
    (None, "Geometric"),
    ("Random crop $a{=}0.75$", ["crop_scale@0.75"]), ("Rotation $75^\\circ$", ["rotate@75.0", "rotate@75"]),
]


def table_main():
    ours = table("final256", "stylering", fallback="final_fr256")
    gs = table("final_gs256q", "gaussian_shading", fallback="final_gs256")
    others = [(table("final_refq", m, fallback="final"), lab) for m, lab in (("ringid", "RingID"), ("stable_signature", "Stable Sig."), ("vine", "VINE-R"))]
    lines = []
    avg = {k: [] for k in ("o_t", "o_a", "g_t", "g_a", "r", "s", "v")}
    for lab, cells in ROWS_MAIN:
        if lab is None:
            lines.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\emph{{{cells}}}}}\\\\")
            continue
        ot, oa = val(ours, cells, "tpr_1fpr"), val(ours, cells, "bit_acc_all")
        gt, ga = val(gs, cells, "tpr_1fpr"), val(gs, cells, "bit_acc_all")
        rest = [val(df, cells, "tpr_1fpr") for df, _ in others]
        if lab != "No attack" and not cells[0].startswith(("crop_scale", "rotate")):   # average: non-geometric rows
            for k, v in zip(avg, (ot, oa, gt, ga, *rest)):
                if v is not None:
                    avg[k].append(v)
        tprs = bold_best([ot, gt, *rest])
        accs = bold_best([oa, ga], 3)
        lines.append(f"{lab} & {tprs[0]} & {accs[0]} & {tprs[1]} & {accs[1]} & {tprs[2]} & {tprs[3]} & {tprs[4]}\\\\")
    m = {k: (sum(v) / len(v) if v else None) for k, v in avg.items()}
    tprs, accs = bold_best([m["o_t"], m["g_t"], m["r"], m["s"], m["v"]]), bold_best([m["o_a"], m["g_a"]], 3)
    lines.append("\\midrule\n" + f"Average (non-geometric) & {tprs[0]} & {accs[0]} & {tprs[1]} & {accs[1]} & {tprs[2]} & {tprs[3]} & {tprs[4]}\\\\")
    body = r"""\begin{table*}[t]
\centering
\caption{Robustness against non-geometric and basic geometric attacks (TPR@1\%FPR and bit accuracy). Methods with other capacities are listed by TPR only; style-transfer rows average three styles. Best in bold; second best underlined unless at chance level.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{\method-256 (ours)}} & \multicolumn{2}{c}{GS-256~\cite{yang2024gaussian}} & RingID~\cite{ci2024ringid} & Stable Sig.~\cite{fernandez2023stable} & VINE-R~\cite{lu2025vine}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Attack & TPR & Acc & TPR & Acc & TPR & TPR & TPR\\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_main", body.replace("{lcccccccc}", "{lccccccc}"))


# %% [markdown]
# ## Table II -- head-to-head with the full GaussMarker (GNR) on the 15 key cells

# %%
KEY15 = [("No attack", ["clean"]), ("Regen.\\ $\\times1$", ["regen_sd@1.0"]),
         ("Regen.\\ $\\times2$", ["regen_sd@2.0"]), ("Regen.\\ $\\times4$", ["regen_sd@4.0"]),
         ("VAE-Cheng $q{=}3$", ["vae_cheng@3.0", "vae_cheng@3"]),
         ("img2img $s{=}0.3$", ["img2img_anime@0.3"]), ("img2img $s{=}0.5$", ["img2img_anime@0.5"]),
         ("AdaIN $\\alpha{=}0.6$", ["adain_anime@0.6"]), ("AdaIN $\\alpha{=}1.0$", ["adain_anime@1.0"]),
         ("IP2P $g_I{=}1.0$", ["ip2p_anime@1.0"]), ("JPEG $Q{=}25$", ["jpeg@25.0", "jpeg@25"]),
         ("Gaussian blur $r{=}8$", ["blur@8.0", "blur@8"]), ("Random crop $a{=}0.75$", ["crop_scale@0.75"]),
         ("Rotation $75^\\circ$", ["rotate@75.0", "rotate@75"]), ("Horizontal mirror", ["mirror@0.0", "mirror@0"]),
         ("img2img $0.3$ + crop $0.75$", ["combo_style_crop@1.0", "combo_style_crop@1"])]


GM_FUSION = {"gm256": "bit agreement minus the normalized ring distance", "gm256a": "bit agreement with the known message after the restorer"}


def gm_table():
    """The stronger of the two GaussMarker score fusions (by mean TPR over the attacked key cells), used for the
    whole column -- never picked per cell."""
    best, best_v, best_e = None, -1.0, None
    for e in ("gm256a", "gm256"):
        df = table(e, "gaussmarker_full")
        if df is None:
            continue
        v = [val(df, cells + [c.replace("@1.0", "@1") for c in cells], "tpr_1fpr") for lab, cells in KEY15 if lab != "No attack"]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v) if v else -1
        if m > best_v:
            best, best_v, best_e = df, m, e
    if best_e:
        NOTES.append(f"GaussMarker column uses {best_e} ({GM_FUSION[best_e]})")
    return best, best_e


def table_gm():
    ours = [table("final256", "stylering", fallback="final_fr256"), table("geo256", "stylering"), table("combo256", "stylering")]
    gs = [table("final_gs256q", "gaussian_shading", fallback="final_gs256"), table("combo_gs256", "gaussian_shading")]
    gm, gm_e = gm_table()

    def pick(dfs, cells, col):
        for df in dfs:
            v = val(df, cells, col)
            if v is not None:
                return v
        return None
    lines = []
    for lab, cells in KEY15:
        o = (pick(ours, cells, "tpr_1fpr"), pick(ours, cells, "bit_acc_all"))
        g = (pick(gs, cells, "tpr_1fpr"), pick(gs, cells, "bit_acc_all"))
        k = (val(gm, cells, "tpr_1fpr"), val(gm, cells, "bit_acc_all"))
        t, a = bold_best([o[0], g[0], k[0]]), bold_best([o[1], g[1], k[1]], 3)
        lines.append(f"{lab} & {t[0]} & {a[0]} & {t[1]} & {a[1]} & {t[2]} & {a[2]}\\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Comparison with the full GaussMarker (trained GNR, official 256-bit recipe and single-message protocol) on 15 key cells (anime style for the style-transfer attacks). TPR@1\%FPR / bit accuracy. GaussMarker is scored with """ + GM_FUSION.get(gm_e or "gm256", "") + r""", the stronger of the two score fusions we evaluated. """ + MARK_NOTE + r"""}
\label{tab:gm}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\method-256} & \multicolumn{2}{c}{GS-256} & \multicolumn{2}{c}{GaussMarker-256}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Attack & TPR & Acc & TPR & Acc & TPR & Acc\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_gm", body)


# %% [markdown]
# ## Table III -- geometric and compound attacks

# %%
GEO = [("Rotation $15^\\circ$", ["rotate@15.0"]), ("Rotation $30^\\circ$", ["rotate@30.0"]), ("Rotation $45^\\circ$", ["rotate@45.0"]),
       ("Rotation $75^\\circ$", ["rotate@75.0"]), ("Rotation $90^\\circ$", ["rotate@90.0"]),
       ("Random crop $a{=}0.9$", ["crop_scale@0.9"]), ("Random crop $a{=}0.75$", ["crop_scale@0.75"]),
       ("Random crop $a{=}0.5$", ["crop_scale@0.5"]),
       ("Horizontal mirror", ["mirror@0.0"]), ("Vertical flip", ["mirror@1.0"]),
       (None, "Compound edits"),
       ("img2img $0.3$ + crop $0.75$", ["combo_style_crop@1.0"]), ("IP2P $1.5$ + mirror", ["combo_style_mirror@1.0"]),
       ("Regen.\\ $\\times1$ + rot.\\ $30^\\circ$", ["combo_regen_rotate@1.0"]), ("Crop $0.75$ + rot.\\ $30^\\circ$", ["combo_crop_rotate@1.0"])]


def table_geo():
    ours = [table("geo256", "stylering"), table("combo256", "stylering")]
    gso = [table("geo_gs256", "gaussian_shading"), table("combo_gs256", "gaussian_shading")]
    gm, _ = gm_table()

    def pick(dfs, cells, col):
        for df in dfs:
            v = val(df, cells, col)
            if v is not None:
                return v
        return None
    lines = []
    for lab, cells in GEO:
        if lab is None:
            lines.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\emph{{{cells}}}}}\\\\")
            continue
        cols = [(pick(d, cells, "tpr_1fpr"), pick(d, cells, "bit_acc_all")) for d in (ours, gso, [gm])]
        t = bold_best([c[0] for c in cols]); a = bold_best([c[1] for c in cols], 3)
        lines.append(f"{lab} & " + " & ".join(f"{t[i]} & {a[i]}" for i in range(3)) + "\\\\")
    body = r"""\begin{table*}[t]
\centering
\caption{Geometric and compound attacks (TPR@1\%FPR / bit accuracy, 256 bits). GS-256 uses its original decoder. GaussMarker-256 was evaluated on the key cells only (Table~\ref{tab:gm}). """ + MARK_NOTE + r"""}
\label{tab:geo}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\method-256} & \multicolumn{2}{c}{GS-256} & \multicolumn{2}{c}{GaussMarker-256}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Attack & TPR & Acc & TPR & Acc & TPR & Acc\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_geo", body)


# %% [markdown]
# ## Table IV -- image quality (1,000 images; FID vs. COCO val2017 5k; CLIP ViT-g; t-test vs. no watermark)

# %%
def table_quality():
    q = json.load(open(RES / "finalq" / "quality.json"))
    q256 = json.load(open(RES / "finalq_fr256" / "quality.json"))["stylering"]
    q16 = json.load(open(RES / "finalq_fr2" / "quality.json"))["stylering"]
    rows = [("No watermark", q["none"], False), ("\\method-256 (ours)", q256, True), ("\\method-16 (ours)", q16, True),
            ("Gaussian Shading$^\\dagger$", q["gaussian_shading"], False), ("GaussMarker$^\\dagger$", q["gaussmarker"], False),
            ("RingID", q["ringid"], False), ("Stable Signature", q["stable_signature"], False), ("VINE-R", q["vine"], False)]
    lines = []
    for lab, d, _ in rows:
        p = d.get("ttest_p_clip_vs_none")
        cells = [lab, f"{d['fid']:.2f}", f"{d['clip_mean']:.4f}", '--' if p is None else f'{p:.2f}']
        if lab == "No watermark":                      # reference row, set in bold and separated from the methods
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + "\\\\" + ("\n\\midrule" if lab == "No watermark" else ""))
    body = r"""\begin{table}[t]
\centering
\caption{Image quality over 1{,}000 generations: FID, CLIP score and the $p$-value of a $t$-test of CLIP scores against no watermark. $^\dagger$Measured at 16 bits (distribution-preserving at any capacity).}
\label{tab:quality}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
Method & FID $\downarrow$ & CLIP $\uparrow$ & $p$ (CLIP)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_quality", body)


# %% [markdown]
# ## Table V -- peak gating (16-bit runs: no search / always-on search without gating / gated search)

# %%
def table_gating():
    runs = [("No search", table("full_rep", "stylering")), ("Search, no gating", table("full_rep_srch", "stylering")),
            ("Search + gating", table("full_rep_syncq", "stylering"))]
    cells = [("img2img $0.5$", fam_cells("img2img", "0.5")), ("img2img $0.7$", fam_cells("img2img", "0.7")),
             ("Regen.\\ $\\times2$", ["regen_sd@2.0"]), ("Regen.\\ $\\times4$", ["regen_sd@4.0"])]
    lines = []
    for lab, df in runs:
        thr = None if df is None else float(df["thr_1fpr"].dropna().iloc[0])
        lines.append(f"{lab} & {fmt(thr, 3)} & " + " & ".join(fmt(val(df, c, 'tpr_1fpr')) for _, c in cells) + "\\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Effect of peak gating on non-geometric attacks (16-bit payload, 50 images $\times$ 2 seeds). Threshold $\tau$ at 1\% FPR and TPR. An always-on search without gating inflates $\tau$ and drags weak images to random alignments; with gating the search costs nothing.}
\label{tab:gating}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Detector & $\tau$ & """ + " & ".join(l for l, _ in cells) + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_gating", body)


# %% [markdown]
# ## Table VI -- capacity: 16 vs 256 bits (SpecShade vs Gaussian Shading)

# %%
def table_capacity():
    sets = [("\\method-16", table("final_fr2q", "stylering", fallback="final_fr2")), ("GS-16", table("final_refq", "gaussian_shading", fallback="final")),
            ("\\method-256", table("final256", "stylering", fallback="final_fr256")),
            ("GS-256", table("final_gs256q", "gaussian_shading", fallback="final_gs256"))]
    cells = [("Regen.\\ $\\times2$", ["regen_sd@2.0"]), ("Regen.\\ $\\times4$", ["regen_sd@4.0"]),
             ("img2img $0.5$", fam_cells("img2img", "0.5")), ("img2img $0.7$", fam_cells("img2img", "0.7")),
             ("AdaIN $1.0$", fam_cells("adain", "1.0"))]
    lines = []
    for lab, df in sets:
        lines.append(f"{lab} & " + " & ".join(f"{fmt(val(df, c, 'tpr_1fpr'))} / {fmt(val(df, c, 'bit_acc_all'), 3)}" for _, c in cells) + "\\\\")
    body = r"""\begin{table*}[t]
\centering
\caption{Capacity versus robustness (TPR@1\%FPR / bit accuracy; style-transfer columns average three styles). Votes per bit: \method-16 $\approx$992, GS-16 1{,}024, \method-256 $\approx$62, GS-256 64.}
\label{tab:capacity}
\begin{tabular}{lccccc}
\toprule
Method & """ + " & ".join(l for l, _ in cells) + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_capacity", body)


# %% [markdown]
# ## Table VII (appendix) -- all 41 cells, SpecShade-256 vs GS-256

# %%
FULL_NAMES = {"clean": "No attack", "regen_sd": "Regen.", "vae_cheng": "VAE-Cheng", "vae_bmshj": "VAE-Ball\\'e",
              "jpeg": "JPEG", "blur": "Gaussian blur", "noise": "Gaussian noise", "median": "Median", "resize": "Resize",
              "brightness": "Brightness", "rand_drop": "Random drop", "crop_scale": "Random crop", "rotate": "Rotation"}


def table_full():
    ours = table("final256", "stylering", fallback="final_fr256")
    gs = table("final_gs256q", "gaussian_shading", fallback="final_gs256")
    order = ["clean", "regen_sd@1.0", "regen_sd@2.0", "regen_sd@4.0", "vae_cheng@3.0", "vae_bmshj@3.0"]
    for a, ss in (("img2img", ("0.3", "0.5", "0.7")), ("adain", ("0.3", "0.6", "1.0")), ("ip2p", ("2.0", "1.5", "1.0"))):
        order += [f"{a}_{st}@{s}" for st in ("anime", "comic", "watercolor") for s in ss]
    order += ["jpeg@25.0", "blur@8.0",
              "noise@0.1", "median@7.0", "resize@0.25", "brightness@6.0", "rand_drop@0.8", "crop_scale@0.75", "rotate@75.0"]
    lines = []
    for c in order:
        head, _, st = c.partition("@")
        fam_name = head.split("_")[0]
        if fam_name in ("img2img", "adain", "ip2p"):
            style = head.split("_", 1)[1]
            lab = {"img2img": "img2img", "adain": "AdaIN", "ip2p": "IP2P"}[fam_name] + f" ({style}) {st}"
        else:
            lab = FULL_NAMES.get(head, head) + (f" {st}" if st else "")
        t = bold_best([val(ours, [c], "tpr_1fpr"), val(gs, [c], "tpr_1fpr")])
        a = bold_best([val(ours, [c], "bit_acc_all"), val(gs, [c], "bit_acc_all")], 3)
        lines.append(f"{lab} & {t[0]} & {a[0]} & {t[1]} & {a[1]}\\\\")
    body = r"""\begin{table}[t]
\centering
\caption{All 41 attack settings (TPR@1\%FPR and bit accuracy), 256 bits. Best in bold.}
\label{tab:full}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{\method-256} & \multicolumn{2}{c}{GS-256}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Attack & TPR & Acc & TPR & Acc\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_full", body)


# %% [markdown]
# ## numbers.tex -- every number quoted in the text, as macros (text and tables can never disagree)

# %%
NUM: dict[str, str] = {}


def num(name: str, x, nd=2):
    NUM[name] = fmt(x, nd)


def _jsonl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def _variants(attack, s):
    return [f"{attack}@{s}", f"{attack}@{float(s)}", f"{attack}@{int(float(s))}" if float(s).is_integer() else f"{attack}@{s}"]


def _cell_key(r):
    return "%s@%s" % (r["attack"], r["strength"])


def extra_numbers():
    """Geometry, compound and GaussMarker numbers + data-driven sentences (tables/gm_text.tex)."""
    import statistics as st
    ours_geo, ours_combo = table("geo256", "stylering"), table("combo256", "stylering")
    gs_geo, gs_combo = table("geo_gs256", "gaussian_shading"), table("combo_gs256", "gaussian_shading")
    gss_geo, gss_combo = table("gs256_sync", "gaussian_shading"), table("combo_gs256s", "gaussian_shading")
    rot = [("Fifteen", "15"), ("Thirty", "30"), ("FortyFive", "45"), ("SeventyFive", "75"), ("Ninety", "90")]
    crop = [("Nine", "0.9"), ("SevenFive", "0.75"), ("Half", "0.5")]
    mir = [("H", "0"), ("V", "1")]
    geo_cells = [_variants("rotate", a) for _, a in rot] + [_variants("crop_scale", a) for _, a in crop]
    for tag, df in (("Ours", ours_geo), ("GS", gs_geo), ("GSSync", gss_geo)):
        for k, a in rot:
            num(f"Rot{k}Tpr{tag}", val(df, _variants("rotate", a), "tpr_1fpr")); num(f"Rot{k}Acc{tag}", val(df, _variants("rotate", a), "bit_acc_all"), 3)
        for k, a in crop:
            num(f"Crop{k}Tpr{tag}", val(df, _variants("crop_scale", a), "tpr_1fpr")); num(f"Crop{k}Acc{tag}", val(df, _variants("crop_scale", a), "bit_acc_all"), 3)
        tprs = [val(df, c, "tpr_1fpr") for c in geo_cells]
        tprs = [t for t in tprs if t is not None]
        num(f"GeoMinTpr{tag}", min(tprs) if tprs else None); num(f"GeoMaxTpr{tag}", max(tprs) if tprs else None)
    for tag, df in (("Ours", ours_geo), ("GS", gs_combo), ("GSSync", gss_combo)):
        for k, a in mir:
            num(f"Mirror{k}Tpr{tag}", val(df, _variants("mirror", a), "tpr_1fpr")); num(f"Mirror{k}Acc{tag}", val(df, _variants("mirror", a), "bit_acc_all"), 3)
    combos = [("StyleCrop", "combo_style_crop"), ("StyleMirror", "combo_style_mirror"), ("RegenRot", "combo_regen_rotate"), ("CropRot", "combo_crop_rotate")]
    for tag, df in (("Ours", ours_combo), ("GS", gs_combo), ("GSSync", gss_combo)):
        ts = []
        for k, a in combos:
            t = val(df, _variants(a, "1"), "tpr_1fpr"); ts += [t] if t is not None else []
            num(f"Combo{k}Tpr{tag}", t); num(f"Combo{k}Acc{tag}", val(df, _variants(a, "1"), "bit_acc_all"), 3)
        num(f"ComboAvgTpr{tag}", sum(ts) / len(ts) if ts else None)
    # gating z-scores and the crop-0.9 behaviour, from the per-image detection records
    rows = _jsonl(RES / "geo256/stylering/detect_no_ecc.jsonl")
    if rows:
        by = {}
        for r in rows:
            if r["kind"] != "clean":
                by.setdefault(_cell_key(r), []).append(r)
        rz = [st.median((r.get("meta") or {}).get("sync_z", 0) for r in v) for k, v in by.items() if k.startswith("rotate")]
        cz = [st.median((r.get("meta") or {}).get("crop_z", 0) for r in v) for k, v in by.items() if k.startswith("crop_scale")]
        num("RotZMin", min(rz), 0); num("RotZMax", max(rz), 0); num("CropZMin", min(cz), 0); num("CropZMax", max(cz), 0)
        c9 = next((v for k, v in by.items() if k in ("crop_scale@0.9",)), [])
        if c9:
            spur = [r for r in c9 if not (r.get("meta") or {}).get("gated", True) and (r.get("meta") or {}).get("theta", 0) not in (0, 0.0)]
            num("CropNineSpurious", len(spur) / len(c9)); num("CropNineSpuriousAcc", st.mean(r["bit_acc"] for r in spur) if spur else None, 3)
        fl = [r for k, v in by.items() if k.startswith("mirror") for r in v]
        num("MirrorFlipRate", sum(bool((r.get("meta") or {}).get("flipped")) for r in fl) / len(fl) if fl else None)
    crows = [r for r in _jsonl(RES / "combo256/stylering/detect_no_ecc.jsonl") if r["kind"] != "clean" and r["attack"] == "combo_crop_rotate"]
    num("CropRotFallback", sum(bool((r.get("meta") or {}).get("geo_fallback")) and not (r.get("meta") or {}).get("geo_gated", True) for r in crows) / len(crows) if crows else None)
    # GaussMarker head-to-head on the 15 key cells
    gm, gm_e = gm_table()
    ours_all = [table("final256", "stylering", fallback="final_fr256"), ours_geo, ours_combo]
    def pick(cells):
        for df in ours_all:
            v = val(df, cells, "tpr_1fpr")
            if v is not None:
                return v
        return None
    keys = [(lab, cells) for lab, cells in KEY15 if lab != "No attack"]
    text = []
    for k in ("GmWins", "GmTies", "GmLosses", "GmLossesWord", "GmCells", "GmAvgTprOurs", "GmAvgTprGM"):
        num(k, None)                                   # defined even before the GaussMarker run finishes
    if gm is not None:
        wins, ties, losses, lose_labs, ot, gt = 0, 0, 0, [], [], []
        for lab, cells in keys:
            o, g = pick(cells), val(gm, cells + [c.replace("@1.0", "@1") for c in cells], "tpr_1fpr")
            if o is None or g is None:
                continue
            ot.append(o); gt.append(g)
            if o > g + 0.02:
                wins += 1
            elif g > o + 0.02:
                losses += 1; lose_labs.append(lab)
            else:
                ties += 1
        num("GmWins", wins, 0); num("GmTies", ties, 0); num("GmLosses", losses, 0); num("GmCells", wins + ties + losses, 0)
        NUM["GmLossesWord"] = "none" if losses == 0 else str(losses)
        num("GmAvgTprOurs", sum(ot) / len(ot) if ot else None); num("GmAvgTprGM", sum(gt) / len(gt) if gt else None)
        sent = (f"As shown in Fig.~\\ref{{fig:gm}}, \\method-256 achieves a higher TPR than GaussMarker-256 on {wins} of the "
                f"{wins + ties + losses} key attacks")
        sent += (f", the same TPR (within $0.02$) on {ties} and a lower TPR on {losses} ({', '.join(lose_labs)})." if losses
                 else f" and the same TPR (within $0.02$) on the other {ties}.")
        text.append(sent)
        text.append(f"On average, the TPR of \\method\\ is {fmt(sum(ot) / len(ot))}, whereas that of GaussMarker is {fmt(sum(gt) / len(gt))}.")
        acc_best = 0
        for lab, cells in keys:
            go = val(gm, cells + [c.replace("@1.0", "@1") for c in cells], "bit_acc_all")
            oo = next((val(df, cells, "bit_acc_all") for df in ours_all if val(df, cells, "bit_acc_all") is not None), None)
            if go is not None and oo is not None and go > oo + 0.005:
                acc_best += 1
        nulls = [json.loads(l)["score"] for l in open(RES / gm_e / "gaussmarker_full/detect_null.jsonl")]
        hi = sum(x >= 0.9 for x in nulls); tau = sorted(nulls)[int(0.99 * len(nulls)) - 1]
        num("GmAccBetter", acc_best, 0); num("GmNullHigh", hi, 0); num("GmNullN", len(nulls), 0); num("GmTau", tau, 3)
        if gm_e == "gm256a":
            n_null = f"{len(nulls):,}".replace(",", "{,}")
            text.append(f"Although GaussMarker decodes more bits correctly than \\method\\ on {acc_best} of these attacks, its TPR is lower. "
                        f"This is because its restorer, which is trained to map its inputs onto the deployed message, also maps some unwatermarked "
                        f"images onto it: {hi} of the {n_null} unwatermarked images reach a bit agreement of at least $0.9$ with that message. "
                        f"Therefore, its threshold at 1\\% FPR is a bit agreement of {fmt(tau, 3)}, which most regenerated or stylized images do not reach.")
        g_rows = _jsonl(RES / "gm256/gaussmarker_full/detect.jsonl")   # restorer effect: per-image meta (fusion-independent)
        eff = []
        for lab, a, s in (("a $75^\\circ$ rotation", "rotate", "75"), ("a $0.75$ crop", "crop_scale", "0.75"),
                          ("two rounds of regeneration", "regen_sd", "2"), ("img2img $s{=}0.5$", "img2img_anime", "0.5")):
            rs = [r for r in g_rows if r.get("kind") != "clean" and r.get("attack") == a and abs(float(r.get("strength", -1)) - float(s)) < 1e-6]
            if not rs:
                continue
            w = st.mean((r.get("meta") or {}).get("acc_known", 0) for r in rs); wo = st.mean((r.get("meta") or {}).get("acc_known_nognr", 0) for r in rs)
            eff.append(f"from {fmt(wo, 3)} to {fmt(w, 3)} after {lab}")
        if eff:
            text.append("The restorer raises its bit accuracy " + (", ".join(eff[:-1]) + " and " + eff[-1] if len(eff) > 1 else eff[0]) + ".")
    else:
        text.append("\\todo{GaussMarker-256 results not available yet.}")
    (OUT / "gm_text.tex").write_text("% generated by code/notebooks/tables.py -- do not edit\n" + " ".join(text) + "\n")


def numbers():
    extra_numbers()
    ours = table("final256", "stylering", fallback="final_fr256")
    gs = table("final_gs256q", "gaussian_shading", fallback="final_gs256")
    attacked = [c for lab, c in ROWS_MAIN if lab not in (None, "No attack") and not c[0].startswith(("crop_scale", "rotate"))]
    for tag, df in (("Ours", ours), ("GS", gs)):
        num(f"AvgTpr{tag}", sum(val(df, c, "tpr_1fpr") for c in attacked) / len(attacked))
        num(f"AvgAcc{tag}", sum(val(df, c, "bit_acc_all") for c in attacked) / len(attacked), 3)
        for key, cells in (("ImgFive", fam_cells("img2img", "0.5")), ("ImgThree", fam_cells("img2img", "0.3")),
                           ("ImgSeven", fam_cells("img2img", "0.7")), ("AdainOne", fam_cells("adain", "1.0")),
                           ("RegenOne", ["regen_sd@1.0"]), ("RegenTwo", ["regen_sd@2.0"]), ("RegenFour", ["regen_sd@4.0"]),
                           ("Drop", ["rand_drop@0.8"]), ("Bright", ["brightness@6.0"]), ("Blur", ["blur@8.0"]),
                           ("Jpeg", ["jpeg@25.0"]), ("Crop", ["crop_scale@0.75"]), ("Rot", ["rotate@75.0"]),
                           ("Clean", ["clean"])):
            num(f"{key}Tpr{tag}", val(df, cells, "tpr_1fpr"))
            num(f"{key}Acc{tag}", val(df, cells, "bit_acc_all"), 3)
        num(f"Thr{tag}", float(df["thr_1fpr"].dropna().iloc[0]), 3)
    for tag, df in (("SixteenOurs", table("final_fr2q", "stylering", fallback="final_fr2")),
                    ("SixteenGS", table("final_refq", "gaussian_shading", fallback="final"))):
        for key, cells in (("ImgFive", fam_cells("img2img", "0.5")), ("ImgSeven", fam_cells("img2img", "0.7")),
                           ("RegenTwo", ["regen_sd@2.0"]), ("RegenFour", ["regen_sd@4.0"])):
            num(f"{key}Tpr{tag}", val(df, cells, "tpr_1fpr"))
    q = json.load(open(RES / "finalq" / "quality.json")); q256 = json.load(open(RES / "finalq_fr256" / "quality.json"))["stylering"]
    num("FidNone", q["none"]["fid"]); num("FidOurs", q256["fid"]); num("FidGS", q["gaussian_shading"]["fid"])
    num("ClipNone", q["none"]["clip_mean"], 4); num("ClipOurs", q256["clip_mean"], 4); num("ClipP", q256["ttest_p_clip_vs_none"])
    for tag, exp in (("NoSearch", "full_rep"), ("NoGate", "full_rep_srch"), ("Gated", "full_rep_syncq")):
        df = table(exp, "stylering")
        num(f"Thr{tag}", float(df["thr_1fpr"].dropna().iloc[0]), 3)
        num(f"ImgSevenTpr{tag}", val(df, fam_cells("img2img", "0.7"), "tpr_1fpr"))
        num(f"ImgFiveTpr{tag}", val(df, fam_cells("img2img", "0.5"), "tpr_1fpr"))
        num(f"RegenFourTpr{tag}", val(df, ["regen_sd@4.0"], "tpr_1fpr"))
    lines = [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(NUM.items())]
    (OUT / "numbers.tex").write_text("% generated by code/notebooks/tables.py -- do not edit\n" + "\n".join(lines) + "\n")
    print("wrote", OUT / "numbers.tex", len(lines), "macros")


if __name__ == "__main__":
    for fn in (table_main, table_gm, table_geo, table_quality, table_gating, table_capacity, table_full, numbers):
        fn()
    (OUT / "NOTES.txt").write_text("\n".join(NOTES) + "\n")
    print("\n".join(NOTES))
