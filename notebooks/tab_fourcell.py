# %% [markdown]
# # Domain x vote-rule comparison under geometric and compound attacks with ONE synchronisation (CPU only)
# Four detectors on their own images: GS-256 hard / soft vote + the SpecShade searches (gsync_geo, gsync_combo; variants
# gsh / gss) and SpecShade-256 hard / soft cell reads with the unified detector (geo256hu + combo256hu, geo256u + combo256u).
# Writes $TAB_OUT/tab_fourcell.tex (supplement, TPR / Acc of the four detectors on every cell + detection time) and
# $TAB_OUT/numbers_fourcell.tex (averages and runtimes quoted in the text).
# Usage (GPU node): TAB_OUT=$TAB_OUT python notebooks/tab_fourcell.py

# %%
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("TAB_OUT", ROOT.parent / "docs" / "manuscripts" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)

DET = {  # label, [(exp, method, variant)], null file
    "gsh": ("GS-256, hard vote + sync.", [("gsync_geo", "gaussian_shading", "gsh"), ("gsync_combo", "gaussian_shading", "gsh")]),
    "gss": ("GS-256, soft vote + sync.", [("gsync_geo", "gaussian_shading", "gss"), ("gsync_combo", "gaussian_shading", "gss")]),
    "ourh": ("\\method-256, hard vote", [("geo256hu", "stylering", "no_ecc"), ("combo256hu", "stylering", "no_ecc")]),
    "ours": ("\\method-256, soft vote", [("geo256u", "stylering", "no_ecc"), ("combo256u", "stylering", "no_ecc")]),
}
CELLS = [("Rotation $15^{\\circ}$", "rotate@15"), ("Rotation $30^{\\circ}$", "rotate@30"), ("Rotation $45^{\\circ}$", "rotate@45"),
         ("Rotation $75^{\\circ}$", "rotate@75"), ("Rotation $90^{\\circ}$", "rotate@90"), ("Crop $a{=}0.9$", "crop_scale@0.9"),
         ("Crop $a{=}0.75$", "crop_scale@0.75"), ("Crop $a{=}0.5$", "crop_scale@0.5"), ("Mirror", "mirror@0"), ("Vertical flip", "mirror@1"),
         ("Crop $0.75$ + rot.\\ $30^{\\circ}$", "combo_crop_rotate@1"), ("IP2P $1.5$ + mirror", "combo_style_mirror@1"),
         ("img2img $0.3$ + crop $0.75$", "combo_style_crop@1"), ("Regen.\\ $\\times1$ + rot.\\ $30^{\\circ}$", "combo_regen_rotate@1")]


def load(exp, method, variant):
    p = RES / exp / "tables" / f"main_{variant}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p); df = df[df["method"] == method]
    return df.set_index("cell") if len(df) else None


def cell(dfs, key, col):
    for df in dfs:
        if df is None:
            continue
        for k in (key, key.replace("@", "@") + ".0", key.split("@")[0] + "@" + str(float(key.split("@")[1]))):
            if k in df.index:
                return float(df.loc[k, col])
    return float("nan")


def det_time(exp, method, variant):
    """Mean extraction time per attacked image (t_det_s, incl. all searches) and the mean for unwatermarked images."""
    f = RES / exp / method / (f"detect_{variant}.jsonl" if variant != "soft" else "detect.jsonl")
    rows = [json.loads(l) for l in open(f)] if f.exists() else []
    t = [r["t_det_s"] for r in rows if r.get("kind") == "atk" and "t_det_s" in r]
    g = RES / exp / method / (f"detect_null_{variant}.jsonl" if variant != "soft" else "detect_null.jsonl")
    nrows = [json.loads(l) for l in open(g)] if g.exists() else []
    tn = [r["t_det_s"] for r in nrows if "t_det_s" in r]
    return (float(np.mean(t)) if t else float("nan")), (float(np.mean(tn)) if tn else float("nan"))


tabs = {k: [load(*s) for s in srcs] for k, (lab, srcs) in DET.items()}
avail = {k: any(d is not None for d in v) for k, v in tabs.items()}
print("available:", avail)
rows, means = [], {k: {"t": [], "a": []} for k in DET}
for lab, key in CELLS:
    vals = []
    for k in DET:
        t, a = cell(tabs[k], key, "tpr_1fpr"), cell(tabs[k], key, "bit_acc_all")
        vals.append((t, a))
        if t == t:
            means[k]["t"].append(t); means[k]["a"].append(a)
    top = np.nanmax([v[0] for v in vals]) if any(v[0] == v[0] for v in vals) else float("nan")
    cells = []
    for t, a in vals:
        if t != t:
            cells.append("-- & --"); continue
        ts = f"{t:.2f}"; cells.append((f"\\textbf{{{ts}}}" if t >= top - 0.005 else ts) + f" & {a:.3f}")
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
rows.append("\\midrule\nAverage & " + " & ".join((f"{np.mean(m['t']):.2f} & {np.mean(m['a']):.3f}" if m["t"] else "-- & --") for m in means.values()) + "\\\\")
times = {k: det_time(*DET[k][1][0]) for k in DET}
TP = RES / "time_paths.json"               # SpecShade timed with the final fallback rule (src/proto_time_paths.py): aligned path / unwatermarked
if TP.exists():
    tp = json.load(open(TP)); times["ours"] = times["ourh"] = (tp["geo256u"]["mean"], tp["null"]["mean"])
rows.append("Detection time (s), attacked / unwm. & " + " & ".join(
    (f"\\multicolumn{{2}}{{c}}{{{t:.1f} / {tn:.1f}}}" if t == t else "\\multicolumn{2}{c}{--}") for t, tn in times.values()) + "\\\\")
body = r"""\begin{table*}[t]
\centering
\caption{Embedding domain $\times$ vote rule under geometric and compound attacks with the same synchronisation (rotation search, zoom and shift search, flip candidate, gated joint fallback; 256 bits; TPR@1\%FPR / bit accuracy, 100 prompts $\times$ 2 attack seeds, thresholds from 1,000 unwatermarked images processed by each detector). Detection time excludes the DDIM inversion (for \\method\\ with the joint search run only when neither flip candidate is aligned). Best TPR in bold.}
\label{tab:fourcell}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{""" + DET["gsh"][0] + r"""} & \multicolumn{2}{c}{""" + DET["gss"][0] + r"""} & \multicolumn{2}{c}{""" + DET["ourh"][0] + r"""} & \multicolumn{2}{c}{""" + DET["ours"][0] + r"""}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Attack & TPR & Acc & TPR & Acc & TPR & Acc & TPR & Acc\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(OUT / "tab_fourcell.tex").write_text("% generated by code/notebooks/tab_fourcell.py -- do not edit\n" + body)
lines = ["% generated by code/notebooks/tab_fourcell.py -- do not edit"]
tag = {"gsh": "GSH", "gss": "GSS", "ourh": "OurH", "ours": "OurS"}
for k in DET:
    if means[k]["t"]:
        lines += [f"\\newcommand{{\\nFcAvgTpr{tag[k]}}}{{{np.mean(means[k]['t']):.2f}}}", f"\\newcommand{{\\nFcAvgAcc{tag[k]}}}{{{np.mean(means[k]['a']):.3f}}}"]
    t, tn = times[k]
    if t == t:
        lines += [f"\\newcommand{{\\nFcTime{tag[k]}}}{{{t:.1f}}}", f"\\newcommand{{\\nFcTimeNull{tag[k]}}}{{{tn:.1f}}}"]
for lab, key in CELLS:
    for k in DET:
        t = cell(tabs[k], key, "tpr_1fpr")
        if t == t:
            name = "".join(ch for ch in key.replace("@", "").replace("_", "").replace(".", "").title() if ch.isalnum())
            lines.append(f"\\newcommand{{\\nFc{tag[k]}{name}}}{{{t:.2f}}}")
(OUT / "numbers_fourcell.tex").write_text("\n".join(lines) + "\n")
print("\n".join(rows))
