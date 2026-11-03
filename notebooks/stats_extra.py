# %% [markdown]
# # Extra detection statistics from the per-image scores (CPU only, no new experiments)
# AUC, TPR@0.1%FPR and bootstrap confidence intervals of SpecShade-256, GS-256 and GaussMarker-256 on the 15 key attacks
# (the cells of the GaussMarker comparison) and over all cells of Table I.
# Writes $TAB_OUT/tab_stats.tex (table*) and $TAB_OUT/numbers_stats.tex (macros used in the text).
# Usage (GPU node): TAB_OUT=$TAB_OUT python notebooks/stats_extra.py

# %%
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("TAB_OUT", ROOT.parent / "docs" / "manuscripts" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


def jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def sc(x):
    return -np.inf if (x is None or not np.isfinite(x)) else float(x)


def cells(exp, method, det):
    """{(attack, strength): [scores]} of the attacked images of one experiment/detector file."""
    d = {}
    for r in jsonl(RES / exp / method / det):
        if r.get("kind") == "clean":
            continue
        d.setdefault((r["attack"], round(float(r["strength"]), 4)), []).append(sc(r["score"]))
    return d


def null(exp, method, name):
    return np.asarray([sc(r["score"]) for r in jsonl(RES / exp / method / name)], dtype=float)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.sort(np.asarray(neg, float))
    lo, hi = np.searchsorted(neg, pos, side="left"), np.searchsorted(neg, pos, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(neg)))


def tpr(pos, neg, fpr):
    if len(neg) * fpr < 1.0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr, method="higher")
    return float((np.asarray(pos) > thr).mean())


def ci_halfwidth(pos, neg, fpr=0.01, n=2000):
    pos = np.asarray(pos, float)
    thr = np.quantile(neg, 1.0 - fpr, method="higher")
    hit = (pos > thr).astype(float)
    b = hit[RNG.integers(0, len(hit), (n, len(hit)))].mean(axis=1)
    return float((np.quantile(b, 0.975) - np.quantile(b, 0.025)) / 2)


# sources: list of (experiment, method, detections file, null file); the first source that holds a cell wins
SRC = {
    "ours": [("final256", "stylering", "detect_no_ecc.jsonl", "detect_null_no_ecc.jsonl"),
             ("geo256", "stylering", "detect_no_ecc.jsonl", "detect_null_no_ecc.jsonl"),
             ("combo256", "stylering", "detect_no_ecc.jsonl", "detect_null_no_ecc.jsonl")],
    "gs": [("final_gs256q", "gaussian_shading", "detect.jsonl", "detect_null.jsonl"),
           ("combo_gs256", "gaussian_shading", "detect.jsonl", "detect_null.jsonl"),
           ("geo_gs256", "gaussian_shading", "detect.jsonl", "detect_null.jsonl")],
    "gm": [("gm256a", "gaussmarker_full", "detect.jsonl", "detect_null.jsonl")],
    "sfw": [("final_sfw", "sfw_hsqr", "detect_known.jsonl", "detect_null_known.jsonl")],     # verification protocol (known key)
    "mx": [("final_maxsive", "maxsive", "detect.jsonl", "detect_null.jsonl")],              # one fixed watermark, as GaussMarker
}
DATA = {m: [(cells(e, mm, d), null(e, mm, n)) for e, mm, d, n in src] for m, src in SRC.items()}


def stats(method, attack, strength, bootstrap=False):
    for pos_by_cell, neg in DATA[method]:
        pos = pos_by_cell.get((attack, round(strength, 4)))
        if pos:
            r = {"auc": auc(pos, neg), "tpr01": tpr(pos, neg, 0.001), "tpr1": tpr(pos, neg, 0.01), "n": len(pos), "nneg": len(neg)}
            if bootstrap:
                r["ci"] = ci_halfwidth(pos, neg)
            return r
    return None


def prompt_ids(exp, method, det):
    """{(attack, strength): [prompt id per score]} aligned with cells()."""
    import re
    d = {}
    for r in jsonl(RES / exp / method / det):
        if r.get("kind") == "clean":
            continue
        d.setdefault((r["attack"], round(float(r["strength"]), 4)), []).append(int(re.search(r"/p(\d+)__", r["image"]).group(1)))
    return d


PIDS = {m: [prompt_ids(e, mm, d) for e, mm, d, n in src] for m, src in SRC.items()}


def ci_grouped(pos, pid, neg, fpr=0.01, n=2000):
    """95% bootstrap interval of the TPR with prompts as the resampling unit (both seeds of a prompt move together)."""
    pos, pid = np.asarray(pos, float), np.asarray(pid)
    thr = np.quantile(neg, 1.0 - fpr, method="higher")
    hit = (pos > thr).astype(float)
    groups = [hit[pid == p] for p in np.unique(pid)]
    g_hit = np.asarray([g.sum() for g in groups]); g_n = np.asarray([len(g) for g in groups])
    idx = RNG.integers(0, len(groups), (n, len(groups)))
    b = g_hit[idx].sum(axis=1) / g_n[idx].sum(axis=1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def stats_grouped(method, attack, strength):
    for (pos_by_cell, neg), pid_by_cell in zip(DATA[method], PIDS[method]):
        pos = pos_by_cell.get((attack, round(strength, 4)))
        if pos:
            lo, hi = ci_grouped(pos, pid_by_cell[(attack, round(strength, 4))], neg)
            return {"tpr1": tpr(pos, neg, 0.01), "lo": lo, "hi": hi}
    return None


# %% the 15 key attacks of the GaussMarker comparison (anime style for the style-transfer attacks)
KEYS = [("Regen.\\ $\\times1$", "regen_sd", 1.0), ("Regen.\\ $\\times2$", "regen_sd", 2.0), ("Regen.\\ $\\times4$", "regen_sd", 4.0),
        ("VAE-Cheng", "vae_cheng", 3.0), ("img2img $s{=}0.3$", "img2img_anime", 0.3), ("img2img $s{=}0.5$", "img2img_anime", 0.5),
        ("AdaIN $\\alpha{=}0.6$", "adain_anime", 0.6), ("AdaIN $\\alpha{=}1.0$", "adain_anime", 1.0), ("IP2P $g_I{=}1.0$", "ip2p_anime", 1.0),
        ("JPEG $Q{=}25$", "jpeg", 25.0), ("Blur $r{=}8$", "blur", 8.0), ("Crop $a{=}0.75$", "crop_scale", 0.75),
        ("Rotation $75^{\\circ}$", "rotate", 75.0), ("Mirror", "mirror", 0.0), ("img2img + crop", "combo_style_crop", 1.0)]
M = ("ours", "gs", "gm", "sfw", "mx")          # AUC for all five; TPR intervals for the three 256-bit payload methods
M3 = ("ours", "gs", "gm")


def best(vals, nd):
    ok = [v for v in vals if v == v]
    top = max(ok) if ok else None
    return [("--" if v != v else (f"\\textbf{{{v:.{nd}f}}}" if top is not None and v >= top - 0.5 * 10 ** (-nd) else f"{v:.{nd}f}")) for v in vals]


rows, acc, WINS = [], {m: {"auc": [], "tpr01": [], "tpr1": []} for m in M}, {m: 0 for m in M}
ABOVE = {"gs": 0, "gm": 0}          # key attacks on which the interval of SpecShade lies entirely above that of the other method
for lab, a, s_ in KEYS:
    st = {m: stats(m, a, s_) for m in M}
    sg = {m: stats_grouped(m, a, s_) for m in M}
    for m in M:
        if st[m]:
            acc[m]["auc"].append(st[m]["auc"]); acc[m]["tpr01"].append(st[m]["tpr01"]); acc[m]["tpr1"].append(st[m]["tpr1"])
    _v = [st[m]["auc"] if st[m] else float("nan") for m in M]
    for m, v in zip(M, _v):
        WINS[m] += int(v >= np.nanmax(_v) - 0.0005)
    for m in ("gs", "gm"):
        if sg[m] and sg["ours"]["lo"] > sg[m]["hi"]:
            ABOVE[m] += 1
    aucs = best(_v, 3)
    top = max(sg[m]["tpr1"] for m in M3 if sg[m])
    tp = []
    for m in M3:
        if not sg[m]:
            tp.append("--"); continue
        s = f"{sg[m]['tpr1']:.2f} [{sg[m]['lo']:.2f}, {sg[m]['hi']:.2f}]"
        tp.append(f"\\textbf{{{s}}}" if sg[m]["tpr1"] >= top - 0.005 else s)
    rows.append(f"{lab} & " + " & ".join(aucs) + " & " + " & ".join(tp) + "\\\\")
mean = lambda m, k: float(np.nanmean(acc[m][k]))
avg_a = best([mean(m, "auc") for m in M], 3)
avg_t = best([mean(m, "tpr1") for m in M3], 2)
both = [(o, g) for o, g in zip(acc["ours"]["tpr01"], acc["gs"]["tpr01"]) if o == o and g == g]   # rows where both have a 1,000-image threshold
N_BOTH = len(both)
rows.append("\\midrule\nAverage & " + " & ".join(avg_a) + " & " + " & ".join(avg_t) + "\\\\")
body = r"""\begin{table*}[t]
\centering
\caption{AUC of the five compared methods and TPR@1\%FPR with its 95\% confidence interval of the 256-bit methods on the 15 key attacks. The intervals are obtained by a bootstrap over prompts, so that the two attacked versions of a prompt are resampled together. Best in bold.}
\label{tab:stats}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{5}{c}{AUC} & \multicolumn{3}{c}{TPR@1\%FPR [95\% CI]}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}
Attack & \textbf{\method} & GS & GaussMarker & SFW & MaXsive & \textbf{\method} & GS & GaussMarker\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(OUT / "tab_stats.tex").write_text("% generated by code/notebooks/stats_extra.py -- do not edit\n" + body)

# %% held-out calibration: threshold from half of the unwatermarked images, false-positive rate on the other half
from scipy.stats import beta, binom


def clopper_pearson(k, n, alpha=0.05):
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


HELD = {}
for m in ("ours", "gs"):
    neg = DATA[m][0][1]
    k_cal = int((neg > np.quantile(neg, 0.99, method="higher")).sum())            # realized count on the calibration set itself
    fprs, dmax = [], 0.0
    for rep in range(200):
        perm = RNG.permutation(len(neg)); cal, test = neg[perm[:500]], neg[perm[500:]]
        thr = np.quantile(cal, 0.99, method="higher")
        fprs.append(float((test > thr).mean()))
        if rep == 0:
            k0, n0 = int((test > thr).sum()), len(test)
            thr_full = np.quantile(neg, 0.99, method="higher")
            for lab, a, s_ in KEYS:
                pos = next((c.get((a, round(s_, 4))) for c, _ in DATA[m] if c.get((a, round(s_, 4)))), None)
                if pos:
                    pos = np.asarray(pos)
                    dmax = max(dmax, abs(float((pos > thr).mean()) - float((pos > thr_full).mean())))
    HELD[m] = {"k_cal": k_cal, "ci_cal": clopper_pearson(k_cal, len(neg)), "fpr_split": k0 / n0, "ci_split": clopper_pearson(k0, n0),
               "fpr_mean": float(np.mean(fprs)), "tpr_dmax": dmax}
print("held-out", json.dumps(HELD, indent=1))

# %% user identification from the decoded payloads: Top-1 accuracy among N enrolled users with random 256-bit identifiers.
# For a decoded payload at Hamming distance d from the true identifier, a random identifier of another user is closer or as
# close with probability F(d) = P[Bin(256, 1/2) <= d], so the true user is ranked first with probability (1 - F(d))^(N-1).
N_USERS = (10 ** 3, 10 ** 6)


def hamming(a, b):
    return bin(int(a.split(":")[-1], 16) ^ int(b.split(":")[-1], 16)).count("1")


def ident(method, attack, strength):
    out = {}
    for e, mm, d, n in SRC[method]:
        rows = [r for r in jsonl(RES / e / mm / d) if r.get("kind") != "clean" and r["attack"] == attack and abs(float(r["strength"]) - strength) < 1e-4]
        if rows:
            dist = np.asarray([hamming(r["payload_hex"], r["bits_dec_hex"]) if r.get("bits_dec_hex") else 128 for r in rows])
            tail = 1.0 - binom.cdf(dist, 256, 0.5)
            for N in N_USERS:
                out[N] = float(np.mean(tail ** (N - 1)))
            out["acc"] = float(np.mean([float(r["bit_acc"]) if r["bit_acc"] is not None else 0.5 for r in rows]))
            return out
    return None


irows, iavg = [], {m: {N: [] for N in N_USERS} for m in ("ours", "gs")}
for lab, a, s_ in KEYS:
    st = {m: ident(m, a, s_) for m in ("ours", "gs")}
    vals = []
    for N in N_USERS:
        v = [st[m][N] if st[m] else float("nan") for m in ("ours", "gs")]
        for m, x in zip(("ours", "gs"), v):
            if x == x:
                iavg[m][N].append(x)
        vals += best(v, 2)
    irows.append(f"{lab} & " + " & ".join(vals) + "\\\\")
irows.append("\\midrule\nAverage & " + " & ".join(sum((best([float(np.mean(iavg[m][N])) for m in ("ours", "gs")], 2) for N in N_USERS), [])) + "\\\\")
ibody = r"""\begin{table}[t]
\centering
\caption{Top-1 identification accuracy among $10^{3}$ and $10^{6}$ enrolled users with random 256-bit identifiers, computed from the decoded payloads on the 15 key attacks. Best in bold.}
\label{tab:ident}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{$10^{3}$ users} & \multicolumn{2}{c}{$10^{6}$ users}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Attack & \textbf{\method} & GS & \textbf{\method} & GS\\
\midrule
""" + "\n".join(irows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_ident.tex").write_text("% generated by code/notebooks/stats_extra.py -- do not edit\n" + ibody)
print("\n".join(irows))

# %% all non-geometric cells of Table I: averages and bootstrap confidence intervals
ours_cells = sorted(DATA["ours"][0][0].keys())
nongeo = [c for c in ours_cells if not c[0].startswith(("rotate", "crop_scale", "mirror", "combo"))]
tab = {}
for m in ("ours", "gs"):
    tab[m] = [r for r in (stats(m, a, s, bootstrap=(m == "ours")) for a, s in nongeo) if r]
nm = {}
nm["AvgAucOurs"] = np.mean([r["auc"] for r in tab["ours"]]); nm["AvgAucGS"] = np.mean([r["auc"] for r in tab["gs"]])
nm["AvgTprZeroOneOurs"] = np.mean([o for o, _ in both]); nm["AvgTprZeroOneGS"] = np.mean([g for _, g in both]); nm["NBoth"] = N_BOTH
hw = [r["ci"] for r in tab["ours"]]
nm["AucWinsOurs"] = WINS["ours"]; nm["AucWinsGM"] = WINS["gm"]; nm["CiMax"] = max(hw); nm["CiMedian"] = float(np.median(hw)); nm["NCells"] = len(nongeo)
# prompt-grouped half-widths over the same cells, and the held-out calibration numbers
ghw = []
for a, s in nongeo:
    r = stats_grouped("ours", a, s)
    if r:
        ghw.append((r["hi"] - r["lo"]) / 2)
nm["CiGroupMax"] = max(ghw); nm["CiGroupMedian"] = float(np.median(ghw))
nm["CiAboveGS"] = ABOVE["gs"]; nm["CiAboveGM"] = ABOVE["gm"]
for m, tag in (("sfw", "SFW"), ("mx", "MX"), ("gm", "GM"), ("ours", "Ours"), ("gs", "GS")):
    if acc[m]["tpr1"]:
        nm[f"KeyAvgTpr{tag}"] = float(np.nanmean(acc[m]["tpr1"])); nm[f"KeyAvgAuc{tag}"] = float(np.nanmean(acc[m]["auc"]))
nm["NFprCalOurs"] = HELD["ours"]["k_cal"]
nm["FprCalLo"] = HELD["ours"]["ci_cal"][0] * 100; nm["FprCalHi"] = HELD["ours"]["ci_cal"][1] * 100
nm["FprSplitOurs"] = HELD["ours"]["fpr_split"] * 100; nm["FprSplitLo"] = HELD["ours"]["ci_split"][0] * 100; nm["FprSplitHi"] = HELD["ours"]["ci_split"][1] * 100
nm["FprSplitMeanOurs"] = HELD["ours"]["fpr_mean"] * 100; nm["FprSplitMeanGS"] = HELD["gs"]["fpr_mean"] * 100
nm["TprSplitDmaxOurs"] = HELD["ours"]["tpr_dmax"]; nm["TprSplitDmaxGS"] = HELD["gs"]["tpr_dmax"]
for N, tag in zip(N_USERS, ("Thousand", "Million")):
    nm[f"Ident{tag}Ours"] = float(np.mean(iavg["ours"][N])); nm[f"Ident{tag}GS"] = float(np.mean(iavg["gs"][N]))
for lab, a, s_, tag in (("", "regen_sd", 2.0, "RegenTwo"), ("", "img2img_anime", 0.5, "ImgFive"), ("", "regen_sd", 4.0, "RegenFour")):
    r = ident("ours", a, s_)
    nm[f"Ident{tag}MillionOurs"] = r[10 ** 6]; nm[f"Ident{tag}ThousandOurs"] = r[10 ** 3]
lines = ["% generated by code/notebooks/stats_extra.py -- do not edit"]
for k, v in nm.items():
    if k.startswith(("AvgAuc", "Ci", "KeyAvgAuc")) and not k.startswith("CiAbove"):
        lines.append(f"\\newcommand{{\\n{k}}}{{{v:.3f}}}")
    elif k.startswith(("Avg", "Ident", "TprSplit", "KeyAvgTpr")):
        lines.append(f"\\newcommand{{\\n{k}}}{{{v:.2f}}}")
    elif k.startswith("Fpr"):
        lines.append(f"\\newcommand{{\\n{k}}}{{{v:.1f}}}")
    else:
        lines.append(f"\\newcommand{{\\n{k}}}{{{int(v)}}}")
(OUT / "numbers_stats.tex").write_text("\n".join(lines) + "\n")
print(json.dumps({k: float(v) for k, v in nm.items()}, indent=1))
print("\n".join(rows))
