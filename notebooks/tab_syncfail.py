# %% [markdown]
# # Supplementary table: synchronisation accuracy and failure analysis (CPU only)
# For every geometric and compound cell of SpecShade-256: how often each search accepts a peak, how accurate the recovered
# parameters are (angle error, zoom error, shift consistency, flip detection), and the blind vs. oracle-aligned TPR of the
# compound attacks (results/oracle_sync, src/proto_oracle_sync.py). Reads the detection metadata of geo256u (fallback geo256)
# and combo256u. Writes $TAB_OUT/tab_syncfail.tex and $TAB_OUT/numbers_syncfail.tex.
# Usage (GPU node): TAB_OUT=$TAB_OUT python notebooks/tab_syncfail.py

# %%
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.environ.get("TAB_OUT", ROOT.parent / "docs" / "manuscripts" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)


def jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def first_complete(*exps):
    for e in exps:
        d, a = RES / e / "stylering" / "detect_no_ecc.jsonl", RES / e / "stylering" / "atk.jsonl"
        if d.exists() and a.exists() and len(jsonl(d)) >= len(jsonl(a)):
            return e
    return exps[-1]


GEO, COMBO = first_complete("geo256u", "geo256"), first_complete("combo256u", "combo256")
print("sources:", GEO, COMBO)
rows = [r for e in (GEO, COMBO) for r in jsonl(RES / e / "stylering" / "detect_no_ecc.jsonl") if r.get("kind") == "atk"]
null = np.asarray([r["score"] for r in jsonl(RES / COMBO / "stylering" / "detect_null_no_ecc.jsonl") if r["score"] is not None])
if not len(null):
    null = np.asarray([r["score"] for r in jsonl(RES / GEO / "stylering" / "detect_null_no_ecc.jsonl") if r["score"] is not None])
thr = float(np.quantile(null, 0.99, method="higher"))
ORC = json.load(open(RES / "oracle_sync" / "summary.json"))["attacks"] if (RES / "oracle_sync" / "summary.json").exists() else {}


def crop_offset(seed, frac, w=512):
    """Centre offset of the random crop in latent pixels (CropScale draws the box from default_rng(seed))."""
    cw = int(w * np.sqrt(frac)); rng = np.random.default_rng(int(seed))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, w - cw + 1))
    return (x0 + cw / 2 - w / 2) / 8, (y0 + cw / 2 - w / 2) / 8


def ang_err(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def cell(attack, strength):
    R = [r for r in rows if r["attack"] == attack and abs(float(r["strength"]) - strength) < 1e-6]
    M = [r.get("meta") or {} for r in R]
    return R, M


def pct(x):
    return f"{100 * x:.0f}\\%"


lines, nums = [], {}
# ---- rotations ----
for ang in (15, 30, 45, 75, 90):
    R, M = cell("rotate", ang)
    if not R:
        continue
    truth = (360 - ang) % 360
    acc_rot = np.mean([not m.get("gated", True) for m in M])
    err = np.asarray([ang_err(m.get("theta", 0.0), truth) if not m.get("flipped") else 180.0 for m in M])
    tpr = np.mean([r["score"] > thr for r in R])
    lines.append(f"Rotation ${ang}^{{\\circ}}$ & {len(R)} & rotation {pct(acc_rot)} & angle, median $|\\Delta\\theta|$ = ${np.median(err):.1f}^{{\\circ}}$ & {pct(np.mean(err <= 1.0))} & {tpr:.2f} & --\\\\")
    nums[f"RotErr{ang}"] = float(np.median(err)); nums[f"RotOk{ang}"] = float(np.mean(err <= 1.0))
# ---- crops ----
for a in (0.9, 0.75, 0.5):
    R, M = cell("crop_scale", a)
    if not R:
        continue
    s_true = np.sqrt(a)
    acc_crop = np.mean([not m.get("crop_gated", True) for m in M])
    es = np.asarray([m.get("scale", 1.0) for m in M])
    seeds = sorted({r["atk_seed"] for r in R})
    dist = []                      # shift error against the crop box: the undo-shift is minus the centre offset (quarter-pixel grid)
    for sd in seeds:
        T = np.asarray([(m.get("ty", 0.0), m.get("tx", 0.0)) for r, m in zip(R, M) if r["atk_seed"] == sd])
        dx, dy = crop_offset(sd, a)
        dist += list(np.sqrt((T[:, 0] + dy) ** 2 + (T[:, 1] + dx) ** 2))
        med = np.median(T, axis=0)
        print(f"crop a={a} seed={sd}: median estimate (ty, tx) = ({med[0]:+.2f}, {med[1]:+.2f}) px; crop-box centre offset (dx, dy) = ({dx:+.2f}, {dy:+.2f}) px")
    dist = np.asarray(dist)
    tpr = np.mean([r["score"] > thr for r in R])
    lines.append(f"Crop $a{{=}}{a}$ & {len(R)} & shift/zoom {pct(acc_crop)} & zoom, median $|\\Delta\\gamma|$ = ${np.median(np.abs(es - s_true)):.3f}$; shift, median error ${np.median(dist):.2f}$ px & {pct(np.mean((np.abs(es - s_true) <= 0.02) & (dist <= 1.0)))} & {tpr:.2f} & --\\\\")
    nums[f"CropScaleErr{int(a*100)}"] = float(np.median(np.abs(es - s_true))); nums[f"CropShiftErr{int(a*100)}"] = float(np.median(dist))
# ---- mirror ----
for st, lab in ((0, "Mirror"), (1, "Vertical flip")):
    R, M = cell("mirror", st)
    if not R:
        continue
    fl = np.asarray([bool(m.get("flipped")) for m in M])
    ok = fl if st == 0 else np.asarray([bool(m.get("flipped")) and ang_err(m.get("theta", 0.0), 180.0) <= 1.0 for m in M])
    tpr = np.mean([r["score"] > thr for r in R])
    lines.append(f"{lab} & {len(R)} & flip candidate {pct(fl.mean())} & " + ("--" if st == 0 else "angle $180^{\\circ}$ + flip") + f" & {pct(ok.mean())} & {tpr:.2f} & --\\\\")
    nums[f"FlipRate{st}"] = float(fl.mean())
# ---- compound ----
for name, lab in (("combo_crop_rotate", "Crop $0.75$ + rot.\\ $30^{\\circ}$"), ("combo_style_mirror", "IP2P $1.5$ + mirror"),
                  ("combo_style_crop", "img2img $0.3$ + crop $0.75$"), ("combo_regen_rotate", "Regen.\\ $\\times1$ + rot.\\ $30^{\\circ}$")):
    R, M = cell(name, 1.0)
    if not R:
        continue
    joint = np.asarray([bool(m.get("geo_fallback")) and not m.get("geo_gated", True) for m in M])
    seq = np.asarray([(not m.get("gated", True)) or (not m.get("crop_gated", True)) for m in M])
    tpr = np.mean([r["score"] > thr for r in R])
    o = ORC.get(name, {})
    orc = f"{o['oracle_tpr']:.2f}" if o else "--"
    if name == "combo_crop_rotate":
        acc = f"joint {pct(joint.mean())}"; par = "angle $330^{\\circ}$, zoom $0.87$"; okc = pct(joint.mean())
    elif name == "combo_style_mirror":
        fl = np.asarray([bool(m.get("flipped")) for m in M]); acc = f"flip candidate {pct(fl.mean())}"; par = "--"; okc = pct(fl.mean())
    else:
        acc = f"sequential {pct(seq.mean())}, joint {pct(joint.mean())}"
        par = "accepted peaks correct" if o else "--"
        okc = pct(o["blind_frac_accepted_correct"]) + (f" (wrong {pct(o['blind_frac_accepted_wrong'])})" if o else "")
    lines.append(f"{lab} & {len(R)} & {acc} & {par} & {okc} & {tpr:.2f} & {orc}\\\\")
    nums[f"Joint{name}"] = float(joint.mean())
body = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Synchronisation of \method-256 on the geometric and compound attacks: share of images for which a search accepts a peak, accuracy of the recovered parameters (angle error against the applied rotation; zoom error against $\sqrt{a}$ and shift error against the centre of the crop box, in latent pixels), share within tolerance ($1^{\circ}$, $0.02$ zoom, 1 px), and TPR@1\%FPR of the blind detector and, for the compound attacks analysed in Sec.~IV-D, of the oracle alignment.}
\label{tab:syncfail}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrlllcc}
\toprule
Attack & $n$ & Search accepted & Recovered parameters & Within tol. & TPR blind & TPR oracle\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(OUT / "tab_syncfail.tex").write_text("% generated by code/notebooks/tab_syncfail.py -- do not edit\n" + body)
(OUT / "numbers_syncfail.tex").write_text("% generated by code/notebooks/tab_syncfail.py -- do not edit\n" + "\n".join(
    f"\\newcommand{{\\nSf{k}}}{{{v:.3f}}}" for k, v in nums.items()) + "\n")
print("threshold", thr); print("\n".join(lines))
