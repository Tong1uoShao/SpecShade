"""Evaluation metrics: bit accuracy, TPR@FPR, CLIP score, FID, Welch t-test."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def bit_acc(a, b) -> float:
    a, b = np.asarray(a).astype(int).ravel(), np.asarray(b).astype(int).ravel()
    return float((a == b).mean())


def tpr_at_fpr(pos_scores, neg_scores, fpr: float) -> tuple[float, float]:
    """Threshold = (1-fpr) quantile of negatives (larger score = more watermark-like).
    Returns (tpr, threshold). None / non-finite scores (rejected images) count as -inf on
    BOTH sides, so a rejected null image is a true negative, never dropped. Returns
    (nan, nan) when the negative set is too small to resolve fpr (fpr * len(neg) < 1)."""
    def _f(s):
        return -np.inf if (s is None or not np.isfinite(s)) else float(s)
    neg = np.asarray([_f(s) for s in neg_scores], dtype=float)
    pos = np.asarray([_f(s) for s in pos_scores], dtype=float)
    if neg.size == 0 or fpr * neg.size < 1.0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr, method="higher"))
    return float((pos > thr).mean()) if pos.size else float("nan"), thr


def welch_ttest(a, b) -> float:
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def clip_scores(image_paths: list[Path], prompts: list[str], model_name: str = "ViT-g-14",
                pretrained: str = "laion2b_s34b_b88k", device: str | None = None, batch: int = 16) -> np.ndarray:
    """OpenCLIP image-text cosine similarity per (image, prompt) [3-convention]."""
    import open_clip
    import torch
    from PIL import Image
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tok = open_clip.get_tokenizer(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in image_paths[i:i + batch]]).to(device)
            txt = tok(prompts[i:i + batch]).to(device)
            fi = model.encode_image(ims); ft = model.encode_text(txt)
            fi, ft = fi / fi.norm(dim=-1, keepdim=True), ft / ft.norm(dim=-1, keepdim=True)
            out.append((fi * ft).sum(-1).float().cpu().numpy())
    return np.concatenate(out)


def fid(gen_dir: str | Path, ref_dir: str | Path) -> float:
    """clean-fid between two image folders (Inception weights must be cached offline)."""
    from cleanfid import fid as cf
    return float(cf.compute_fid(str(gen_dir), str(ref_dir), mode="clean", num_workers=0, verbose=False))
