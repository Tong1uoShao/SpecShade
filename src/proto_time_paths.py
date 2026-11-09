"""Per-image extraction time of the unified detector with the final fallback rule, by path: watermarked images aligned by
the sequential searches (geo256u latents), watermarked images that reach the joint fallback (cmat256 latents), and
unwatermarked images (finalq/none latents). Reports mean and P95 (extraction only; inversion is 0.46 s)."""
import glob, json, sys, time, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from models import build_method
from utils.io import load_config
class FB: latent = 64
m = build_method("stylering", load_config(ROOT / "configs/final256_unified.yaml"), FB())
out = {}
for name, pat, n in (("geo256u", "results/geo256u/stylering/inv/*.pt", 80), ("cmat256", "results/cmat256/stylering/inv/*.pt", 60), ("null", "results/finalq/none/inv/*.pt", 60)):
    files = sorted(glob.glob(str(ROOT / pat))); rng = np.random.default_rng(1); files = [files[i] for i in rng.choice(len(files), n, replace=False)]
    ts, fb = [], 0
    for f in files:
        z = torch.load(f, map_location="cpu", weights_only=True); torch.cuda.synchronize(); t0 = time.time(); r = m.extract(z); torch.cuda.synchronize(); ts.append(time.time() - t0); fb += int(bool((r.meta or {}).get("geo_fallback")))
    ts = np.array(ts); out[name] = {"n": n, "mean": float(ts.mean()), "p95": float(np.quantile(ts, 0.95)), "fallback_frac": fb / n,
                                     "mean_fast": float(ts[ts < 4].mean()) if (ts < 4).any() else None, "mean_slow": float(ts[ts >= 4].mean()) if (ts >= 4).any() else None}
    print(name, json.dumps(out[name]), flush=True)
(ROOT / "results/time_paths.json").write_text(json.dumps(out, indent=1))
