# SpecShade

Code for **SpecShade: Training-Free Semantic Watermarking Robust to Regeneration and Geometric Attacks** (Jin Xiao, under review).

SpecShade embeds a multi-bit payload into the signs of the Fourier coefficients of the initial noise of a latent diffusion model while keeping their magnitudes, so that the noise stays Gaussian and the diffusion model is used unchanged. Each bit is read by a magnitude-weighted vote over its Fourier coefficients with a payload-agnostic detection score, and a blind self-synchronization that uses the key pattern itself as the template recovers scaling, translation, rotation and reflection before reading. Nothing is trained.

## Layout

```
src/core/            fourier_rings (ring cells, chips, synchronization search), modulation (embed / read), allocation, ecc, sd_backend
src/models/          base interface, stylering.py (SpecShade), baseline/{gaussian_shading, gs_official, gaussmarker, gaussmarker_full, ringid, sfw, maxsive, stable_signature, vine}
src/attacks/         registry, classical, regeneration, style_{img2img, ip2p, adain}, combo (compound edits)
src/generate.py / attack.py / detect.py / detect_null.py / evaluate.py     pipeline stages (all resume by file)
src/proto_freq_survival.py   per-ring sign survival and the same-error-field ablation (Fig. 9a, b)
src/proto_oracle_sync.py     oracle-aligned vs. blind detection on compound attacks
src/proto_diversity.py       generation diversity under a fixed payload (supplement)
notebooks/           figures.py, tables.py, stats_extra.py (AUC, prompt-clustered confidence intervals, held-out calibration, identification), tab_syncabl.py,
                     tab_sd21.py, tab_fourcell.py (domain x vote rule under geometric attacks), tab_syncfail.py (synchronization accuracy), quality_ci.py (FID/KID
                     bootstrap), runtime_mem.py, fig_gate.py, cap_summary.py
configs/             final256_unified.yaml (THE SpecShade-256 detector of every table), final256_hard_unified.yaml (hard cell reads), final256_heldout.yaml,
                     final_gs256.yaml / final_gs256_soft.yaml (GS-256 hard / soft vote), final_gs256_sync_gsh.yaml / _gss.yaml (GS-256 + the same synchronization),
                     final_gm256.yaml, final_sfw.yaml, final_maxsive.yaml, final256_sd21.yaml, cap_*.yaml, pilot_*.yaml, attacks.yaml, methods.yaml;
                     final256.yaml and final256_combo.yaml are the configurations of the original runs (see Reproducing the paper)
scripts/             env.sh, prefetch_models.sh, clone_baselines.sh, paper256_*.sh (original 256-bit batches), unified_queue.sh, editnull_queue.sh, abl2x2_queue.sh,
                     fourcell_queue.sh, after_fourcell.sh, pilot_*_queue.sh, extra_queue.sh, cap_lite.sh, sfw_queue.sh, maxsive_queue.sh, sd21_queue.sh
tests/               CPU unit tests and GPU smoke tests
data/prompts/        prompt lists (Stable-Diffusion-Prompts)
```

## Setup

- Python >= 3.11, PyTorch, `pip install -r requirements.txt` (plus `torchvision` and `qrcode` for the SFW baseline). Tested with torch 2.10 and diffusers 0.39 on one RTX 4090.
- Models are loaded from the local HuggingFace cache; `bash scripts/prefetch_models.sh` downloads them once (SD1.5 for generation and inversion, SDXL and InstructPix2Pix for the attacks, OpenCLIP ViT-g-14 and the CompressAI codecs). Afterwards every script runs with `HF_HUB_OFFLINE=1`.
- AdaIN weights (`vgg_normalised.pth`, `decoder.pth` from the AdaIN release) go to `data/weights/adain/`; style exemplars are made by `python scripts/make_styles.py`.
- Baselines: `bash scripts/clone_baselines.sh` clones the official repositories of Gaussian Shading, GaussMarker, RingID, MaXsive and SFW into `src/models/baseline/vendor/` (read-only; the wrappers in `src/models/baseline/` adapt them to this pipeline). GaussMarker's noise restorer is trained with the vendor script as listed in `configs/methods.yaml`; Stable Signature and VINE-R use their released weights.
- `scripts/env.sh` reads `PY` (interpreter), `CODE_DIR` (repository root) and `HF_HOME`; every shell script sources it.

## Pipeline

Each experiment lives in `results/<exp>/<method>/` and goes through five stages, all of which resume from their output files:

```
python src/generate.py    --exp final256 --method stylering --n 100 --config configs/final256.yaml
python src/attack.py      --exp final256 --method stylering --blocks style,classical,regeneration --seeds 97,101 --config configs/final256.yaml
python src/detect.py      --exp final256 --method stylering --kind both --config configs/final256.yaml
python src/detect_null.py --exp final256 --method stylering --none_exp finalq --config configs/final256.yaml
python src/evaluate.py    --exp final256 --methods stylering --config configs/final256.yaml --variant no_ecc
```

`finalq` holds the 1,000 unwatermarked images used for every 1 %-FPR threshold (`generate.py --exp finalq --method none --n 1000`). The attack blocks are defined in `configs/attacks.yaml` (`style`, `classical`, `regeneration`, `geometry`, `geometry_frac`, `mirror`, `combined`, `key15`, `capacity`, `crop_sweep`). The exact batches that produced the paper's 256-bit results are `scripts/paper256_a.sh` (geometric and compound attacks), `paper256_b.sh` / `paper256_c.sh` (GaussMarker, main table, GS-256), `paper256_d.sh` (reference methods on the common null set) and `paper256_frac.sh` (non-integer rotation angles); `scripts/extra_queue.sh` runs the per-ring survival analysis, the synchronization ablation and the capacity sweep, and `scripts/sfw_queue.sh` / `maxsive_queue.sh` the SFW and MaXsive baselines on the 15 key attacks.

Figures and tables of the paper are produced by `notebooks/figures.py`, `notebooks/tables.py`, `notebooks/stats_extra.py`, `notebooks/tab_syncabl.py` and `notebooks/cap_summary.py` from the `results/` files (`FIG_OUT` / `TAB_OUT` select the output directories).

## Reproducing the paper (one detector for every table)

Every SpecShade number of the paper comes from one detector configuration, `configs/final256_unified.yaml`: rotation search in 1° steps, zoom 0.18–1.0 and shift ±32 px search with the flip candidate, and the joint angle × zoom × shift fallback (5°, 0.02 steps) armed on every input and triggered only when both sequential searches are gated (κ_rot = 6, κ_crop = 8). The decision threshold is the 99th percentile of its scores on the 1,000 unwatermarked images of `finalq` (0.164 for SpecShade-256); the same threshold is used for every attack, for the 500 held-out and the 900 edited unwatermarked images.

```
bash scripts/unified_queue.sh      # 15 key attacks + 4 compound attacks + null set + 500 held-out unwatermarked images (Tables II/III/VI)
bash scripts/editnull_queue.sh     # 900 edited unwatermarked images: false positives after regeneration / style transfer / geometric edits (Table S2)
bash scripts/abl2x2_queue.sh       # embedding domain x vote rule on the 15 key attacks (Fig. 9a)
bash scripts/fourcell_queue.sh     # GS-256 hard/soft vote with the SAME synchronization, SpecShade hard/soft, compound-attack matrix (Table II, supplement)
bash scripts/after_fourcell.sh     # the remaining cells of the 41-attack table under the unified detector; fixed-payload diversity
python notebooks/tables.py; python notebooks/stats_extra.py; python notebooks/figures.py; python notebooks/tab_fourcell.py; python notebooks/tab_syncfail.py
```

`configs/final256.yaml` and `configs/final256_combo.yaml` are the configurations of the original runs (`scripts/paper256_*.sh`): sequential searches only, with the joint fallback (then 3°/0.01) enabled on the compound block alone. The unified detector was run on the same attacked images (`unified_queue.sh`, `after_fourcell.sh`) and reproduces their threshold and every cell, because the fallback's gate never passes on an unwatermarked image and never changes an image already aligned by the sequential searches; the table scripts read the unified-detector directories (`final256ua`, `geo256u`, `combo256u`) and fall back to the original ones only when those are absent. The per-image scores, decoded bits and synchronization metadata of every run are the `detect*.jsonl` files described below.

## Result files

`gen.jsonl` (prompt_id, gen_seed, payload_hex, image) · `atk.jsonl` (src_image, attack, strength, atk_seed, image) · `detect*.jsonl` (per image: bit_acc, score, decoded bits, synchronization metadata) · `detect_null*.jsonl` (scores of the unwatermarked images) · `tables/main_<variant>.csv` (TPR@1%FPR and bit accuracy per attack setting).

## License

MIT (see `LICENSE`). The official baseline repositories cloned by `scripts/clone_baselines.sh` keep their own licenses (SFW: CC BY-NC 4.0).

## Citation

```
@article{xiao2026specshade,
  title   = {SpecShade: Training-Free Semantic Watermarking Robust to Regeneration and Geometric Attacks},
  author  = {Xiao, Jin},
  year    = {2026},
  note    = {under review}
}
```
