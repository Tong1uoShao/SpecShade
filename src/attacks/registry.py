"""Attack registry: engine name -> lazily built Attack object; attack-matrix parsing.

An Attack is called as attack(image, strength, seed, prompt="") -> PIL.Image (512x512).
Engines hold heavy models; `free_all()` releases them (HAMI 13 GB: one big model at a time).
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class AttackSpec:
    name: str          # matrix key, e.g. img2img_anime
    engine: str        # registry engine
    style: str | None
    strengths: tuple


class Attack:
    engine: str = "base"
    heavy: bool = False

    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "") -> Image.Image:
        raise NotImplementedError

    def free(self) -> None:
        pass


_INSTANCES: dict[str, Attack] = {}


def _factory(engine: str, cfg: dict) -> Attack:
    if engine == "none":
        from attacks.classical import Identity
        return Identity()
    if engine == "style_img2img":
        from attacks.style_img2img import StyleImg2Img
        return StyleImg2Img(cfg)
    if engine == "style_ip2p":
        from attacks.style_ip2p import StyleIP2P
        return StyleIP2P(cfg)
    if engine == "style_adain":
        from attacks.style_adain import StyleAdaIN
        return StyleAdaIN(cfg)
    if engine in ("mirror", "jpeg", "rotate", "crop_scale", "gauss_blur", "gauss_noise", "brightness",
                  "rand_drop", "median", "resize"):
        from attacks.classical import CLASSICAL
        return CLASSICAL[engine]()
    if engine == "diffusion_regen":
        from attacks.regeneration import DiffusionRegen
        return DiffusionRegen(cfg)
    if engine in ("vae_compress", "vae_compress_b"):
        from attacks.regeneration import VAECompress
        return VAECompress(cfg, model="cheng2020-anchor" if engine == "vae_compress" else "bmshj2018-factorized")
    if engine == "combo":
        from attacks.combo import Combo
        return Combo(cfg)
    if engine == "adaptive_lowfreq":
        from attacks.adaptive import LowFreqPerturb
        return LowFreqPerturb(cfg)
    raise KeyError(f"unknown attack engine: {engine}")


def get_attack(engine: str, cfg: dict) -> Attack:
    if engine not in _INSTANCES:
        _INSTANCES[engine] = _factory(engine, cfg)
    return _INSTANCES[engine]


def free_all() -> None:
    for a in list(_INSTANCES.values()):
        a.free()
    _INSTANCES.clear()


def build_matrix(attacks_cfg: dict, blocks: list[str], names: list[str] | None = None) -> list[AttackSpec]:
    """Flatten configs/attacks.yaml blocks into AttackSpecs (optionally filtered by name)."""
    specs = []
    for block in blocks:
        for name, ent in attacks_cfg.get(block, {}).items():
            if names and name not in names:
                continue
            specs.append(AttackSpec(name, ent["engine"], ent.get("style"), tuple(ent["strengths"])))
    return specs


def run_attacks(images: list[dict], specs: list[AttackSpec], seeds: list[int], cfg: dict,
                out_dir, meta_path, logger=None) -> None:
    """Shared attack loop (attack.py and probe.py): engines outer, images inner (one heavy
    model resident at a time), resume by output-file existence via meta jsonl.

    images: [{"image": path(rel to ROOT), "prompt": str, "src_key": str}]
    """
    from pathlib import Path
    import time
    from utils.io import append_jsonl, done_keys, ensure_dir
    root = Path(cfg["_root"])
    ensure_dir(out_dir)
    done = done_keys(meta_path, "image")
    by_engine: dict[str, list[AttackSpec]] = {}
    for s in specs:
        by_engine.setdefault(s.engine, []).append(s)
    for engine, engine_specs in by_engine.items():
        atk = None
        for spec in engine_specs:
            for strength in spec.strengths:
                for seed in seeds:
                    for im in images:
                        stem = Path(im["image"]).stem
                        rel = str(Path(out_dir).relative_to(root) / f"{stem}__{spec.name}_{strength}_{seed}.png")
                        if rel in done:
                            continue
                        if atk is None:
                            atk = get_attack(engine, cfg)
                        t0 = time.time()
                        kw = {"style": spec.style} if spec.style else {}
                        out = atk(Image.open(root / im["image"]).convert("RGB"), strength, seed, im["prompt"], **kw)
                        out.save(root / rel)
                        append_jsonl(meta_path, {"src_image": im["image"], "src_key": im.get("src_key", stem),
                                                 "attack": spec.name, "engine": engine, "strength": strength,
                                                 "atk_seed": seed, "image": rel,
                                                 "t_atk_s": round(time.time() - t0, 2)})
        if atk is not None and atk.heavy:
            atk.free()
            _INSTANCES.pop(engine, None)
            if logger:
                logger.info(f"freed engine {engine}")
