"""# OURS compound attacks: a chain of existing engines applied in order.

The chain is given in the AttackSpec `style` field as  "engine[/style]@strength+engine[/style]@strength+...",
e.g. "style_img2img/anime@0.3+crop_scale@0.75" (stylise, then random crop + resize back). Every step uses the
same attack seed, so a chain is as reproducible as its parts. The matrix strength is unused (keep it 1).
"""
from __future__ import annotations

from PIL import Image

from attacks.registry import Attack, get_attack


def parse_chain(chain: str) -> list[tuple[str, str | None, float]]:
    steps = []
    for part in chain.split("+"):
        head, strength = part.strip().rsplit("@", 1)
        engine, style = (head.split("/", 1) + [None])[:2]
        steps.append((engine, style, float(strength)))
    if not steps:
        raise ValueError(f"empty attack chain: {chain!r}")
    return steps


class Combo(Attack):
    engine = "combo"
    heavy = True                      # may hold a diffusion engine; sub-engines live in the registry cache

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._held: set[str] = set()      # heavy sub-engines currently resident

    def _release(self, keep: set[str]) -> None:
        from attacks import registry
        for engine in list(self._held - keep):
            inst = registry._INSTANCES.pop(engine, None)
            if inst is not None:
                inst.free()
            self._held.discard(engine)

    def __call__(self, image: Image.Image, strength: float, seed: int, prompt: str = "",
                 style: str | None = None) -> Image.Image:
        if not style:
            raise ValueError("combo attack needs the chain in the `style` field")
        steps = parse_chain(style)
        self._release({e for e, _, _ in steps})          # one chain's big models at a time (24 GB card)
        out = image.convert("RGB")
        for engine, sub_style, s in steps:
            kw = {"style": sub_style} if sub_style else {}
            atk = get_attack(engine, self.cfg)
            if atk.heavy:
                self._held.add(engine)
            out = atk(out, s, seed, prompt, **kw)
        return out

    def free(self) -> None:
        self._release(set())
