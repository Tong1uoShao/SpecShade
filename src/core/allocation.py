"""Survival map -> allocation table (part of the secret key), plus control strategies.

# OURS: Method 3.4. `measured` ranks cells by the probe's sign-survival rate S(g);
#   `uniform` / `lowcut` / `highring` are the ablation controls of Part 3, exp. 5-6
#   (lowcut = VINE-style 'just use low frequencies' [5]; highring = worst-case control).
# Pure functions, CPU-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .fourier_rings import Cell
from .modulation import CellCalib


@dataclass
class SurvivalMap:
    cells: list[Cell]
    S: np.ndarray                       # sign-survival rate per cell, in [0,1]
    mu: np.ndarray                      # mean of y*sigma_true per cell (attack-averaged)
    sigma: np.ndarray                   # std  of y*sigma_true per cell
    per_attack: dict[str, np.ndarray] = field(default_factory=dict)  # attack name -> S
    meta: dict = field(default_factory=dict)

    def to_calib(self, attack_set: str = "all") -> CellCalib:
        return CellCalib(mu={c.key(): float(m) for c, m in zip(self.cells, self.mu)},
                         sigma={c.key(): float(s) for c, s in zip(self.cells, self.sigma)},
                         attack_set=attack_set)

    def scores(self, attack_subset: list[str] | None = None) -> np.ndarray:
        if not attack_subset:
            return np.asarray(self.S, dtype=np.float64)
        missing = [a for a in attack_subset if a not in self.per_attack]
        if missing:
            raise KeyError(f"attacks not in survival map: {missing}")
        return np.mean([self.per_attack[a] for a in attack_subset], axis=0)

    def save(self, path: str) -> None:
        keys = np.array([c.key() for c in self.cells], dtype=np.int64)
        np.savez(path, keys=keys, S=self.S, mu=self.mu, sigma=self.sigma,
                 attack_names=np.array(list(self.per_attack.keys())),
                 attack_S=np.array([self.per_attack[a] for a in self.per_attack])
                 if self.per_attack else np.zeros((0, len(self.cells))),
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str) -> "SurvivalMap":
        d = np.load(path, allow_pickle=False)
        cells = [Cell(int(c), int(r), None if q < 0 else int(q)) for c, r, q in d["keys"]]
        names = [str(a) for a in d["attack_names"]]
        per_attack = {a: d["attack_S"][i] for i, a in enumerate(names)}
        return cls(cells=cells, S=d["S"], mu=d["mu"], sigma=d["sigma"],
                   per_attack=per_attack, meta=json.loads(str(d["meta"])))


def _full_image(cells: list[Cell]) -> list[int]:
    return [i for i, c in enumerate(cells) if c.q is None]


def allocate(smap: SurvivalMap, n_code: int, strategy: str = "measured",
             attack_subset: list[str] | None = None,
             r_cut: int | None = None, r_hi: int | None = None) -> list[Cell]:
    """Return the ordered list of n_code cells that carry the codeword.

    Position i of the returned list carries codeword bit i; with a systematic BCH code the
    first k positions are the message bits, so `measured` puts them on the best cells.
    """
    idx = _full_image(smap.cells)          # allocation never uses quadrant cells
    cells = [smap.cells[i] for i in idx]
    if n_code > len(cells):
        raise ValueError(f"n_code={n_code} exceeds available cells {len(cells)}")
    if strategy == "measured":
        s = smap.scores(attack_subset)[idx]
        order = np.argsort(-s, kind="stable")        # descending survival
        return [cells[i] for i in order[:n_code]]
    if strategy == "uniform":
        by_r = sorted(range(len(cells)), key=lambda i: (cells[i].r, cells[i].c))
        pick = np.linspace(0, len(by_r) - 1, n_code).round().astype(int)
        return [cells[by_r[i]] for i in pick]
    if strategy == "lowcut":
        n_ch = len({c.c for c in cells})
        r_cut = r_cut or int(np.ceil(n_code / n_ch))
        cand = sorted([c for c in cells if c.r <= r_cut], key=lambda c: (c.r, c.c))
        if len(cand) < n_code:
            raise ValueError(f"lowcut r_cut={r_cut} yields {len(cand)} < {n_code} cells")
        return cand[:n_code]
    if strategy == "highring":
        n_ch = len({c.c for c in cells})
        r_max = max(c.r for c in cells)
        r_hi = r_hi or (r_max - int(np.ceil(n_code / n_ch)) + 1)
        cand = sorted([c for c in cells if c.r >= r_hi], key=lambda c: (-c.r, c.c))
        if len(cand) < n_code:
            raise ValueError(f"highring r_hi={r_hi} yields {len(cand)} < {n_code} cells")
        return cand[:n_code]
    raise ValueError(f"unknown allocation strategy: {strategy}")


def allocate_repeat(cells: list[Cell], n_code: int, sizes, pair_parts: bool = False) -> tuple[list[Cell], np.ndarray]:
    """# OURS full-budget: every cell carries one codeword position and positions are repeated
    across cells. Greedy longest-first balancing (largest cell into the lightest position) so
    that the vote count per position is as even as ring geometry allows.

    pair_parts: the real- and imaginary-part cells of the same ring go to the SAME position.
      Without it the greedy fill happens to make every position purely real or purely imaginary,
      and a 180-degree rotation (spectrum conjugation, which negates every imaginary read-out)
      then leaves the blind synchronisation statistic unchanged -- an exact alias. Mixing both
      parts in every position makes the alias read R - I instead of R + I, so it is rejected.

    Returns (cells, pos_of) with pos_of[i] the codeword position carried by cells[i]."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if len(sizes) != len(cells):
        raise ValueError("sizes must match cells")
    if len(cells) < n_code:
        raise ValueError(f"n_code={n_code} exceeds available cells {len(cells)}")
    load = np.zeros(n_code)
    pos_of = np.zeros(len(cells), dtype=int)
    if pair_parts:
        groups: dict[tuple[int, int, int | None], list[int]] = {}
        for i, c in enumerate(cells):
            groups.setdefault((c.c, c.r, c.q), []).append(i)
        units = list(groups.values())
        # more positions than ring pairs (e.g. 256 bit on 176 rings): split the largest pairs into
        # single cells until every position can get a unit; the split (pure) positions hold the
        # largest rings, the paired (mixed) ones still break the 180-degree alias.
        n_split = max(0, n_code - len(units))
        if n_split > 0:
            order = np.argsort([-sizes[u].sum() for u in units], kind="stable")
            split_ids = set(int(j) for j in order[:n_split])
            units = [[i] for j, u in enumerate(units) if j in split_ids for i in u] + \
                    [u for j, u in enumerate(units) if j not in split_ids]
        usize = np.array([sizes[u].sum() for u in units])
        for j in np.argsort(-usize, kind="stable"):
            p = int(np.argmin(load))
            for i in units[j]:
                pos_of[i] = p
            load[p] += usize[j]
        return list(cells), pos_of
    for i in np.argsort(-sizes, kind="stable"):
        p = int(np.argmin(load))
        pos_of[i] = p
        load[p] += sizes[i]
    return list(cells), pos_of
