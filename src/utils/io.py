"""Stateless IO helpers: config, jsonl, resume-by-existence, md5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_keys(path: str | Path, key: str) -> set:
    """Set of already-written values of `key` in a jsonl (resume support)."""
    return {row[key] for row in read_jsonl(path) if key in row}


def exists_nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
