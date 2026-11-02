"""Timestamped file+console logger: logs/YY-MM-DD_HH-MM-SS_{stage}.log"""
from __future__ import annotations

import logging
import time
from pathlib import Path


def get_logger(stage: str, log_dir: str | Path = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%y-%m-%d_%H-%M-%S')}_{stage}.log"
    logger = logging.getLogger(f"stylering.{stage}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(Path(log_dir) / name, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger
