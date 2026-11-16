"""Configuration loading and repository paths (CLAUDE.md section 4: no hard-coded absolute paths)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def find_root(start: Path | None = None) -> Path:
    """Return the repository root: ``$TZBANK_ROOT`` if set, else the first parent holding config/settings.yaml."""
    env = os.environ.get("TZBANK_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "settings.yaml").exists():
            return parent
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "config" / "settings.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate the repository root (config/settings.yaml). Set TZBANK_ROOT.")


@lru_cache(maxsize=4)
def _load_settings(root: str) -> dict[str, Any]:
    with open(Path(root) / "config" / "settings.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def settings(root: Path | None = None) -> dict[str, Any]:
    """Return config/settings.yaml as a dict."""
    return _load_settings(str(root or find_root()))


def path(key: str, root: Path | None = None) -> Path:
    """Absolute path for a key under ``paths`` or ``inputs`` in settings.yaml."""
    r = root or find_root()
    s = settings(r)
    rel = s.get("paths", {}).get(key) or s.get("inputs", {}).get(key)
    if rel is None:
        raise KeyError(f"No path configured for {key!r}")
    return r / rel


def valuation_dates(root: Path | None = None) -> dict[str, pd.Timestamp]:
    """Decision D1 valuation dates as Timestamps, keyed 'historical' and 'current'."""
    return {k: pd.Timestamp(v) for k, v in settings(root)["valuation_dates"].items()}


def banks(root: Path | None = None) -> list[dict[str, Any]]:
    """The seven-bank universe from config/banks.yaml."""
    with open(path("banks", root), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["banks"]


def bank_basis(root: Path | None = None) -> dict[str, str]:
    """Reporting basis per bank_id (GROUP or BANK)."""
    return {b["bank_id"]: b["reporting_basis"] for b in banks(root)}


def evidence_classes(root: Path | None = None) -> list[str]:
    """Blueprint section 6 evidence-class vocabulary."""
    return list(settings(root)["evidence_classes"])
