"""Shared file helpers: deterministic CSV writing, hashing and git metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(file: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _retry(fn: Callable[[], object], attempts: int = 6, wait: float = 0.5) -> None:
    """Retry a file write: on Windows a scanner or indexer can briefly lock a freshly written file (EINVAL/EACCES)."""
    for i in range(attempts):
        try:
            fn()
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(wait * (i + 1))


def write_csv(df: pd.DataFrame, file: Path, require_evidence: bool = False) -> Path:
    """Write a CSV with a header row and stable formatting (CLAUDE.md section 5).

    With ``require_evidence`` the frame must carry a non-null ``evidence_class`` column.
    """
    if require_evidence:
        if "evidence_class" not in df.columns:
            raise ValueError(f"{file.name}: output table lacks evidence_class")
        if df["evidence_class"].isna().any():
            raise ValueError(f"{file.name}: evidence_class has nulls")
    file.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    _retry(lambda: out.to_csv(file, index=False, lineterminator="\n", encoding="utf-8"))
    return file


def write_text(text: str, file: Path) -> Path:
    """Write UTF-8 text with LF line endings."""
    file.parent.mkdir(parents=True, exist_ok=True)

    def _w() -> None:
        with open(file, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    _retry(_w)
    return file


def write_json(obj: Any, file: Path) -> Path:
    """Write pretty JSON."""
    return write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), file)


def git_commit(root: Path) -> str:
    """Current git commit hash, with '+dirty' if the tree has changes; 'no-git' when unavailable."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        commit = head.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"
    try:
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                            text=True, check=True)
        if st.stdout.strip():
            commit += "+dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return commit


def rel(file: Path, root: Path) -> str:
    """POSIX path of ``file`` relative to ``root``."""
    return Path(file).resolve().relative_to(root.resolve()).as_posix()
