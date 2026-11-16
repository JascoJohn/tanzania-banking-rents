"""Run logs (CLAUDE.md section 4): outputs/run_logs/<timestamp>_<stage>.json."""

from __future__ import annotations

import datetime as dt
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from tzbank import __version__
from tzbank.config import find_root, path, settings
from tzbank.io import git_commit, write_json


def assumption_register_version(root: Path | None = None) -> str:
    """Register Version from data/processed/assumptions.csv (or the workbook), else 'unset'."""
    root = root or find_root()
    f = path("processed", root) / "assumptions.csv"
    if not f.exists():
        f = path("interim", root) / "inputs_assumptions.csv"
    if not f.exists():
        return "unknown (assumptions not yet ingested)"
    df = pd.read_csv(f, dtype=str)
    col = "Register Version"
    vals = sorted(df[col].dropna().unique()) if col in df.columns else []
    if not vals:
        return "unset (assumption register not calibrated; decision D8)"
    return ";".join(vals)


def write_run_log(stage: str, outputs: list[str], status: str, extra: dict[str, Any] | None = None,
                  root: Path | None = None) -> Path:
    """Write one run log JSON and return its path. The file stem is the run-log ID."""
    root = root or find_root()
    from tzbank.ingest.manifest import input_hashes  # local import avoids a cycle

    now = dt.datetime.now()
    run_id = f"{now:%Y%m%dT%H%M%S}_{stage}"
    log = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "started": now.isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "tzbank_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": settings(root),
        "assumption_register_version": assumption_register_version(root),
        "input_sha256": input_hashes(root) if path("manifest", root).exists() else {},
        "outputs": outputs,
    }
    if extra:
        log.update(extra)
    out = path("run_logs", root) / f"{run_id}.json"
    n = 1
    while out.exists():
        out = path("run_logs", root) / f"{run_id}_{n}.json"
        n += 1
    # the latest log of each stage is also kept under run_logs/latest/ and committed, so the published dashboard
    # can show the run-log ID of what it displays without the timestamped history (prompt 26, D97)
    write_json(log, path("run_logs", root) / "latest" / f"{stage}.json")
    return write_json(log, out)


def latest_run_id(stage_prefix: str = "", root: Path | None = None) -> str | None:
    """Most recent run-log ID, optionally for one stage."""
    d = path("run_logs", root or find_root())
    logs = sorted(p.stem for p in d.glob("*.json") if not stage_prefix or f"_{stage_prefix}" in p.stem)
    return logs[-1] if logs else None
