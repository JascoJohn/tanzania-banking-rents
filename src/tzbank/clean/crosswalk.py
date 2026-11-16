"""Variable crosswalk (prompt 01 section 4.1): config/variable_crosswalk.csv maps every source variable name to one
canonical name, or lists it as deliberately excluded with a reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path

CROSSWALK_COLUMNS = ["canonical_name", "phase1_name", "inputs_name", "unit", "basis", "definition",
                     "blueprint_chapters", "evidence_class", "source", "source_sheets", "status", "exclusion_reason"]


def load_crosswalk(root: Path | None = None) -> pd.DataFrame:
    """Read the crosswalk with empty strings for blanks."""
    df = pd.read_csv(path("crosswalk", root or find_root()), dtype=str).fillna("")
    missing = set(CROSSWALK_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"crosswalk missing columns: {sorted(missing)}")
    return df


def phase1_map(cw: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(sheet, Phase 1 variable) -> canonical name, for mapped rows."""
    m = cw[(cw["source"] == "phase1") & (cw["status"] == "mapped")]
    return {(s, n): c for s, n, c in zip(m["source_sheets"], m["phase1_name"], m["canonical_name"], strict=False)}


def inputs_map(cw: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(sheet, inputs-workbook variable) -> canonical name, for mapped rows."""
    m = cw[(cw["source"] == "inputs_workbook") & (cw["status"] == "mapped")]
    return {(s, n): c for s, n, c in zip(m["source_sheets"], m["inputs_name"], m["canonical_name"], strict=False)}


def unmapped(cw: pd.DataFrame, phase1_long: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Source variables that are neither mapped nor listed as excluded (must be empty)."""
    known_p1 = set(zip(cw.loc[cw["source"] == "phase1", "source_sheets"], cw.loc[cw["source"] == "phase1", "phase1_name"],
                       strict=False))
    known_iw = set(zip(cw.loc[cw["source"] == "inputs_workbook", "source_sheets"],
                       cw.loc[cw["source"] == "inputs_workbook", "inputs_name"], strict=False))
    rows = []
    for sheet, var in sorted(set(zip(phase1_long["sheet"], phase1_long["variable"], strict=False))):
        if (sheet, var) not in known_p1:
            rows.append({"source": "phase1", "sheet": sheet, "variable": var})
    for sheet, df in inputs.items():
        if "Variable" not in df.columns:
            continue
        for var in sorted(df["Variable"].dropna().astype(str).unique()):
            if (sheet, var) not in known_iw:
                rows.append({"source": "inputs_workbook", "sheet": sheet, "variable": var})
    return pd.DataFrame(rows, columns=["source", "sheet", "variable"])


def chapters_of(cw: pd.DataFrame) -> dict[str, set[int]]:
    """canonical_name -> set of blueprint chapters that need it."""
    out: dict[str, set[int]] = {}
    for c, ch in zip(cw["canonical_name"], cw["blueprint_chapters"], strict=False):
        if not c:
            continue
        out.setdefault(c, set()).update(int(x) for x in ch.split(";") if x.strip())
    return out
