"""Phase 1 seven-bank panel ingest (prompt 01 section 3.1; blueprint section 6 data audit, 'Phase 1 panel v2').

Reads every ``RAW_*`` sheet of the read-only Phase 1 workbook into one long table. The workbook is opened read-only
and never written. Cell values are taken as typed (RAW_* sheets hold no formulas).
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.config import find_root, path

LONG_COLUMNS = ["bank_id", "year", "variable", "value", "unit", "status", "confidence", "source_document", "sheet",
                "category", "value_raw", "page_table", "definition", "as_of_date", "notes", "row_number"]

# Column header candidates per role (Phase 1 sheets differ slightly).
_ROLE = {
    "variable": ["Variable", "Ownership Category", "Disclosure Item"],
    "value": ["Reported Value", "Value (%)", "Value", "1=Disclosed/0=Not/NA"],
    "unit": ["Unit"],
    "status": ["Status", "Status (REPORTED/CALCULATED/DERIVED/NOT DISCLOSED/NOT FOUND/NOT APPLICABLE/CONFLICTING)"],
    "source_document": ["Source Document"],
    "confidence": ["Confidence"],
    "page_table": ["Page/Table"],
    "definition": ["Definition as Stated by Source"],
    "as_of_date": ["As-Of Date"],
    "notes": ["Notes"],
}
_DEFAULT_UNIT = {"RAW_OWNERSHIP": "ratio", "RAW_GOVERNANCE": "count", "RAW_DISCLOSURE": "binary"}
_STATUS_WORDS = {"MISSING", "NOT DISCLOSED", "NOT FOUND", "NOT APPLICABLE", "CONFLICTING", "TO COLLECT"}


def _sheet_frame(ws) -> pd.DataFrame:
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else f"_col{i}" for i, h in enumerate(rows[0])]
    df = pd.DataFrame(rows[1:], columns=header, dtype=object)
    df["row_number"] = range(2, len(df) + 2)
    return df


def _pick(df: pd.DataFrame, role: str) -> pd.Series:
    for c in _ROLE[role]:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def _extra_notes(df: pd.DataFrame) -> pd.Series:
    """Concatenate unnamed trailing columns (e.g. correction notes in RAW_CAPITAL col K) into Notes."""
    extra = [c for c in df.columns if c.startswith("_col") and c != "row_number"]
    out = []
    for i in df.index:
        parts = [df.at[i, c] for c in ["Notes", *extra] if c in df.columns and df.at[i, c] is not None]
        text = " ".join(str(p).strip() for p in parts if str(p).strip())
        out.append(text or None)
    return pd.Series(out, index=df.index, dtype=object)


def read_phase1(file: Path | None = None) -> pd.DataFrame:
    """Return all RAW_* sheets as one long table (Phase 1 columns, unvalidated). Blueprint §6, data layer."""
    file = file or path("phase1")
    wb = openpyxl.load_workbook(file, read_only=True, data_only=False)
    parts = []
    for ws in wb.worksheets:
        if not ws.title.startswith("RAW_"):
            continue
        df = _sheet_frame(ws)
        df = df[df["Bank ID"].astype(str).str.fullmatch(r"B\d{2}")].copy()
        out = pd.DataFrame(index=df.index)
        out["bank_id"] = df["Bank ID"].astype(str)
        out["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        var = _pick(df, "variable").astype(str).str.strip()
        out["variable"] = var
        out["category"] = None
        if ws.title == "RAW_OWNERSHIP":
            is_count = var.eq("Number of Shareholders")
            out["category"] = var.where(~is_count, None)
            out["variable"] = var.where(is_count, "Ownership Share")
        raw = _pick(df, "value")
        out["value_raw"] = raw.map(lambda v: None if v is None else str(v))
        out["value"] = pd.to_numeric(raw, errors="coerce")
        unit = _pick(df, "unit")
        out["unit"] = unit.where(unit.notna(), _DEFAULT_UNIT.get(ws.title))
        out.loc[out["variable"].eq("Number of Shareholders"), "unit"] = "count"
        status = _pick(df, "status").astype(object)
        # A status word typed in the value cell (and no status) becomes the status.
        raw_up = raw.astype(str).str.strip().str.upper()
        status = status.where(status.notna() | ~raw_up.isin(_STATUS_WORDS), raw_up)
        out["status"] = status.map(lambda s: None if s is None else str(s).strip().upper())
        for role in ["source_document", "confidence", "page_table", "definition", "as_of_date"]:
            out[role] = _pick(df, role).map(lambda v: None if v is None else str(v).strip())
        out["notes"] = _extra_notes(df)
        out["sheet"] = ws.title
        out["row_number"] = df["row_number"]
        parts.append(out)
    wb.close()
    long = pd.concat(parts, ignore_index=True)[LONG_COLUMNS]
    long["as_of_date"] = pd.to_datetime(long["as_of_date"], errors="coerce")
    return long


def read_bank_universe(file: Path | None = None) -> pd.DataFrame:
    """BANK_UNIVERSE sheet rows B01-B07 (the seven-bank sample)."""
    df = pd.read_excel(file or path("phase1"), sheet_name="BANK_UNIVERSE", dtype=object)
    return df[df["Bank ID"].astype(str).str.fullmatch(r"B0[1-7]")].reset_index(drop=True)


def read_version_log(file: Path | None = None) -> pd.DataFrame:
    """VERSION_LOG sheet (confirms the authoritative v2.0-disclosure version)."""
    return pd.read_excel(file or path("phase1"), sheet_name="VERSION_LOG", dtype=object)


_RESTATED = re.compile(r"\brestated\b", re.I)
_NOT_RESTATED = re.compile(r"\b(not|no)\s+restat", re.I)


def phase1_is_restated(source_document: str | None, notes: str | None) -> bool:
    """True when the Phase 1 source cites a restated comparative (e.g. 'AR 2011 ... 2009 comparative, Restated')."""
    s = source_document if isinstance(source_document, str) else ""
    return bool(_RESTATED.search(s)) and not _NOT_RESTATED.search(s)


def write_phase1_long(root: Path | None = None) -> Path:
    """Write data/interim/phase1_long.csv."""
    from tzbank.io import write_csv

    root = root or find_root()
    return write_csv(read_phase1(path("phase1", root)), path("interim", root) / "phase1_long.csv")
