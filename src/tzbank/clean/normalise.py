"""Normalise both sources to one canonical long table (prompt 01 sections 4.2-4.4; CLAUDE.md sections 3.4-3.6).

Money is full TZS, ratios are decimals, dates are real dates. Originals and restatements stay as separate rows with
``is_restated``; ``forecast_base_<date>`` marks the latest restatement eligible at each valuation date (D5).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.clean.crosswalk import inputs_map, load_crosswalk, phase1_map
from tzbank.clean.splits import adjust_long_values, load_corporate_actions
from tzbank.config import bank_basis, find_root, path, valuation_dates
from tzbank.ingest.inputs_workbook import load_interim
from tzbank.ingest.phase1 import phase1_is_restated
from tzbank.io import write_csv
from tzbank.pit.pubdates import phase1_unmapped_date

CANON_COLUMNS = [
    "bank_id", "period_end", "period_type", "fiscal_year", "canonical_name", "source_variable", "category", "entity",
    "basis", "basis_rule", "value", "value_text", "unit", "status", "doc_id", "source_document", "page_note",
    "publication_date", "provenance", "source_sheet", "source_row_id", "confidence", "definition", "notes",
    "is_restated", "evidence_class", "pubdate_basis",
]
BANK_SHEETS = ["Historical Financials", "Interim Results", "Quarterly Results", "Regulatory Capital", "Market Data"]
KEY = ["bank_id", "entity", "basis", "period_type", "period_end", "canonical_name", "category"]
# The bank's own reporting entities (consolidated group or bank solo); subsidiary segment rows are excluded from
# bank-level series.
PRIMARY_ENTITIES = ("Group", "Bank solo", "CRDB Bank (Tanzania, parent)")

_BANK_BASIS = re.compile(r"bank basis|bank-only|bank only|standalone|bank solo|bank-solo|capital - bank\b|\(bank\)",
                         re.I)
_GROUP_BASIS = re.compile(r"\bgroup\b", re.I)
_OBSERVED = {"REPORTED"}
_DERIVED = {"COMPUTED", "CALCULATED", "DERIVED"}


def evidence_from_status(status: pd.Series) -> pd.Series:
    """Blueprint §6: REPORTED -> Observed; COMPUTED/CALCULATED/DERIVED -> Derived; gaps keep Observed (the status
    column records that the value is missing)."""
    s = status.fillna("").str.upper()
    return np.where(s.isin(_DERIVED), "Derived", "Observed")


def phase1_basis(notes: pd.Series, source: pd.Series, bank_ids: pd.Series, default: dict[str, str]) -> pd.DataFrame:
    """Basis per Phase 1 row: stated in Notes/Source Document, else the bank's reporting basis (config/banks.yaml)."""
    text = notes.fillna("").astype(str) + " | " + source.fillna("").astype(str)
    bank = text.str.contains(_BANK_BASIS)
    group = text.str.contains(_GROUP_BASIS) & ~bank
    basis = np.where(bank, "BANK", np.where(group, "GROUP", bank_ids.map(default)))
    rule = np.where(bank | group, "stated", "bank_default")
    return pd.DataFrame({"basis": basis, "basis_rule": rule}, index=notes.index)


def canonical_phase1(p1: pd.DataFrame, cw: pd.DataFrame, source_map: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Phase 1 long -> canonical rows (retired RAW_DATA and excluded governance/disclosure items dropped)."""
    m = phase1_map(cw)
    p1 = p1.copy()
    p1["canonical_name"] = [m.get((s, v)) for s, v in zip(p1["sheet"], p1["variable"], strict=False)]
    p1 = p1[p1["canonical_name"].notna()].copy()
    b = phase1_basis(p1["notes"], p1["source_document"], p1["bank_id"], bank_basis(root))
    sm = source_map.rename(columns={"Bank ID": "bank_id", "Phase 1 Source Label": "source_document"})
    sm = sm[["bank_id", "source_document", "Mapped Doc ID", "publication_date"]].drop_duplicates(
        ["bank_id", "source_document"])
    p1 = p1.merge(sm, on=["bank_id", "source_document"], how="left")
    out = pd.DataFrame({
        "bank_id": p1["bank_id"],
        "period_end": pd.to_datetime(p1["year"].astype(str) + "-12-31"),
        "period_type": "FY",
        "fiscal_year": p1["year"].astype("Int64"),
        "canonical_name": p1["canonical_name"],
        "source_variable": p1["variable"],
        "category": p1["category"],
        "entity": np.where(b["basis"].to_numpy() == "GROUP", "Group", "Bank solo"),
        "basis": b["basis"].to_numpy(),
        "basis_rule": b["basis_rule"].to_numpy(),
        "value": p1["value"],
        "value_text": p1["value_raw"].where(p1["value"].isna()),
        "unit": p1["unit"].replace({"TZS": "TZS", "shares": "shares"}),
        "status": p1["status"],
        "doc_id": p1["Mapped Doc ID"],
        "source_document": p1["source_document"],
        "page_note": p1["page_table"],
        "publication_date": pd.to_datetime(p1["publication_date"]).fillna(
            p1["year"].map(lambda y: phase1_unmapped_date(int(y)) if pd.notna(y) else pd.NaT)),
        "provenance": "phase1",
        "source_sheet": p1["sheet"],
        "source_row_id": p1["sheet"] + ":" + p1["row_number"].astype(str),
        "confidence": p1["confidence"],
        "definition": p1["definition"],
        "notes": p1["notes"],
        "pubdate_basis": np.where(p1["publication_date"].notna(), "document", "phase1_unmapped"),
        "is_restated": [phase1_is_restated(s, n) for s, n in zip(p1["source_document"], p1["notes"], strict=False)],
    })
    out["evidence_class"] = evidence_from_status(out["status"])
    return out


def canonical_inputs(sheets: dict[str, pd.DataFrame], cw: pd.DataFrame) -> pd.DataFrame:
    """Inputs-workbook bank sheets -> canonical rows."""
    m = inputs_map(cw)
    parts = []
    for sheet in BANK_SHEETS:
        df = sheets[sheet]
        canon = [m.get((sheet, v)) for v in df["Variable"]]
        parts.append(pd.DataFrame({
            "bank_id": df["Bank ID"],
            "period_end": df["Period End"],
            "period_type": df["Period Type"],
            "fiscal_year": df["Period End"].dt.year.astype("Int64"),
            "canonical_name": canon,
            "source_variable": df["Variable"],
            "category": None,
            "entity": df["Entity"],
            "basis": df["Basis"].str.upper(),
            "basis_rule": "stated",
            "value": df["value_numeric"],
            "value_text": df["value_raw"].where(df["value_numeric"].isna()),
            "unit": df["Unit"],
            "status": df["Status"].str.upper(),
            "doc_id": df["Doc ID"],
            "source_document": df["Doc ID"],
            "page_note": df.get("Page / Note"),
            "publication_date": df["publication_date"],
            "provenance": "inputs_workbook",
            "source_sheet": sheet,
            "source_row_id": df["Row ID"],
            "confidence": df.get("Confidence"),
            "definition": None,
            "notes": df.get("Notes"),
            "is_restated": False,
            "pubdate_basis": "document",
        }))
    out = pd.concat(parts, ignore_index=True)
    out = flag_restatements(out)
    out["evidence_class"] = evidence_from_status(out["status"])
    return out


def flag_restatements(df: pd.DataFrame) -> pd.DataFrame:
    """Within each key, the earliest-published row is the original; later rows (or rows noted RESTATED) are
    restatements. CLAUDE.md §3.6; D21 tie-break: on equal publication dates the row from the report for that
    fiscal year (Doc ID ending FY<year>) is the original."""
    df = df.copy()
    k = [c for c in KEY if c in df.columns]
    own = [isinstance(d, str) and pd.notna(y) and d.endswith(f"FY{int(y)}")
           for d, y in zip(df["doc_id"], df["fiscal_year"], strict=False)]
    df["_not_own_year"] = [not o for o in own]
    order = df.sort_values(["publication_date", "_not_own_year", "source_row_id"], na_position="last")
    rank = order.groupby(k, dropna=False).cumcount()
    df["vintage_rank"] = rank.reindex(df.index)
    noted = df["notes"].fillna("").str.upper().str.startswith("RESTATED")
    df["is_restated"] = (df["vintage_rank"] > 0) | noted
    return df.drop(columns=["vintage_rank", "_not_own_year"])


def forecast_base(df: pd.DataFrame, valuation_date: pd.Timestamp) -> pd.Series:
    """True for the latest-published row per key among rows eligible at ``valuation_date`` (decision D5)."""
    from tzbank.pit import eligible

    k = [c for c in KEY if c in df.columns]
    el = eligible(df, valuation_date, warn=False)
    el = el[el["value"].notna()]
    latest = el.sort_values(["publication_date", "source_row_id"]).groupby(k, dropna=False).tail(1)
    flag = pd.Series(False, index=df.index)
    flag[latest.index] = True
    return flag


def build_canonical(root: Path | None = None) -> pd.DataFrame:
    """Canonical long table from both sources (before precedence), with split-adjusted values."""
    root = root or find_root()
    cw = load_crosswalk(root)
    p1 = pd.read_csv(path("interim", root) / "phase1_long.csv", dtype={"bank_id": str, "value_raw": str},
                     low_memory=False)
    sheets = {s: load_interim(s, root) for s in BANK_SHEETS + ["Phase1 Source Map"]}
    canon = pd.concat([canonical_inputs(sheets, cw), canonical_phase1(p1, cw, sheets["Phase1 Source Map"], root)],
                      ignore_index=True)
    canon = canon[CANON_COLUMNS]
    canon = normalise_units(canon)
    canon["value_split_adj"] = adjust_long_values(canon, load_corporate_actions(root))
    for name, d in valuation_dates(root).items():
        canon[f"forecast_base_{name}"] = forecast_base(canon, d)
    return canon


def normalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise unit labels; flag ratio values typed as percentages (> 1.5) without changing them."""
    df = df.copy()
    u = df["unit"].fillna("").str.strip()
    df["unit"] = u.replace({"": None, "TZS mn": "TZS", "TZS m": "TZS"})
    ratio = df["unit"].eq("ratio")
    df["unit_flag"] = np.where(ratio & (df["value"].abs() > 1.5), "ratio_above_1.5_verify", None)
    return df


def write_canonical(root: Path | None = None) -> Path:
    """Write data/interim/canonical_long.csv."""
    root = root or find_root()
    return write_csv(build_canonical(root), path("interim", root) / "canonical_long.csv")


def load_canonical(root: Path | None = None, file: Path | None = None) -> pd.DataFrame:
    """Read canonical_long.csv (or bank_period_long.csv) with types restored."""
    root = root or find_root()
    f = file or path("interim", root) / "canonical_long.csv"
    df = pd.read_csv(f, dtype={"bank_id": str, "doc_id": str, "value_text": str, "category": str, "source_row_id": str},
                     low_memory=False)
    for c in ["period_end", "publication_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in [c for c in df.columns if c.startswith("forecast_base") or c == "is_restated"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    df["fiscal_year"] = df["fiscal_year"].astype("Int64")
    return df
