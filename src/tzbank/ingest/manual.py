"""Manual entries (decision D29; prompt 02b section 1.4).

Values read by a person from a held document go into ``data/raw/manual_entries/manual_values.csv``. ``run all``
validates the file and writes ``data/interim/manual_values.csv`` plus ``reports/validation/manual_entry_issues.csv``.
In a D18 workbook session the extraction ledger picks the rows up (``extraction_method=manual``) and the same writer
puts them into the working workbook. An invalid row is never written; a missing source or page is critical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.clean.crosswalk import load_crosswalk
from tzbank.config import find_root, path
from tzbank.io import write_csv

MANUAL_COLUMNS = ["bank_id", "fiscal_year", "period_type", "variable", "value", "unit", "basis", "file", "page",
                  "as_reported_label", "checked_by", "checked_on", "confidence", "notes"]
UNITS = {"TZS", "TZS '000", "TZS million", "TZS billion", "ratio", "TZS per share", "shares", "count"}
SCALE = {"TZS": 1.0, "TZS '000": 1e3, "TZS million": 1e6, "TZS billion": 1e9}


def manual_file(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / "manual_entries" / "manual_values.csv"


def ensure_file(root: Path | None = None) -> Path:
    """Create the (empty) manual-entry file with its header if it does not exist."""
    f = manual_file(root)
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(f, index=False, lineterminator="\n")
    return f


def read_manual(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(valid rows in full TZS / ratio, issues)."""
    root = root or find_root()
    df = pd.read_csv(ensure_file(root), dtype=str, keep_default_na=False).replace({"": None})
    issues = []
    if list(df.columns) != MANUAL_COLUMNS:
        issues.append({"row": None, "column": None, "issue": f"header differs: {list(df.columns)}", "critical": True})
        return pd.DataFrame(columns=MANUAL_COLUMNS), pd.DataFrame(issues)
    known = set(load_crosswalk(root)["inputs_name"]) - {""}

    def blank(v: object) -> bool:
        return v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip()

    ok = []
    for i, r in df.iterrows():
        n = i + 2
        problems = []
        if blank(r["bank_id"]) or not str(r["bank_id"]).startswith("B0"):
            problems.append(("bank_id", "unknown bank", True))
        if r["variable"] not in known:
            problems.append(("variable", f"not an inputs-workbook variable in the crosswalk: {r['variable']!r}", True))
        v = pd.to_numeric(r["value"], errors="coerce")
        if pd.isna(v):
            problems.append(("value", "not numeric", True))
        if r["unit"] not in UNITS:
            problems.append(("unit", f"unit must be one of {sorted(UNITS)}", True))
        if blank(r["file"]) or blank(r["page"]):
            problems.append(("file/page", "value with no source page", True))
        if blank(r["checked_by"]):
            problems.append(("checked_by", "who read the value is required", True))
        if r["unit"] == "ratio" and pd.notna(v) and abs(v) > 1.5:
            problems.append(("value", "ratio above 1.5: entered as a percentage?", False))
        for col, msg, crit in problems:
            issues.append({"row": n, "column": col, "issue": msg, "critical": crit})
        if not any(c for _, _, c in problems):
            ok.append(r.to_dict() | {"value_full": float(v) * SCALE.get(r["unit"], 1.0)})
    valid = pd.DataFrame(ok, columns=[*MANUAL_COLUMNS, "value_full"])
    valid["extraction_method"] = "manual"
    valid["evidence_class"] = "Observed"
    return valid, pd.DataFrame(issues, columns=["row", "column", "issue", "critical"])


def write_manual(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    valid, issues = read_manual(root)
    return [write_csv(valid, path("interim", root) / "manual_values.csv"),
            write_csv(issues, path("validation", root) / "manual_entry_issues.csv")]
