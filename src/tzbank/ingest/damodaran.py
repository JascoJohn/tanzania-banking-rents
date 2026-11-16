"""Damodaran archived datasets -> data/interim/damodaran_archive.csv (blueprint §2 historical COE; prompt 05 fix 3).

Files (git-ignored, hashed in MANIFEST) in ``data/raw/documents/damodaran/``:
- ``ctrypremYY``: country risk premiums; suffix YY = data at end-YY, published January YY+1. Tanzania rows: rating-based
  default spread, total ERP and CRP from YY = 18 (Moody's B1/B2); PRS-based estimates for YY = 15-17; not covered
  for YY <= 14 (unrated, no estimate).
- ``betaemergYY``: emerging-market industry betas, same suffix convention; row "Banks (Regional)" (the row the
  inputs workbook uses for the January 2025 value, MA-0021).
- ``histimpl.xls``: implied ERP history; "Implied ERP (FCFE)" for year Y is the start-of-year Y+1 value (public in
  January Y+1).

Point in time: fiscal year t (31 December) uses the version public at that date, i.e. suffix t-1 (January t release);
the mid-year (July) updates are not archived. If the documents are absent (fresh clone), the committed interim file
is kept.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

log = logging.getLogger("tzbank")
COLUMNS = ["input", "fiscal_year", "value", "version", "published", "file", "sheet", "row_label", "basis",
           "evidence_class"]


def _ctry(f: Path, yy: int) -> list[dict]:
    xl = pd.ExcelFile(f)
    if "ERPs by country" not in xl.sheet_names:
        return []
    df = xl.parse("ERPs by country", header=None).fillna("")
    s = df.astype(str)
    hit = [i for i in range(len(df)) if (s.iloc[i] == "Tanzania").any()]
    if not hit:
        return []
    r = hit[0]
    h = next(i for i in range(r - 1, -1, -1) if s.iloc[i].str.strip().eq("Country").any())
    c0 = list(s.iloc[r]).index("Tanzania")
    head = [x.strip() for x in list(s.iloc[h])[c0:]]
    vals = list(df.iloc[r])[c0:]

    def get(*names):
        for n in names:
            if n in head:
                return vals[head.index(n)]
        return None

    if "Rating-based Default Spread" in head:
        ds, erp, crp, basis = (get("Rating-based Default Spread"), get("Total Equity Risk Premium"),
                               get("Country Risk Premium"), "Moody's " + str(get("Moody's rating")))
    else:
        ds, erp, crp, basis = get("Default Spread"), get("ERP"), get("CRP"), \
            f"PRS-based (unrated; PRS score {get('PRS Composite Risk Score')})"
    out = []
    for k, v in (("ds", ds), ("crp", crp), ("total_erp", erp)):
        if isinstance(v, (int, float)) and v != "":
            out.append({"input": k, "fiscal_year": 2000 + yy + 1, "value": float(v), "version": f"ctryprem{yy:02d}",
                        "published": f"{2000 + yy + 1}-01", "file": f"documents/damodaran/{f.name}",
                        "sheet": "ERPs by country", "row_label": "Tanzania", "basis": basis})
    return out


def _beta(f: Path, yy: int) -> list[dict]:
    xl = pd.ExcelFile(f)
    for sh in xl.sheet_names:
        b = xl.parse(sh, header=None).fillna("")
        s = b.astype(str)
        for i in range(len(b)):
            if s.iloc[i, 0].strip() == "Banks (Regional)":
                return [{"input": "beta", "fiscal_year": 2000 + yy + 1, "value": float(b.iloc[i, 2]),
                         "version": f"betaemerg{yy:02d}", "published": f"{2000 + yy + 1}-01",
                         "file": f"documents/damodaran/{f.name}", "sheet": sh, "row_label": "Banks (Regional)",
                         "basis": f"levered beta, {int(b.iloc[i, 1])} firms"}]
    return []


def _erp(f: Path) -> list[dict]:
    df = pd.read_excel(f, header=None).fillna("")
    s = df.astype(str)
    h = next(i for i in range(len(df)) if s.iloc[i, 0].strip() == "Year")
    c = [x.strip() for x in s.iloc[h]].index("Implied ERP (FCFE)")
    out = []
    for i in range(h + 1, len(df)):
        y = df.iloc[i, 0]
        if isinstance(y, (int, float)) and y != "" and 2006 <= int(y) <= 2030 and df.iloc[i, c] != "":
            out.append({"input": "erp", "fiscal_year": int(y) + 1, "value": float(df.iloc[i, c]),
                        "version": f"histimpl (end-{int(y)} value = start of {int(y) + 1})",
                        "published": f"{int(y) + 1}-01", "file": f"documents/damodaran/{f.name}",
                        "sheet": "Historical Impl Premiums", "row_label": str(int(y)), "basis": "Implied ERP (FCFE)"})
    return out


def write_damodaran_archive(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    d = path("documents", root) / "damodaran"
    out = path("interim", root) / "damodaran_archive.csv"
    if not d.exists() or not any(d.iterdir()):
        log.info("damodaran: documents not present; keeping %s", out.name)
        return [out] if out.exists() else []
    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in sorted(d.glob("ctryprem*")):
            rows += _ctry(f, int(f.stem[-2:]))
        for f in sorted(d.glob("betaemerg*")):
            rows += _beta(f, int(f.stem[-2:]))
        if (d / "histimpl.xls").exists():
            rows += _erp(d / "histimpl.xls")
    df = pd.DataFrame(rows, columns=COLUMNS[:-1]).assign(evidence_class="Observed")
    return [write_csv(df.sort_values(["input", "fiscal_year"]).reset_index(drop=True), out, require_evidence=True)]
