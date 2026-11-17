"""Step 20: build, recalculate and check the Excel companion model (D104).

Writes:

- ``outputs/valuation/Tanzania_Banking_Rents_Model.xlsx``: the recalculated workbook (formulas with cached values);
- ``outputs/valuation/excel_model_check.csv``: item, python, excel, diff, pass for every live key output, plus one row
  per error type (``#REF!``, ``#DIV/0!``, ``#VALUE!``, ``#NAME?``, ``#N/A``, ``#NUM!``) with its count;
- ``outputs/valuation/excel_model_sheets.csv``: live (formula) and static cells per sheet;
- ``outputs/valuation/excel_model_figures.csv``: every report figure (number, chapter sheet, native / heatmap / image,
  source range).

The stage fails (``ExcelModelCheckError``) if any key item fails or any formula returns an error. Tolerance: 0.01%
relative, or TZS 0.01 for per-share values; text items (ratings, guard statuses) must match exactly. Without a
spreadsheet engine (no Excel, no LibreOffice) the workbook is written unrecalculated and every check row reads
NOT RUN; the stage then warns instead of failing.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from tzbank.config import find_root, path
from tzbank.excel.chapters import TOL_REL
from tzbank.excel.model import TITLE, build_model
from tzbank.excel.recalc import RecalcUnavailable, normalise, recalc
from tzbank.io import write_csv

log = logging.getLogger("tzbank")
OUT_NAME = "Tanzania_Banking_Rents_Model.xlsx"
ERRORS = ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!")
AUTHOR = "Jasco John"


class ExcelModelCheckError(RuntimeError):
    """A live workbook value differs from the Python output beyond tolerance, or a formula returns an error."""


def _ok(kind: str, py, xv) -> tuple[bool, float | None]:
    if kind == "text":
        return str(py) == str(xv), None
    if not isinstance(xv, (int, float)) or isinstance(xv, bool):
        return False, None
    py = float(py)
    diff = float(xv) - py
    if not math.isfinite(diff):
        return False, diff
    rel = abs(diff) / abs(py) if py else (0.0 if diff == 0 else math.inf)
    if kind == "ps":
        return abs(diff) <= 0.01 or rel <= TOL_REL, diff
    if kind == "abs":
        return abs(diff) <= 1e-8, diff
    return rel <= TOL_REL or abs(diff) <= 1e-9, diff


def compare(xlsx: Path, items: list[dict]) -> tuple[pd.DataFrame, dict]:
    wb = load_workbook(xlsx, data_only=True)
    rows = []
    for it in items:
        xv = wb[it["sheet"]][it["cell"]].value
        ok, diff = _ok(it["kind"], it["python"], xv)
        rows.append({"item": it["item"], "sheet": it["sheet"], "cell": it["cell"], "python": it["python"],
                     "excel": xv, "diff": diff, "tolerance": {"ps": "TZS 0.01 or 0.01%", "abs": "1e-8 absolute",
                                                             "text": "exact"}.get(it["kind"], "0.01%"),
                     "pass": bool(ok)})
    errs = {e: 0 for e in ERRORS}
    where: dict[str, str] = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value in errs:
                    errs[c.value] += 1
                    where.setdefault(c.value, f"{ws.title}!{c.coordinate}")
    return pd.DataFrame(rows), {"counts": errs, "first": where}


def write_excel_model(root: Path | None = None, engine: str = "auto") -> list[Path]:
    root = root or find_root()
    vdir = path("outputs", root) / "valuation"
    vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / OUT_NAME
    wb, checks, live, figs = build_model(root)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "model_unrecalculated.xlsx"
        wb.save(raw)
        try:
            used = recalc(raw, out, engine)
        except RecalcUnavailable as e:
            used = None
            shutil.copyfile(raw, out)
            log.warning("excel_model: no spreadsheet engine (%s); workbook written without recalculation, checks "
                        "NOT RUN", e)
    normalise(out, TITLE, AUTHOR)
    sheets = pd.DataFrame([{"sheet": s, "live_formula_cells": live.live.get(s, 0),
                            "static_cells": live.static.get(s, 0), "evidence_class": "Derived"}
                           for s in dict.fromkeys([*live.live, *live.static])])
    files = [out, write_csv(sheets, vdir / "excel_model_sheets.csv", require_evidence=True)]
    if figs:  # the report-figure map (owner review of step 20)
        files.append(write_csv(pd.DataFrame(figs), vdir / "excel_model_figures.csv", require_evidence=True))
    if used is None:
        chk = pd.DataFrame([{**it, "excel": None, "diff": None, "tolerance": "", "pass": "NOT RUN"}
                            for it in checks.items])
        chk["evidence_class"] = "Derived"
        files.append(write_csv(chk, vdir / "excel_model_check.csv", require_evidence=True))
        return files
    chk, errs = compare(out, checks.items)
    for e, n in errs["counts"].items():
        chk.loc[len(chk)] = {"item": f"formula errors {e}", "sheet": errs["first"].get(e, ""), "cell": "",
                             "python": 0, "excel": n, "diff": n, "tolerance": "0", "pass": n == 0}
    chk["evidence_class"] = "Derived"
    chk["engine"] = used
    files.append(write_csv(chk, vdir / "excel_model_check.csv", require_evidence=True))
    bad = chk[~chk["pass"].astype(bool)]
    log.info("excel_model: %d items checked with %s, %d failed", len(chk), used, len(bad))
    if len(bad):
        raise ExcelModelCheckError(f"{len(bad)} of {len(chk)} workbook checks failed, e.g. "
                                   + "; ".join(f"{r.item} (python {r.python}, excel {r.excel})"
                                               for r in bad.head(5).itertuples())
                                   + f" (all in {files[-1].name})")
    return files
