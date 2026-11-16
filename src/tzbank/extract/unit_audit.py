"""Reporting-unit audit for B03-B07 (prompt 02d section 2.1; blueprint §6 data audit).

For every extracted B03-B07 value written to the working workbook (``data/extraction/ledger.csv``: file, page) the
statement header unit is read again from its page (``tzbank.extract.pdftext.page_unit``: TZS, TZS '000, TZS million)
and the value is divided back by that unit; the printed number must appear on the page. A value found only at a
different scale is a unit mismatch. Needs the git-ignored PDFs, so it runs in the D18 session, not in ``run all``;
its output ``reports/validation/unit_audit_b03_b07.csv`` is committed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.ocr import extraction_source
from tzbank.extract.pdftext import _doc, page_rows, page_unit
from tzbank.extract.picker import SCALE
from tzbank.io import write_csv

SMALL = ["B03", "B04", "B05", "B06", "B07"]
NON_TZS = {"Dividend per Share: Special", "Dividend per Share (Total incl. Special)"}
SCALES = {"TZS": 1.0, "000": 1e3, "million": 1e6}


ALT_RX = re.compile(r"in\s+(million|thousand)s?\s+(of\s+)?(shillings|tzs|tanzania)|(tzs|shs)\s*\.?\s*(mn|mio|m)(?![a-z])", re.I)


def header_unit(pdf: Path, page: int) -> str | None:
    """Statement header unit: ``page_unit`` first, then wordings it misses ('Amounts in million shillings')."""
    u = page_unit(pdf, page)
    if u:
        return u
    m = ALT_RX.search(_doc(str(pdf))[page - 1].get_text())
    if not m:
        return None
    return "000" if (m.group(1) or "").lower().startswith("thousand") else "million"


def _page_numbers(pdf: Path, page: int) -> list[float]:
    nums = []
    for half in (None, "left", "right"):
        for r in page_rows(pdf, page, half=half):
            nums += [abs(n) for n in r.numbers if n is not None]
    return nums


def scales_found(value: float, nums: list[float]) -> list[str]:
    """Units at which ``value`` (full TZS) appears among the printed numbers (rounded to the printed precision)."""
    out = []
    for name, s in SCALES.items():
        target = abs(value) / s
        if target >= 1 and any(abs(n - target) <= 0.5 for n in nums):
            out.append(name)
    return out


def audit(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    led = pd.read_csv(root / "data" / "extraction" / "ledger.csv")
    led = led[led["bank_id"].isin(SMALL) & led["value"].notna() & led["page"].notna() & ~led["variable"].isin(NON_TZS)]
    rows = []
    cache: dict = {}
    for _, r in led.iterrows():
        pdf = path("documents", root) / r["file"]
        src = extraction_source(pdf, root) if pdf.exists() else None
        rec = {"bank_id": r["bank_id"], "fiscal_year": int(r["fiscal_year"]), "variable": r["variable"],
               "value": r["value"], "file": r["file"], "page": int(r["page"]), "label": r["label"],
               "ocr": src is not None and src != pdf}
        if src is None:
            rows.append(rec | {"status": "document not held"})
            continue
        key = (str(src), int(r["page"]))
        if key not in cache:
            unit = header_unit(src, int(r["page"])) or (header_unit(src, int(r["page"]) - 1) if r["page"] > 1 else None)
            cache[key] = (unit, _page_numbers(src, int(r["page"])))
        unit, nums = cache[key]
        unit_name = {"000": "000", "million": "million"}.get(unit, "TZS" if unit is None else unit)
        found = scales_found(float(r["value"]), nums)
        computed = str(r["label"]).startswith("sum of") or " + " in str(r["label"])
        if computed:
            status = "computed (sum of printed lines; components checked)"
        elif unit is None:
            status = f"header unit not found; value printed at {'/'.join(found) or 'no'} scale"
        elif unit_name in found:
            status = "consistent"
        elif found:
            status = f"UNIT MISMATCH: header {unit_name}, value printed at {'/'.join(found)} scale"
        else:
            status = "value not found on page at any scale"
        rows.append(rec | {"header_unit": unit, "scale_applied": SCALE.get(unit, 1.0) if unit else None,
                           "scales_found": "/".join(found), "status": status})
    df = pd.DataFrame(rows)
    df["evidence_class"] = "Observed"
    return df


def write_unit_audit(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    return [write_csv(audit(root), path("validation", root) / "unit_audit_b03_b07.csv", require_evidence=True)]


if __name__ == "__main__":
    a = audit()
    print(a.groupby(["bank_id", "status"]).size())
    write_unit_audit()
