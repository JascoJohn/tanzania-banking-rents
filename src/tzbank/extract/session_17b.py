"""D18 workbook session for edit pack A, item B3 (v0.19): CRDB bank-only profit before tax, FY2022-FY2025.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_17b
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

The workbook already holds CRDB's bank-only (parent) total assets, total equity and profit after tax for FY2022-FY2025
(subsidiary layer, spec section 4). Bank-only profit before tax was not held, so the chapter 1 profit share could not
use a bank-solo numerator (CRDB Group includes its Burundi and DRC subsidiaries, which the Tanzanian sector total
excludes). Values are transcribed from each year's own report (point in time), Bank column, TZS million; the owner's
figures from the published audited statements (FY2024 752,852; FY2025 1,000,650) agree.
"""

from __future__ import annotations

import datetime as dt

import openpyxl

from tzbank.config import find_root, path
from tzbank.extract.workbook import _append, _header, _next_id

VERSION = "v0.19"
VARIABLE = "Bank-only Profit Before Tax"
# (fiscal year, doc id, page, value TZS million, where)
ROWS = [
    (2022, "CRDB-AR-FY2022", 99, 477_405, "statement of cash flows, Bank column (profit before income tax)"),
    (2023, "CRDB-AR-FY2023", 124, 585_214, "statement of profit or loss, Bank column"),
    (2024, "CRDB-AR-FY2024", 169, 752_852, "statement of profit or loss, Bank column"),
    (2025, "CRDB-AR-FY2025", 195, 1_000_650, "statement of profit or loss, Bank column"),
]


def run() -> dict[str, int]:
    root = find_root()
    f = path("inputs_workbook", root)
    wb = openpyxl.load_workbook(f)
    hf = wb["Historical Financials"]
    h = _header(hf)
    held = {(hf.cell(r, h["Bank ID"]).value, getattr(hf.cell(r, h["Period End"]).value, "year", None),
             hf.cell(r, h["Variable"]).value) for r in range(2, hf.max_row + 1)}
    added = 0
    for fy, doc, page, value, where in ROWS:
        if ("B01", fy, VARIABLE) in held:
            continue
        _append(hf, {"Row ID": f"HF-{_next_id(hf, 'HF'):04d}", "Bank ID": "B01", "Bank": "CRDB Bank",
                     "Entity": "CRDB Bank (Tanzania, parent)", "Basis": "BANK", "Period Type": "FY",
                     "Period End": dt.datetime(fy, 12, 31), "Variable": VARIABLE, "Value": float(value) * 1e6,
                     "Unit": "TZS", "Status": "REPORTED", "Doc ID": doc, "Page / Note": f"PDF p.{page} {where}",
                     "Confidence": "HIGH",
                     "Notes": "Subsidiary layer (spec section 4) | transcribed, edit pack A item B3 (owner figures "
                              "for FY2024-FY2025 agree) | extraction_method=transcribed"})
        added += 1
    me = wb["Read Me"]
    r = me.max_row + 2
    for line in (f"{VERSION} (2026-11-14, edit pack A B3, D18 working workbook)",
                 f"Historical Financials: CRDB {VARIABLE} FY2022-FY2025 ({added} rows) from each year's report, Bank "
                 "column, so the chapter 1 profit share can use a bank-solo CRDB numerator (D82)."):
        me.cell(r, 1).value = line
        r += 1
    wb.save(f)
    return {"Historical Financials": added}


if __name__ == "__main__":
    print(run())
