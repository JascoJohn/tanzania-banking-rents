"""D18 workbook session for prompt 02f (v0.13): small fixes carried from prompt 02e (section 4).

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_02f
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

1. CRDB FY2025: the three rows with general citations (weighted and year-end share counts, dividend per share) point
   to the pages that print them in the full report, re-checked here.
2. MKCB FY2021: the original publication's profit before tax (condensed statement line 8, 'Operating Income/(Loss)',
   with a nil income tax provision on line 9) as its own row next to the FY2022-report restatement.
"""

from __future__ import annotations

import openpyxl
import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.session_02e import CRDB25, printed

VERSION = "v0.13"
CITATIONS = {  # Row ID -> (variable, pages to check with the printed values, new citation)
    "HF-0143": ("Weighted Average Ordinary Shares", [(292, [2_612]), (325, [2_611_838_584])],
                "PDF p.292 Note 25 (EPS: number of shares 2,612 million); exact count PDF p.325 share-capital note"),
    "HF-0727": ("Ordinary Shares Outstanding (Year End)", [(325, [2_611_838_584]), (134, [2_611_838_584])],
                "PDF p.325 share-capital note (issued and fully paid 2,611,838,584 shares of TZS 25); p.134"),
    "HF-0726": ("Dividend per Share", [(292, [90, 235_065]), (188, [90])],
                "PDF p.292 Note 25 dividends (TZS 90 per share, TZS 235,065 million); p.188 directors' report"),
}


def run() -> dict[str, int]:
    root = find_root()
    for rid, (_var, checks, _cite) in CITATIONS.items():
        for page, vals in checks:
            if not printed(CRDB25, page, vals, root):
                raise ValueError(f"{rid}: values not printed on CRDB FY2025 p.{page}")
    if not printed("MKCB/2021.pdf", 1, [1_148, 3_849], root):
        raise ValueError("MKCB FY2021 operating income not printed")
    led = pd.DataFrame([{
        "bank_id": "B05", "fiscal_year": 2021, "variable": "Profit Before Tax", "value": 1_148e6,
        "file": "MKCB/2021.pdf", "page": 1, "label": "Operating Income/(Loss), current year cumulative (line 8)",
        "column": "current", "gate": "line 8 = line 10 net income 1,148 with nil income tax provision (line 9)",
        "gate_passed": True, "confidence": "HIGH", "row_type": "original", "ocr": True,
        "extraction_method": "gated", "doc_id": "MKCB-FSPUB-FY2021",
        "notes": "Original Q4 2021 BoT publication (prompt 02f section 4); the FY2022 report restates FY2021 PBT to "
                 "(2,706,838)k (restatement row)"}])
    # workbook.update() treats a restated key as already held, so the original row is appended directly (run once)
    counts = {"original_rows_appended": 1}
    wb = openpyxl.load_workbook(path("inputs_workbook", root))
    hf, me = wb["Historical Financials"], wb["Read Me"]
    h = workbook._header(hf)
    fixed = 0
    for r in range(2, workbook._last_row(hf) + 1):
        rid = hf.cell(r, 1).value
        if rid in CITATIONS:
            var, _checks, cite = CITATIONS[rid]
            if hf.cell(r, h["Variable"]).value != var:
                raise ValueError(f"{rid} is not {var}")
            old = hf.cell(r, h["Page / Note"]).value
            hf.cell(r, h["Page / Note"]).value = cite
            hf.cell(r, h["Notes"]).value = f"Citation pointed to pages (prompt 02f; was '{old}') | " + str(
                hf.cell(r, h["Notes"]).value or "")
            fixed += 1
    lines = [f"{VERSION} (2026-11-07, prompt 02f, D18 working workbook)",
             f"Historical Financials: {fixed} CRDB FY2025 citations pointed to printed pages (shares, DPS); MKCB "
             "FY2021 original profit before tax (1,148m, condensed statement line 8) added next to the restatement."]
    r = me.max_row + 2
    for line in lines:
        me.cell(r, 1).value = line
        r += 1
    wb.save(path("inputs_workbook", root))
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp)
    pd.concat([old, led.reindex(columns=old.columns).assign(evidence_class="Observed")], ignore_index=True).to_csv(
        lp, index=False, lineterminator="\n")
    return counts | {"citations_fixed": fixed}


if __name__ == "__main__":
    print(run())
