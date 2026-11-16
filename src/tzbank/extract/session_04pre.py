"""D18 workbook session for the prompt 04 pre-flight (v0.16): staff-cost corrections from the 03b review.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_04pre
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

The owner flagged CRDB FY2023 staff costs (TZS 217.9bn) as low. The check found two extraction errors in the prompt
03a staff-cost lines (``detail_lines.staff_costs``):

1. CRDB FY2013, FY2014, FY2016, FY2023: the extractor took "Salaries and wages", the first line of the employee
   benefit note, instead of the note total (Group column). Corrected to the printed Group note total.
2. NMB FY2017-19, FY2021, FY2023-25: the extractor took "Salaries and other benefits" from the value-added statement,
   which reports PAYE separately under taxes and so understates the expense. Corrected to the income-statement line
   "Employee benefits expense" (Group; the note is headed "Group and Bank"). NMB FY2014: the held value (69,036m) was
   the income tax expense; corrected to the employee benefit note total.

Every new value is re-checked on its page with ``printed``. The extractor regex no longer accepts the component and
value-added labels (``detail_lines.STAFF_RX``).
"""

from __future__ import annotations

from tzbank.config import find_root
from tzbank.extract import workbook
from tzbank.extract.session_02e import printed

VERSION = "v0.16"
M = 1_000_000
# (bank, fiscal year, held value TZS m, corrected value TZS m, file, page, doc id, reason)
FIXES = [
    ("B01", 2013, 59_804, 91_462, "CRDB/2013.pdf", 142, "CRDB-AR-FY2013", "salaries-and-wages line, not note total"),
    ("B01", 2014, 80_527, 120_949, "CRDB/2014.pdf", 137, "CRDB-AR-FY2014", "salaries-and-wages line, not note total"),
    ("B01", 2016, 119_274, 184_370, "CRDB/Annual-Report-2016-.pdf", 201, "CRDB-AR-FY2016",
     "salaries-and-wages line, not note total"),
    ("B01", 2023, 217_931, 338_243, "CRDB/2023 CRDB BANK ANNUAL REPORT.pdf", 301, "CRDB-AR-FY2023",
     "salaries-and-wages line, not note total (note 23, printed p.354)"),
    ("B02", 2014, 69_036, 116_693, "NMB/2014.pdf", 52, "NMB-AR-FY2014", "income tax expense taken in error"),
    ("B02", 2017, 116_299, 154_686, "NMB/2017.pdf", 77, "NMB-AR-FY2017", "value-added statement figure excl. PAYE"),
    ("B02", 2018, 124_228, 166_149, "NMB/2018.pdf", 101, "NMB-AR-FY2018", "value-added statement figure excl. PAYE"),
    ("B02", 2019, 137_910, 182_579, "NMB/2019.pdf", 95, "NMB-AR-FY2019", "value-added statement figure excl. PAYE"),
    ("B02", 2021, 186_673, 243_679, "NMB/2021.pdf", 62, "NMB-AR-FY2021", "value-added statement figure excl. PAYE"),
    ("B02", 2023, 227_450, 297_923, "NMB/NMB 2023 Integrated.pdf", 140, "NMB-AR-FY2023",
     "value-added statement figure excl. PAYE"),
    ("B02", 2024, 255_167, 335_656, "NMB/NMB_Intergrated_Annual_Report_2024.pdf", 122, "NMB-AR-FY2024",
     "value-added statement figure excl. PAYE"),
    ("B02", 2025, 287_558, 379_674, "NMB/NMB_Integrated Annual Report_2025_.pdf", 150, "NMB-AR-FY2025",
     "value-added statement figure excl. PAYE"),
]


def fixes(root=None) -> list[dict]:
    root = root or find_root()
    out = []
    for b, fy, old, new, f, pg, doc, why in FIXES:
        if not printed(f, pg, [new], root):
            raise ValueError(f"{b} FY{fy}: {new:,} not printed on {f} p.{pg}")
        out.append({"bank_id": b, "fiscal_year": fy, "variable": "Staff Costs", "basis": "GROUP",
                    "old_value": old * M, "value": new * M, "doc_id": doc, "page": pg, "file": f, "reason": why,
                    "session": f"{VERSION} (prompt 04 pre-flight)"})
    return out


def run() -> dict[str, int]:
    import pandas as pd

    root = find_root()
    fx = fixes(root)
    lines = [f"{VERSION} (2026-11-08, prompt 04 pre-flight, D18 working workbook)",
             f"Historical Financials: {len(fx)} staff-cost values corrected in place (CRDB note totals; NMB "
             "income-statement employee benefits expense); previous values kept in Notes."]
    n = workbook.correct(fx, lines, root)
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp, low_memory=False)
    add = pd.DataFrame([{"bank_id": x["bank_id"], "fiscal_year": x["fiscal_year"], "variable": x["variable"],
                         "value": x["value"], "row_type": "correction", "file": x["file"], "page": x["page"],
                         "label": "employee benefits total", "column": "current",
                         "gate": "printed on the page (re-checked)", "confidence": "HIGH",
                         "extraction_method": "owner-review correction",
                         "notes": f"CORRECTED {x['session']}: previously {x['old_value']:,.0f} ({x['reason']})",
                         "evidence_class": "Observed", "doc_fiscal_year": x["fiscal_year"], "doc_id": x["doc_id"]}
                        for x in fx])
    pd.concat([old, add.reindex(columns=old.columns)], ignore_index=True).to_csv(lp, index=False,
                                                                                 lineterminator="\n")
    return {"corrected": n}


if __name__ == "__main__":
    print(run())
