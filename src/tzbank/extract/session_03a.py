"""D18 workbook session for prompt 03a (v0.14): NMB and CRDB detail lines for chapter 4 (section 2).

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_03a
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

Writes the accepted rows of ``data/extraction/detail_lines.csv`` (``tzbank.extract.detail_lines``: interest income by
source, gross fee and commission income by type, staff costs, operating KPIs) to Historical Financials with
``extraction_method=gated``, appends them to the extraction ledger, and adds the crosswalk rows.
"""

from __future__ import annotations

import csv

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.session import assign_doc_ids
from tzbank.extract.sources import new_sources
from tzbank.ingest.inputs_workbook import load_interim

VERSION = "v0.14"
RENAME = {"Fee and Commission Income": "Fee and Commission Income (Gross, Note Total)"}
CROSSWALK = {  # inputs-workbook name -> (canonical name, unit, definition)
    "Interest Income: Loans and Advances": ("interest_income_loans", "TZS",
                                            "Interest income on loans and advances (interest-income note lines)"),
    "Interest Income: Government Securities": ("interest_income_securities", "TZS",
                                               "Interest income on Treasury bills, bonds and other securities"),
    "Interest Income: Placements and Interbank": ("interest_income_placements", "TZS",
                                                  "Interest income on placements, interbank and bank balances"),
    "Interest Income: Other": ("interest_income_other", "TZS", "Other interest-income note lines"),
    "Interest Income: Note Total": ("interest_income_note_total", "TZS",
                                    "Printed total of the interest-income note (gate: lines sum to it; within 2% of "
                                    "audited interest income)"),
    "Fee and Commission Income (Gross, Note Total)": ("fee_income_total", "TZS",
                                                      "Gross fee and commission income, printed note total"),
    "Fee Income: Transactions and Accounts": ("fee_income_transactions", "TZS",
                                              "Fees on accounts, transfers, cheques, payments and commissions"),
    "Fee Income: Cards": ("fee_income_cards", "TZS", "Card, ATM and POS fees"),
    "Fee Income: Mobile and Internet Banking": ("fee_income_digital", "TZS",
                                                "Mobile, internet, agency and digital-channel fees"),
    "Fee Income: Loan-Related Fees": ("fee_income_loan_fees", "TZS",
                                      "Loan application, arrangement, guarantee, LC and bancassurance fees"),
    "Fee Income: Foreign Exchange": ("fee_income_fx", "TZS", "FX-related fee lines inside the fee note"),
    "Fee Income: Other": ("fee_income_other", "TZS", "Other fee-note lines"),
    "Staff Costs": ("staff_costs", "TZS", "Employee benefit expenses / staff costs"),
    "Number of Employees": ("kpi_employees", "count", "Employees (narrative / KPI page; MEDIUM)"),
    "Number of Branches": ("kpi_branches", "count", "Branches (narrative / KPI page; MEDIUM)"),
    "Number of Banking Agents": ("kpi_agents", "count", "Banking agents (narrative / KPI page; MEDIUM)"),
    "Number of ATMs": ("kpi_atms", "count", "ATMs (narrative / KPI page; MEDIUM)"),
    "Number of Customers": ("kpi_customers", "count", "Customers (narrative / KPI page; MEDIUM)"),
}


def add_crosswalk(root) -> int:
    f = path("crosswalk", root)
    rows = list(csv.reader(f.open(encoding="utf-8")))
    hdr, have = rows[0], {r[2] for r in rows[1:] if len(r) > 2}
    new = []
    for name, (canon, unit, definition) in CROSSWALK.items():
        if name in have:
            continue
        d = dict.fromkeys(hdr, "")
        d.update(canonical_name=canon, inputs_name=name, unit=unit, basis="GROUP", definition=definition,
                 blueprint_chapters="4", evidence_class="Observed", source="inputs_workbook",
                 source_sheets="Historical Financials", status="mapped")
        new.append([d[h] for h in hdr])
    with f.open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(new)
    return len(new)


def run() -> dict[str, int]:
    root = find_root()
    d = pd.read_csv(root / "data" / "extraction" / "detail_lines.csv")
    d = d[d["gate_passed"].astype(str).eq("True") & d["variable"].isin(list(CROSSWALK) + list(RENAME))].copy()
    d["variable"] = d["variable"].replace(RENAME)
    d["ocr"] = False
    d["extraction_method"] = "gated"
    d["row_type"] = "original"
    d["page"] = d["page"].astype(int)
    d["notes"] = d["notes"].fillna("").astype(str) + " | prompt 03a detail line (" + d["note_kind"] + ")"
    src = load_interim("Sources", root)
    led = assign_doc_ids(d, set(src["Doc ID"]), root)
    new = new_sources(led[led["doc_is_new"]], root) if led["doc_is_new"].any() else pd.DataFrame(columns=["Doc ID"])
    n_cw = add_crosswalk(root)
    lines = [f"{VERSION} (2026-11-08, prompt 03a, D18 working workbook)",
             f"Historical Financials: NMB and CRDB detail lines for chapter 4 ({len(led)} rows: interest income by "
             "source, gross fees by type, staff costs, KPIs) from tzbank.extract.detail_lines; note lines sum to the "
             "printed note totals; KPIs MEDIUM, adjacent-year checked."]
    counts = workbook.update(led, new, pd.DataFrame(columns=["Doc ID"]), lines, root)
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp)
    pd.concat([old, led.reindex(columns=old.columns).assign(evidence_class="Observed")], ignore_index=True).to_csv(
        lp, index=False, lineterminator="\n")
    return counts | {"crosswalk_rows": n_cw, "rows": len(led)}


if __name__ == "__main__":
    print(run())
