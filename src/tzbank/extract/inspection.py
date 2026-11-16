"""PDF inspection report (prompt 02 section 1.1; OCR task section 6): reports/pdf_inspection.csv.

One row per open gap (bank x variable x year) from the prompt 01 audit (data/extraction/coverage_before_prompt02.csv)
plus the NMB/CRDB items of prompt 02 section 1.3. Each row says which held PDF was checked, whether it had a text
layer or went through OCR, whether the item was located and accepted, and why not.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path, settings
from tzbank.extract.ledger import FOLDERS
from tzbank.io import write_csv

GAP_VARS = {"interest_income": "Interest Income", "interest_expense": "Interest Expense",
            "interest_expense_deposits": "Interest Expense on Customer Deposits",
            "customer_deposits": "Customer Deposits", "rwa_total": "Total RWA", "tier1_capital": "Tier 1 Capital",
            "impairment_charges": "Impairment Charges", "gross_loans": "Gross Loans and Advances to Customers",
            "profit_attributable": "Net Profit", "income_tax_expense": "Income Tax Expense"}
FOCUS_ITEMS = ["Interest Expense on Customer Deposits", "Interest Expense on Borrowings",
               "Customer Deposits: Current Accounts", "Customer Deposits: Savings", "Customer Deposits: Time"]
COLUMNS = ["bank_id", "variable", "fiscal_year", "pdf_checked", "text_layer", "ocr_status", "located", "page_or_note",
           "as_reported_label", "basis", "extractable", "reason_if_not", "gate", "evidence_class"]


def document_status(root: Path) -> pd.DataFrame:
    """Every held bank PDF: pages, text layer, OCR status (from data/ocr/ocr_log.csv)."""
    log_f = root / "data" / "ocr" / "ocr_log.csv"
    log = pd.read_csv(log_f) if log_f.exists() else pd.DataFrame(columns=["file", "status", "text_pages_after"])
    ocr = dict(zip(log["file"], log["status"], strict=False))
    rows = []
    for b, folder in FOLDERS.items():
        for f in sorted((path("documents", root) / folder).glob("*.pdf")):
            d = pymupdf.open(f)
            n, t = d.page_count, sum(1 for p in d if len(p.get_text()) > 200)
            d.close()
            rel = f"documents/{folder}/{f.name}"
            scanned = t / max(n, 1) < 0.5
            st = ocr.get(rel)
            rows.append({"bank_id": b, "file": f"{folder}/{f.name}", "pages": n, "text_pages": t,
                         "text_layer": "N" if scanned else "Y",
                         "ocr_status": (str(st) if st else "OCR needed (not run)") if scanned else "not needed",
                         "condensed": n <= 6})
    return pd.DataFrame(rows)


def build(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    docs = document_status(root)
    cand = pd.read_csv(root / "data" / "extraction" / "candidates_all.csv", low_memory=False)
    ledger = pd.read_csv(root / "data" / "extraction" / "ledger.csv")
    cov = pd.read_csv(root / "data" / "extraction" / "coverage_before_prompt02.csv")
    gaps = cov[(cov["available"] == "N") & cov["canonical_name"].isin(GAP_VARS)]
    gaps = gaps[~((gaps["canonical_name"] == "profit_attributable") & gaps["bank_id"].isin(["B01", "B02"]))]
    items = [(r.bank_id, GAP_VARS[r.canonical_name], int(r.fiscal_year)) for r in gaps.itertuples()]
    last = settings(root)["panel_years"][1]
    for b, first in [("B01", 2009), ("B02", 2008)]:
        items += [(b, v, y) for v in FOCUS_ITEMS for y in range(first, last + 1)]
    items = sorted(set(items))
    fy_of_file = cand.dropna(subset=["fiscal_year"]).query("column == 'current'").groupby("file")["fiscal_year"].max()
    rows = []
    for b, var, y in items:
        bd = docs[docs["bank_id"] == b].copy()
        bd["fy"] = bd["file"].map(fy_of_file).fillna(bd["file"].str.extract(r"(\d{4})")[0].astype(float))
        held = bd[bd["fy"].isin([y, y + 1])]
        acc = ledger[(ledger["bank_id"] == b) & (ledger["variable"] == var) & (ledger["fiscal_year"] == y)]
        cnd = cand[(cand["bank_id"] == b) & (cand["variable"] == var) & (cand["fiscal_year"] == y)]
        basis = "Group" if b in {"B01", "B02"} and var not in {"Total RWA", "Tier 1 Capital"} else "Bank"
        row = {"bank_id": b, "variable": var, "fiscal_year": y, "basis": basis,
               "pdf_checked": "; ".join(held["file"]) or "none held",
               "text_layer": "; ".join(sorted(set(held["text_layer"]))) or "",
               "ocr_status": "; ".join(sorted(set(held["ocr_status"]) - {"not needed"})) or "not needed"}
        if len(acc):
            a = acc.iloc[0]
            row |= {"located": "Y", "page_or_note": f"{a['file']} p.{int(a['page'])}", "as_reported_label": a["label"],
                    "extractable": "Y", "reason_if_not": "", "gate": a["gate"]}
        elif len(cnd):
            c = cnd.iloc[0]
            row |= {"located": "Y", "page_or_note": f"{c['file']} p.{int(c['page'])}", "as_reported_label": c["label"],
                    "extractable": "partial", "gate": c["gate"],
                    "reason_if_not": "located but failed the validation gate: manual verification"
                    + (" (OCR text)" if str(c.get("ocr")) == "True" else "")}
        elif held.empty:
            row |= {"located": "N", "extractable": "N", "reason_if_not": "year not held"}
        elif held["text_layer"].eq("N").all() and not held["ocr_status"].str.startswith("OK").any():
            row |= {"located": "N", "extractable": "N", "reason_if_not": "scanned; OCR failed or not run (illegible)"}
        elif held["condensed"].all():
            row |= {"located": "N", "extractable": "N",
                    "reason_if_not": "not disclosed (only a condensed BoT-format publication is held)"}
        else:
            row |= {"located": "N", "extractable": "N",
                    "reason_if_not": "not located by the extractor in the held report: manual check pending"}
        rows.append(row)
    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out["evidence_class"] = "Observed"
    return out


def collection_tasks(insp: pd.DataFrame) -> pd.DataFrame:
    """Only 'year not held' (and confirmed 'not in held PDFs') become owner collection tasks."""
    t = insp[insp["reason_if_not"].isin(["year not held", "not in held PDFs"])]
    g = (t.groupby(["bank_id", "fiscal_year", "reason_if_not"])["variable"].apply(lambda s: "; ".join(sorted(s)))
         .reset_index())
    g["document_needed"] = g.apply(lambda r: f"{r['bank_id']} annual report FY{r['fiscal_year']} (or FY"
                                             f"{r['fiscal_year'] + 1} with comparatives)", axis=1)
    g["task_type"] = "missing annual report"
    owner = pd.DataFrame([
        {"bank_id": "BoT", "fiscal_year": "2008-2012", "reason_if_not": "not held (before the auction calendar; "
         "the calendar has no auction of 15 years or more before 2013, D34)",
         "variable": "long-bond yield (tenor >= 15 years), final quarter of each year (annual COE, blueprint "
                     "section 2; D34)",
         "document_needed": "BoT Annual Reports or Monthly Economic Review, FY2008-FY2012", "task_type": "owner"},
        {"bank_id": "Regulation", "fiscal_year": "2008-2013", "reason_if_not": "not held (D28)",
         "variable": "annual and quarterly publication deadlines",
         "document_needed": "Banking and Financial Institutions (Publication of Financial Statements) Regulations, "
                            "2008, GN 368 of 2008. The file supplied on 2026-11-06 is GN 368 of 2009 (Tanzania Court "
                            "of Appeal Rules) and is not the right instrument", "task_type": "owner",
         "status": "closed: text unavailable; resolved by assumption (D36; gazette No. 50 of 12 Dec 2008 registered "
                   "as REG-GAZ-2008-50)"},
        {"bank_id": "B01", "fiscal_year": 2025, "reason_if_not": "full report not held locally",
         "variable": "Interest Expense on Customer Deposits (and other note items)",
         "document_needed": "CRDB Integrated Annual Report FY2025 (Sources CRDB-AR-FY2025 URL)", "task_type": "owner",
         "status": "closed: full report placed (CRDB/crdb-ir-2025-8th-may-2026.pdf); FY2025 note lines extracted in "
                   "prompt 02e"},
    ])
    g = pd.concat([g, owner], ignore_index=True)
    g["status"] = g.get("status", pd.Series(index=g.index, dtype=object)).fillna("open")
    g["evidence_class"] = "Observed"
    return g


def write_inspection(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    insp = build(root)
    docs = document_status(root)
    return [write_csv(insp, path("reports", root) / "pdf_inspection.csv", require_evidence=True),
            write_csv(docs.assign(evidence_class="Observed"), path("reports", root) / "pdf_documents.csv"),
            write_csv(collection_tasks(insp), path("reports", root) / "collection_tasks.csv", require_evidence=True)]
