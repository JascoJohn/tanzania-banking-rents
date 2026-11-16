"""Manual check list (prompt 02b sections 1-3): the Priority 1 and 2 items no gate could extract, with where to look.

Priority 1: NMB/CRDB variables needed by chapter 4 (4.0-4.6) and chapters 5-8.
Priority 2: B03-B07 core variables (interest income/expense, NII, interest expense on deposits, deposits, gross
loans, allowance, RWA, Tier 1, attributable profit and equity).
Priority 3 (everything else) is counted but not listed.

The owner reads each value and enters it in data/raw/manual_entries/manual_values.csv (D29).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.extract.ocr import extraction_source
from tzbank.io import write_csv

P1_VARS = {"Interest Expense on Customer Deposits", "Interest Expense on Borrowings", "Customer Deposits: Current Accounts",
           "Customer Deposits: Savings", "Customer Deposits: Time", "Interest Income", "Interest Expense",
           "Gross Loans and Advances to Customers", "Total RWA", "Tier 1 Capital", "Impairment Charges",
           "Customer Deposits"}
P2_VARS = {"Interest Income", "Interest Expense", "Net Interest Income", "Interest Expense on Customer Deposits",
           "Customer Deposits", "Gross Loans and Advances to Customers", "Loss Allowance on Loans (ECL)", "Total RWA",
           "Tier 1 Capital", "Net Profit", "Total Equity"}
EXPECTED = {
    "Interest Income": ("Interest income / Interest and similar income (statement of profit or loss)",
                        r"interest (and similar )?income"),
    "Interest Expense": ("Interest expense / Interest and similar expense (statement of profit or loss)",
                         r"interest (and similar )?expense"),
    "Net Interest Income": ("Net interest income", r"net interest income"),
    "Interest Expense on Customer Deposits": ("Interest expense note: deposits from customers (current, savings, "
                                              "time/fixed); sum the deposit lines", r"interest (and similar )?expense"),
    "Interest Expense on Borrowings": ("Interest expense note: borrowings, subordinated debt, bonds",
                                       r"interest (and similar )?expense"),
    "Customer Deposits": ("Deposits from customers (statement of financial position)",
                          r"deposits? (from|due to) customers|customer deposits"),
    "Customer Deposits: Current Accounts": ("Deposits from customers note: current/demand accounts",
                                            r"current accounts|demand deposits"),
    "Customer Deposits: Savings": ("Deposits from customers note: savings accounts", r"savings"),
    "Customer Deposits: Time": ("Deposits from customers note: time/fixed/term deposits", r"(time|fixed|term) deposits"),
    "Gross Loans and Advances to Customers": ("Loans and advances note: gross carrying amount before allowance",
                                              r"gross (loans|carrying)"),
    "Loss Allowance on Loans (ECL)": ("Loans and advances note: allowance for impairment / expected credit losses",
                                      r"allowance|expected credit loss|impairment"),
    "Total RWA": ("Capital management note: total risk-weighted assets", r"risk[- ]weighted"),
    "Tier 1 Capital": ("Capital management note: Tier 1 / core capital", r"tier ?(1|i)|core capital"),
    "Impairment Charges": ("Impairment charge / credit loss expense (statement of profit or loss)",
                           r"impairment|credit loss"),
    "Net Profit": ("Profit for the year (statement of profit or loss)", r"profit for the year|profit after tax"),
    "Total Equity": ("Total equity / shareholders' funds (statement of financial position)",
                     r"total (equity|shareholders)"),
}
SWAHILI = [
    ("Mapato ya Riba", "Interest Income"), ("Gharama za Riba", "Interest Expense"),
    ("Mapato Halisi ya Riba (1 kutoa 2)", "Net Interest Income"),
    ("Punguzo/Ongezeko la tengo katika Mikopo", "Impairment Charges (loan provision charge)"),
    ("Mapato Yasiyo ya Riba", "Non-Interest Income"), ("Gharama Zisizo za Riba", "Operating Expenses"),
    ("Mishahara na Mafao", "staff costs (part of operating expenses)"),
    ("Faida (Hasara) ya Uendeshaji", "Profit Before Tax (operating profit/loss)"),
    ("Tengo la Kodi ya Mapato", "Income Tax Expense"),
    ("Faida (Hasara) Halisi Baada ya Kodi ya Mapato", "Net Profit"), ("JUMLA YA MALI", "Total Assets"),
    ("Mikopo Halisi", "Loans and Advances to Customers (Net)"), ("Amana za Wateja", "Customer Deposits"),
    ("Amana za Benki nyingine na Taasisi za fedha", "deposits from banks (not customer deposits)"),
    ("Mtaji wa Hisa Uliolipwa", "paid-up share capital"), ("MALI/(DHIMA) HALISI", "Total Equity (net assets)"),
    ("Mikopo Chechefu", "non-performing loans (NPL)"),
    ("Tengo kwa Ajili ya Mikopo Chechefu", "allowance for NPLs"),
    ("Uwiano wa Mikopo Ghafi na Jumla ya Amana", "gross loans / total deposits (ratio)"),
    ("mwaka huu / mwaka uliopita", "current year / previous year (columns)"),
]
LIST_COLUMNS = ["priority", "bank_id", "variable", "fiscal_year", "pdf_original", "pdf_ocr", "candidate_pages",
                "expected_label", "reason_not_located", "target_total", "value", "unit", "page", "checked_by",
                "evidence_class"]


def candidate_pages(pdf: Path, pattern: str, limit: int = 6) -> str:
    """Pages whose text matches the label pattern (OCR copy for scanned PDFs)."""
    src = extraction_source(pdf)
    rx = re.compile(pattern, re.I)
    pages = []
    with pymupdf.open(src) as d:
        for i, p in enumerate(d, start=1):
            if rx.search(p.get_text()):
                pages.append(i)
    if len(pages) > limit:
        return ", ".join(map(str, pages[:limit])) + f" (+{len(pages) - limit} more)"
    return ", ".join(map(str, pages))


def audited_ie(root: Path) -> dict[tuple[str, int], float]:
    from tzbank.ingest.inputs_workbook import load_interim

    hf = load_interim("Historical Financials", root)
    ie = hf[(hf["Variable"] == "Interest Expense") & hf["value_numeric"].notna() & hf["Entity"].isin(["Group"])]
    return {(b, int(d.year)): v for b, d, v in zip(ie["Bank ID"], ie["Period End"], ie["value_numeric"], strict=False)}


def build(root: Path | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    root = root or find_root()
    insp = pd.read_csv(path("reports", root) / "pdf_inspection.csv")
    open_ = insp[insp["extractable"] != "Y"].copy()
    open_ = open_[~open_["reason_if_not"].fillna("").str.startswith(("year not held", "not disclosed"))]
    focus = open_["bank_id"].isin(["B01", "B02"])
    open_["priority"] = 3
    open_.loc[focus & open_["variable"].isin(P1_VARS), "priority"] = 1
    open_.loc[~focus & open_["variable"].isin(P2_VARS), "priority"] = 2
    counts = open_["priority"].value_counts().sort_index().to_dict()
    targets = audited_ie(root)
    rows = []
    for _, r in open_[open_["priority"] <= 2].iterrows():
        exp, pat = EXPECTED.get(r["variable"], (r["variable"], re.escape(r["variable"].lower())))
        files = [f for f in str(r["pdf_checked"]).split("; ") if f and f != "none held"]
        orig = [path("documents", root) / f for f in files]
        ocr = [str(extraction_source(o).relative_to(root).as_posix()) for o in orig if extraction_source(o) != o]
        cand = "; ".join(f"{f}: p.{candidate_pages(o, pat)}" for f, o in zip(files, orig, strict=False) if o.exists())
        tgt = targets.get((r["bank_id"], int(r["fiscal_year"]))) if "Interest Expense on" in r["variable"] else None
        rows.append({"priority": r["priority"], "bank_id": r["bank_id"], "variable": r["variable"],
                     "fiscal_year": r["fiscal_year"], "pdf_original": "; ".join(f"data/raw/documents/{f}" for f in files),
                     "pdf_ocr": "; ".join(ocr), "candidate_pages": cand, "expected_label": exp,
                     "reason_not_located": r["reason_if_not"],
                     "target_total": f"interest expense note must sum to audited total TZS {tgt:,.0f}" if tgt else "",
                     "value": None, "unit": None, "page": None, "checked_by": None})
    out = pd.DataFrame(rows, columns=LIST_COLUMNS)
    return out, counts


def add_special_items(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """MCB FY2023 Swahili newspaper page (prompt 02b section 3.1)."""
    extra = []
    for v in ["Interest Income", "Interest Expense", "Net Interest Income", "Customer Deposits", "Net Profit",
              "Total Equity"]:
        extra.append({"priority": 2, "bank_id": "B06", "variable": v, "fiscal_year": 2023,
                      "pdf_original": "data/raw/documents/MCB/2023.pdf", "pdf_ocr": "data/ocr/MCB/2023.pdf",
                      "candidate_pages": "MCB/2023.pdf: p.1 (single newspaper page)",
                      "expected_label": next(sw for sw, en in SWAHILI if en.startswith(v)) + f" (= {v}); see glossary",
                      "reason_not_located": "Swahili newspaper publication (Nipashe, 15 Apr 2024); OCR digits "
                                            "unreliable at this resolution; owner to read (glossary: "
                                            "reports/manual_check_glossary_sw.csv)",
                      "target_total": "", "value": None, "unit": "TZS million", "page": None, "checked_by": None})
    # CRDB FY2025 deposit interest: closed in prompt 02e (full report held; Note 13 extracted)
    extra += detail_line_gaps(root)
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


DETAIL_ITEMS = {"interest_income": ("Interest income by source (loans, government securities, placements, other)",
                                     "interest-income note; lines must sum to the audited interest income"),
                "fee_income": ("Fee and commission income by type (transactions, cards, mobile/internet, loan fees, "
                               "FX, other)", "fee and commission note; lines must sum to the fee total"),
                "staff_costs": ("Staff costs", "employee benefit expenses / staff costs note"),
                "loans_by_sector": ("Loans by economic sector", "credit-risk concentration by industry; lines must "
                                    "sum to gross loans"),
                "loans_by_segment": ("Loans by segment (retail, SME, corporate, government)", "loans note or segment "
                                     "report; lines must sum to gross loans")}


def detail_line_gaps(root: Path, years: range = range(2015, 2026)) -> list[dict]:
    """Prompt 03a: NMB/CRDB detail lines the extractor could not find or reconcile, as Priority 1 items."""
    f = root / "data" / "extraction" / "detail_lines.csv"
    r = root / "data" / "extraction" / "detail_lines_review.csv"
    if not f.exists() or not r.exists():
        return []
    got = pd.read_csv(f)
    have = set(zip(got["bank_id"], got["fiscal_year"], got["note_kind"], strict=False))
    out = []
    for b in ("B01", "B02"):
        for y in years:
            for kind, (var, label) in DETAIL_ITEMS.items():
                if (b, y, kind) in have:
                    continue
                out.append({"priority": 1, "bank_id": b, "variable": var, "fiscal_year": y, "pdf_original": "",
                            "pdf_ocr": "", "candidate_pages": "", "expected_label": label,
                            "reason_not_located": "prompt 03a detail line: not found or not reconciled by "
                                                  "tzbank.extract.detail_lines (data/extraction/detail_lines_review.csv)",
                            "target_total": "", "value": None, "unit": None, "page": None, "checked_by": None})
    return out


def write_manual_list(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    df, counts = build(root)
    df = add_special_items(df, root)
    df["evidence_class"] = "Observed"
    df = df.drop_duplicates(["bank_id", "variable", "fiscal_year"]).sort_values(["priority", "bank_id", "variable",
                                                                               "fiscal_year"])
    gl = pd.DataFrame(SWAHILI, columns=["swahili_label", "canonical_meaning"]).assign(evidence_class="Observed")
    summary = pd.DataFrame({"priority": list(counts), "open_items": list(counts.values())})
    summary["listed"] = summary["priority"].map(df["priority"].value_counts()).fillna(0).astype(int)
    summary["evidence_class"] = "Derived"
    rdir = path("reports", root)
    return [write_csv(df, rdir / "manual_check_list.csv", require_evidence=True),
            write_csv(gl, rdir / "manual_check_glossary_sw.csv"),
            write_csv(summary, rdir / "manual_check_summary.csv")]
