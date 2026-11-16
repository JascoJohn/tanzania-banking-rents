"""Data audit (prompt 01 section 8; blueprint work-sequence step 3: audit the seven-bank and NMB/CRDB datasets
against the variable dictionary and list gaps).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.clean.crosswalk import chapters_of, load_crosswalk
from tzbank.clean.normalise import PRIMARY_ENTITIES, load_canonical
from tzbank.config import banks, find_root, path, settings
from tzbank.io import write_csv, write_text

DOC_FOLDERS = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "Maendeleo", "B05": "MKCB", "B06": "MCB",
               "B07": "MUCOBA"}
AUDIT_CHAPTERS = {4, 5, 6, 7, 8}
MISSING_STATUSES = {"TO COLLECT", "NOT DISCLOSED", "MISSING", "NOT FOUND", "NOT APPLICABLE", "NOT YET PUBLISHED"}

# Blueprint §2-§4 analyses and the canonical inputs each needs ('a|b' = either; first is the blueprint definition).
ANALYSES: dict[str, list[str]] = {
    "4 DuPont / profitability starter": ["profit_attributable|net_profit", "equity_attributable|total_equity",
                                         "total_assets", "net_interest_income", "non_interest_income",
                                         "operating_expenses", "impairment_charges"],
    "4.1 Revenue architecture (interest vs non-interest income)": ["interest_income", "interest_expense",
                                                                   "net_interest_income", "non_interest_income"],
    "4.2 Funding economics (cost of deposits, GDFA)": ["interest_expense_deposits", "customer_deposits"],
    "4.3 Asset deployment (loan yield, securities share)": ["interest_income", "gross_loans|net_loans",
                                                           "government_securities", "total_assets"],
    "4.4 Transaction economics (fee income per asset)": ["non_interest_income", "total_assets"],
    "4.5 Operating leverage (cost-to-income, cost per asset)": ["operating_expenses", "net_interest_income",
                                                               "non_interest_income", "total_assets"],
    "4.6 / 8 Risk economics (cost of risk, return on RWA)": ["impairment_charges", "gross_loans", "rwa_total",
                                                             "net_profit"],
    "5 Persistence (ER = ROE - COE, bank-level inputs)": ["profit_attributable|net_profit",
                                                          "equity_attributable|total_equity"],
    "6 Funding architecture (deposit cost, beta, LDR, borrowings)": ["interest_expense_deposits",
                                                                     "customer_deposits", "borrowings",
                                                                     "gross_loans"],
    "7 Capital allocation (primary RORE, incremental ROE)": ["profit_attributable", "dps", "shares_outstanding",
                                                             "equity_attributable"],
    "8 Credit risk and capital efficiency (NPL, coverage, RWA)": ["impairment_charges", "gross_loans", "npl_ratio",
                                                                  "npl_coverage_ratio", "rwa_total", "tier1_capital"],
}
SUPPORT_THRESHOLD = 0.8


def bank_years(root: Path) -> pd.DataFrame:
    """Audit universe: each bank from its first listed year (NMB FY2008, CRDB FY2009 per D6) to the last panel year."""
    last = settings(root)["panel_years"][1]
    rows = [{"bank_id": b["bank_id"], "fiscal_year": y} for b in banks(root)
            for y in range(max(b["listing_year"], settings(root)["panel_years"][0]), last + 1)]
    return pd.DataFrame(rows)


def audit_variables(root: Path) -> list[str]:
    """Canonical bank-level variables the crosswalk assigns to chapters 4-8."""
    ch = chapters_of(load_crosswalk(root))
    market_level = ("macro_", "peer_", "mkt_", "incl_", "nps_")  # not bank-level
    return sorted(c for c, s in ch.items() if s & AUDIT_CHAPTERS and not c.startswith(market_level))


def coverage_long(root: Path | None = None) -> pd.DataFrame:
    """Bank x year x variable availability (FY, bank's own entity, any basis, original or restated)."""
    root = root or find_root()
    long = load_canonical(root, path("processed", root) / "bank_period_long.csv")
    fy = long[(long["period_type"] == "FY") & long["entity"].isin(PRIMARY_ENTITIES) & long["category"].isna()]
    grid = bank_years(root).merge(pd.DataFrame({"canonical_name": audit_variables(root)}), how="cross")
    have = (fy[fy["value"].notna()].groupby(["bank_id", "fiscal_year", "canonical_name"])
            .agg(bases=("basis", lambda s: ";".join(sorted(set(s)))),
                 provenance=("provenance", lambda s: ";".join(sorted(set(s))))).reset_index())
    gap = (fy[fy["value"].isna()].groupby(["bank_id", "fiscal_year", "canonical_name"])["status"]
           .agg(lambda s: ";".join(sorted(set(s.dropna())))).rename("gap_status").reset_index())
    out = grid.merge(have, how="left", on=["bank_id", "fiscal_year", "canonical_name"]).merge(
        gap, how="left", on=["bank_id", "fiscal_year", "canonical_name"])
    out["available"] = out["bases"].notna().map({True: "Y", False: "N"})
    out["gap_status"] = out["gap_status"].where(out["available"] == "N")
    out.loc[(out["available"] == "N") & out["gap_status"].isna(), "gap_status"] = "NOT IN EITHER SOURCE"
    out["evidence_class"] = "Observed"
    return out.sort_values(["canonical_name", "bank_id", "fiscal_year"]).reset_index(drop=True)


def documents_held(root: Path) -> dict[str, set[int]]:
    """Fiscal years with an annual report PDF registered in MANIFEST.csv under documents/<bank>/ (e.g. 2019.pdf).

    Read from the committed MANIFEST, not the git-ignored folder, so a fresh clone gives the same answer.
    """
    man = pd.read_csv(path("manifest", root), dtype=str)
    out: dict[str, set[int]] = {}
    for b, folder in DOC_FOLDERS.items():
        files = man.loc[man["file"].str.match(rf"documents/{folder}/[^/]+\.pdf$", case=False), "file"]
        out[b] = {int(m.group(1)) for f in files if (m := re.search(r"(20\d\d)", f.rsplit("/", 1)[-1]))}
    return out


def gap_recommendations(cov: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Missing chapter 4-8 inputs per bank: extract from held annual reports, or accept as a limitation.

    A missing FY is extractable if the FY report or the next year's report (comparatives) is held.
    """
    held = documents_held(root)
    want = settings(root)["non_focus_gap_variables"]
    miss = cov[(cov["available"] == "N") & cov["canonical_name"].isin(want)]
    rows = []
    for (b, v), g in miss.groupby(["bank_id", "canonical_name"]):
        years = sorted(g["fiscal_year"])
        ext = [y for y in years if y in held[b] or (y + 1) in held[b]]
        lim = [y for y in years if y not in ext]
        if ext and v == "interest_expense_deposits":
            rec = "Extract from annual-report interest-expense note (deposit line); not captured in either source"
        elif ext:
            rec = "Extract from annual reports held (statement or note), add rows to Historical Financials"
        else:
            rec = "Accept as limitation (no annual report held for these years)"
        rows.append({"bank_id": b, "canonical_name": v, "years_missing": _ranges(years), "n_missing": len(years),
                     "years_extractable_from_held_pdfs": _ranges(ext), "years_no_document_held": _ranges(lim),
                     "recommendation": rec, "evidence_class": "Interpretation"})
    return pd.DataFrame(rows)


def _ranges(years: list[int]) -> str:
    if not years:
        return ""
    years = sorted(years)
    out, start, prev = [], years[0], years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = y
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


def analysis_support(cov: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Share of each bank's years in which every input of an analysis is available (alternatives allowed)."""
    avail = set(map(tuple, cov.loc[cov["available"] == "Y", ["bank_id", "fiscal_year", "canonical_name"]].to_numpy()))
    by = bank_years(root)
    focus = set(settings(root)["focus_banks"])
    rows = []
    for name, req in ANALYSES.items():
        primary_ok, any_ok = [], []
        for b, y in by.itertuples(index=False):
            primary_ok.append(all((b, y, r.split("|")[0]) in avail for r in req))
            any_ok.append(all(any((b, y, a) in avail for a in r.split("|")) for r in req))
        d = by.assign(primary=primary_ok, with_fallback=any_ok)
        share = d.groupby("bank_id")[["primary", "with_fallback"]].mean()
        f_ok = all(share.loc[list(focus), "with_fallback"] >= SUPPORT_THRESHOLD)
        all_ok = all(share["with_fallback"] >= SUPPORT_THRESHOLD)
        level = ("All seven banks" if all_ok else "NMB/CRDB only" if f_ok else "Not supported yet")
        row = {"analysis": name, "inputs": ", ".join(req), "support": level,
               "fallback_used": "yes" if (share["primary"] < share["with_fallback"]).any() else "no"}
        for b in share.index:
            row[f"{b}_share"] = round(float(share.loc[b, "with_fallback"]), 2)
        missing = sorted({r for r in req for b in share.index if share.loc[b, "with_fallback"] < SUPPORT_THRESHOLD
                          and cov[(cov.bank_id == b) & cov.canonical_name.isin(r.split("|"))
                                  & (cov.available == "Y")].empty})
        row["inputs_absent_for_some_bank"] = ", ".join(missing)
        row["evidence_class"] = "Derived"
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_matrix(cov: pd.DataFrame) -> pd.DataFrame:
    """variable x bank: years available / years in the bank's audit range."""
    g = cov.groupby(["canonical_name", "bank_id"])["available"].agg(lambda s: f"{(s == 'Y').sum()}/{len(s)}")
    return g.unstack("bank_id").reset_index()


INSPECTION_VARS = {"Interest Income": "interest_income", "Interest Expense": "interest_expense",
                   "Interest Expense on Customer Deposits": "interest_expense_deposits",
                   "Customer Deposits": "customer_deposits", "Total RWA": "rwa_total", "Tier 1 Capital": "tier1_capital",
                   "Impairment Charges": "impairment_charges",
                   "Gross Loans and Advances to Customers": "gross_loans", "Income Tax Expense": "income_tax_expense"}


def remaining_by_reason(cov: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Still-missing bank-years for the inspected variables, counted by inspection reason."""
    f = path("reports", root) / "pdf_inspection.csv"
    if not f.exists():
        return pd.DataFrame()
    insp = pd.read_csv(f)
    insp["canonical_name"] = insp["variable"].map(INSPECTION_VARS)
    miss = cov[cov["available"] == "N"][["bank_id", "fiscal_year", "canonical_name"]]
    m = miss.merge(insp, on=["bank_id", "fiscal_year", "canonical_name"], how="inner")
    m["reason"] = m["reason_if_not"].fillna("extracted; not yet in panel")
    return (m.groupby(["canonical_name", "reason"])["bank_id"]
            .agg(bank_years="size", banks=lambda b: ", ".join(sorted(set(b)))).reset_index())


def render(cov: pd.DataFrame, gaps: pd.DataFrame, support: pd.DataFrame, root: Path) -> str:
    """reports/data_audit.md text."""
    from tzbank.validate.run import _md_table

    names = {b["bank_id"]: f"{b['bank_id']} {b['ticker']}" for b in banks(root)}
    mat = coverage_matrix(cov).rename(columns=names)
    focus = set(settings(root)["focus_banks"])
    nf = gaps[~gaps["bank_id"].isin(focus)]
    fg = gaps[gaps["bank_id"].isin(focus)]
    held = documents_held(root)
    lines = [
        "# Data audit: chapters 4-8",
        "",
        "Generated by `python -m tzbank run audit` (`tzbank.reporting.audit`). Blueprint work-sequence step 3; "
        "prompt 01 section 8. Source: `data/processed/bank_period_long.csv` (FY, the bank's own entity - Group or Bank solo - any basis). "
        "Bank-year coverage file: `reports/data_audit_coverage.csv`.",
        "",
        "Audit range per bank: from listing year (NMB FY2008, CRDB FY2009, decision D6) to FY2025. A variable is "
        "available when a value exists on any basis; basis mixing is flagged in `bank_year_wide.csv`.",
        "",
        "## 1. Which analyses the data supports",
        "",
        f"A bank counts as covered when every input is available in at least {SUPPORT_THRESHOLD:.0%} of its years "
        "(alternatives such as net profit for attributable profit allowed and marked). External inputs (T-bill "
        "yields, annual COE, reserve requirements) are not bank-level and are listed in section 4.",
        "",
        _md_table(support.drop(columns=["evidence_class"]), "{:.2f}"),
        "",
        "## 2. Coverage matrix (years available / years in range)",
        "",
        _md_table(mat),
        "",
        "## 3. Gaps for the five non-focus banks (B03-B07)",
        "",
        "Inputs required by chapters 4-8 (prompt 01 section 8.2). Recommendation: extract where an annual report for "
        "the year (or the next year, for comparatives) is held in `data/raw/documents/`; otherwise accept as a "
        "limitation (blueprint L5). Extraction adds rows for B03-B07 to the inputs workbook's Historical Financials "
        "sheet (same schema); Phase 1 is not edited.",
        "",
        _md_table(nf.drop(columns=["evidence_class"])),
        "",
        "### Gaps for NMB and CRDB",
        "",
        _md_table(fg.drop(columns=["evidence_class"])),
        "",
        "### Annual-report PDFs held (by year in file name)",
        "",
        _md_table(pd.DataFrame([{"bank": names[b], "years": _ranges(sorted(y))} for b, y in held.items()])),
        "",
        "## 4. Remaining gaps by reason (after prompt 02 extraction)",
        "",
        "From `reports/pdf_inspection.csv` (held PDFs checked, text layer or OCR). Only 'year not held' and 'not in "
        "held PDFs' become owner collection tasks (`reports/collection_tasks.csv`).",
        "",
        _md_table(remaining_by_reason(cov, root)),
        "",
        "## 5. External series still needed (not bank-level)",
        "",
        "- Treasury bond yields for the annual COE: see `reports/data_audit_auction_yields_needed.csv` (2012-2025) "
        "and long-bond yields for 2008-2011 (BoT Monthly Economic Review).",
        "- T-bill yield history, CBR and statutory reserve requirement history (chapters 4.2 and 6).",
        "- Deposit-mix, fee and loan-sector notes (chapters 2, 4.2, 4.4, 6): to extract from held PDFs.",
        "- BoT sector totals and credit by activity (chapters 1-3): `data/raw/bot/` is empty.",
        "",
        "Evidence class: coverage counts are Observed/Derived; recommendations are Interpretation.",
    ]
    return "\n".join(lines)


def write_audit(root: Path | None = None) -> list[Path]:
    """Write reports/data_audit.md, data_audit_coverage.csv, data_audit_gaps.csv, data_audit_analysis_support.csv."""
    root = root or find_root()
    cov = coverage_long(root)
    gaps = gap_recommendations(cov, root)
    support = analysis_support(cov, root)
    rdir = path("reports", root)
    return [
        write_csv(cov, rdir / "data_audit_coverage.csv", require_evidence=True),
        write_csv(gaps, rdir / "data_audit_gaps.csv", require_evidence=True),
        write_csv(support, rdir / "data_audit_analysis_support.csv", require_evidence=True),
        write_text(render(cov, gaps, support, root), rdir / "data_audit.md"),
    ]
