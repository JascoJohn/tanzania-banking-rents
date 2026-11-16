"""Gated extraction from held PDFs (prompt 02 sections 1.1-1.3).

Every candidate value carries the check that admitted it. A value is accepted only if a gate passes:

- income statement: interest income - interest expense = NII (0.5%) within the same document and column;
- loans: gross loans - allowance = net loans on the balance sheet (0.5%), searched over the note's rows;
- capital: Tier 1 / RWA = the Tier 1 ratio reported in the same note or in Phase 1 (0.5 pp);
- deposit interest: components sum to no more than total interest expense (B03-B07 only; for NMB and CRDB deposit
  interest is accepted only when the note lines sum to the audited total, ``tzbank.extract.ledger``, prompt 02d);
- other lines: within 1% of Phase 1 for the same bank-year, or the line sits on the gated income statement page.

Candidates that pass no gate are written for manual review, never to the workbook.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.extract.condensed import parse_condensed
from tzbank.extract.pdftext import Row, page_rows, page_unit, text_pages
from tzbank.extract.picker import SCALE, _amounts, pick_document


@dataclass
class Candidate:
    bank_id: str
    fiscal_year: int
    variable: str  # inputs-workbook variable name
    value: float | None  # full TZS (or ratio)
    file: str
    page: int
    label: str
    column: str  # current / prior
    gate: str
    gate_passed: bool
    confidence: str
    notes: str = ""
    extra: dict = field(default_factory=dict)


IS_MAP = {"interest_income": "Interest Income", "interest_expense": "Interest Expense",
          "net_interest_income": "Net Interest Income", "non_interest_income": "Non-Interest Income",
          "operating_expenses": "Operating Expenses", "impairment_charges": "Impairment Charges",
          "profit_before_tax": "Profit Before Tax", "income_tax_expense": "Income Tax Expense",
          "net_profit": "Net Profit", "fee_commission_income": "Fee and Commission Income"}
BS_MAP = {"customer_deposits": "Customer Deposits", "total_equity": "Total Equity", "total_assets": "Total Assets",
          "net_loans": "Loans and Advances to Customers (Net)"}
DEP_MAP = {"interest_expense_time_deposits": "Interest Expense on Time Deposits",
           "interest_expense_savings_deposits": "Interest Expense on Savings Deposits",
           "interest_expense_current_accounts": "Interest Expense on Current Accounts",
           "interest_expense_call_deposits": "Interest Expense on Call Deposits",
           "interest_expense_customer_deposits_total": "Interest Expense on Customer Deposits",
           "interest_expense_borrowings": "Interest Expense on Borrowings"}
EXPENSE_LIKE = {"Interest Expense", "Operating Expenses", "Impairment Charges", "Income Tax Expense"}


def fiscal_year_of(pdf: Path, page: int) -> int | None:
    """Fiscal year of the 'current' column: a heading row holding 'YYYY YYYY-1' (else the most frequent years)."""
    for r in page_rows(pdf, page):
        ys = [int(t) for t in re.findall(r"\b(20[0-3]\d)\b", r.text)]
        for a, b in zip(ys, ys[1:], strict=False):
            if a - b == 1 and 2000 <= a <= 2026:
                return a
    t = pymupdf.open(pdf)[page - 1].get_text()
    ys = [int(y) for y in re.findall(r"\b(20[0-3]\d)\b", t)]
    ys = [y for y in ys if 2000 <= y <= 2026]
    if not ys:
        return None
    # the two most frequent years are the current and prior columns
    s = pd.Series(ys).value_counts()
    top = sorted(s.index[:2])
    return int(max(top))


def _pos(v: float | None, variable: str) -> float | None:
    """Expenses are recorded positive (inputs-workbook convention); other lines keep their sign."""
    if v is None:
        return None
    return abs(v) if variable in EXPENSE_LIKE or variable.startswith("Interest Expense") else v


def _close(a: float | None, b: float | None, tol: float) -> bool:
    return a is not None and b is not None and b != 0 and abs(a - b) / abs(b) <= tol


def identity_interest(pdf: Path, page: int) -> tuple | None:
    """Rows above 'Net interest income' with one income row and one or two expense rows summing to NII in both
    columns (0.1%). Returns ((cur_ii, pri_ii, label), (cur_ie, pri_ie, labels)) or None. Label-independent."""
    from itertools import combinations

    rows = page_rows(pdf, page)
    idx = next((i for i, r in enumerate(rows) if re.match(r"^net interest income\s*$", r.label.strip(), re.I)
                and len(_amounts(r)) >= 2), None)
    if idx is None:
        return None
    def last2(r: Row) -> list[float]:
        # the two year columns are the last two numbers; dashes are nil (0)
        ns = [n for n in r.numbers if not (1990 <= abs(n) <= 2035 and float(n).is_integer())]
        return ns[-2:] if len(ns) >= 2 else []

    nii = last2(rows[idx])
    if len(nii) < 2:
        return None
    above = [r for r in rows[max(0, idx - 8):idx] if len(last2(r)) == 2 and not re.search(r"(?i)net interest", r.label)]
    _amounts_local = last2
    for r_in in above:
        a = _amounts_local(r_in)
        if a[0] <= 0:
            continue
        others = [r for r in above if r is not r_in]
        for k in (1, 2):
            for combo in combinations(others, k):
                ex = [_amounts_local(r) for r in combo]
                cur = a[0] - sum(abs(e[0]) for e in ex)
                pri = a[1] - sum(abs(e[1]) for e in ex)
                if _close(cur, nii[0], 0.001) and _close(pri, nii[1], 0.001):
                    return ((a[0], a[1], r_in.label or "(label not in text layer)"),
                            (sum(abs(e[0]) for e in ex), sum(abs(e[1]) for e in ex),
                             " + ".join(r.label or "(label not in text layer)" for r in combo)))
    return None


def full_report_candidates(bank_id: str, pdf: Path) -> list[Candidate]:
    """Income statement, balance sheet, loans, capital and interest-expense note from one full report."""
    from tzbank.extract.picker import Pick

    picks = {p.item: p for p in pick_document(pdf)}
    anchor = picks.get("net_interest_income") or picks.get("interest_income")
    if anchor is None:
        return []
    is_page = anchor.page
    ident = identity_interest(pdf, is_page)
    if ident:
        (ci, pi_, li), (ce, pe, le) = ident
        unit = page_unit(pdf, is_page)
        picks["interest_income"] = Pick("interest_income", is_page, li[:80], ci, pi_, unit, "identity")
        picks["interest_expense"] = Pick("interest_expense", is_page, le[:80], -ce, -pe, unit, "identity")
    fy = fiscal_year_of(pdf, is_page)
    if fy is None:
        return []
    out: list[Candidate] = []
    rel = f"{pdf.parent.name}/{pdf.name}"

    def sc(v, unit):
        return None if v is None or unit not in SCALE else v * SCALE[unit]

    for col, year in [("current", fy), ("prior", fy - 1)]:
        conf = "HIGH" if col == "current" else "MEDIUM"
        val = {k: sc(getattr(p, col), p.unit) for k, p in picks.items()}
        ii, ie, nii = val.get("interest_income"), val.get("interest_expense"), val.get("net_interest_income")
        is_ok = ii is not None and ie is not None and nii is not None and _close(ii - abs(ie), nii, 0.005)
        for item, var in IS_MAP.items():
            if item in picks and picks[item].page == is_page and val.get(item) is not None:
                gate = "II-IE=NII on statement page" if item in {"interest_income", "interest_expense",
                                                                    "net_interest_income"} else "on gated IS page"
                out.append(Candidate(bank_id, year, var, _pos(val[item], var), rel, picks[item].page,
                                     picks[item].label, col, gate, is_ok, conf))
        for item, var in BS_MAP.items():
            if item in picks and val.get(item) is not None:
                out.append(Candidate(bank_id, year, var, val[item], rel, picks[item].page, picks[item].label, col,
                                     "balance sheet (Phase 1 check)", False, conf))
        # interest-expense note: deposit-type components
        comps = {k: v for k, v in val.items() if k in DEP_MAP and v is not None and picks[k].rule == "ie"}
        dep_parts = [abs(v) for k, v in comps.items() if k in {"interest_expense_time_deposits",
                                                              "interest_expense_savings_deposits",
                                                              "interest_expense_current_accounts",
                                                              "interest_expense_call_deposits"}]
        total_dep = sum(dep_parts) if dep_parts else (abs(comps["interest_expense_customer_deposits_total"])
                                                     if "interest_expense_customer_deposits_total" in comps else None)
        dep_ok = total_dep is not None and ie is not None and 0.2 * abs(ie) <= total_dep <= abs(ie) * 1.001
        if bank_id in {"B01", "B02"}:
            # prompt 02d 1.3: NMB/CRDB deposit interest only through the note reconciliation (sum to total)
            dep_ok = False
        for k, v in comps.items():
            out.append(Candidate(bank_id, year, DEP_MAP[k], abs(v), rel, picks[k].page, picks[k].label, col,
                                 "deposit interest <= total interest expense", dep_ok, conf))
        if dep_parts and total_dep is not None:
            out.append(Candidate(bank_id, year, "Interest Expense on Customer Deposits", total_dep, rel,
                                 next(picks[k].page for k in comps), "sum of deposit-type lines", col,
                                 "deposit interest <= total interest expense", dep_ok, conf,
                                 notes="COMPUTED = sum of " + ", ".join(picks[k].label for k in comps
                                                                         if k != "interest_expense_borrowings")))
        # capital
        if "rwa_total_adjusted" in picks:
            picks["rwa_total"] = picks["rwa_total_adjusted"]
        cap = [picks[k] for k in ("rwa_total", "tier1_capital") if k in picks]
        if len(cap) == 2:
            rwa, t1 = sc(getattr(picks["rwa_total"], col), picks["rwa_total"].unit), sc(
                getattr(picks["tier1_capital"], col), picks["tier1_capital"].unit)
            ratio = (t1 / rwa) if rwa and t1 else None
            for item, var, v in [("rwa_total", "Total RWA", rwa), ("tier1_capital", "Tier 1 Capital", t1)]:
                out.append(Candidate(bank_id, year, var, v, rel, picks[item].page, picks[item].label, col,
                                     "Tier1/RWA vs reported Tier 1 ratio", False, conf,
                                     extra={"implied_tier1_ratio": ratio}))
        # loans: search note rows for a gross/allowance pair reconciling to balance-sheet net loans
        nl = val.get("net_loans")
        if nl:
            pair = find_gross_allowance(pdf, nl, col)
            if pair:
                (g, gp, gl), (a, ap, al) = pair
                out.append(Candidate(bank_id, year, "Gross Loans and Advances to Customers", g, rel, gp, gl, col,
                                     "gross - allowance = BS net loans (0.5%)", True, conf))
                out.append(Candidate(bank_id, year, "Loss Allowance on Loans (ECL)", a, rel, ap, al, col,
                                     "gross - allowance = BS net loans (0.5%)", True, conf))
    return out


GROSS_RX = re.compile(r"gross|total loans|loans and advances to customers|total$", re.I)
ALLOW_RX = re.compile(r"allowance|impairment|provision|expected credit|ecl|less", re.I)


def find_gross_allowance(pdf: Path, net_loans: float, col: str) -> tuple | None:
    """First (gross, allowance) amounts on one page with gross - allowance = net loans within 0.5%."""
    for p in text_pages(pdf):
        rows = page_rows(pdf, p)
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        s = SCALE[unit]
        gross = [(abs(n) * s, r.label) for r in rows if GROSS_RX.search(r.label) for n in _amounts(r)]
        allow = [(abs(n) * s, r.label) for r in rows if ALLOW_RX.search(r.label) for n in _amounts(r)]
        for g, gl in gross:
            for a, al in allow:
                if a < g and _close(g - a, net_loans, 0.005):
                    return (g, p, gl[:70]), (a, p, al[:70])
    return None


def condensed_candidates(bank_id: str, pdf: Path) -> list[Candidate]:
    """BoT-format condensed statement (DSE publication): income statement lines gated by II - IE = NII."""
    rows = parse_condensed(pdf)
    if not rows:
        return []
    t = " ".join(p.get_text() for p in pymupdf.open(pdf))
    m = re.findall(r"(?i)(?:year|period|quarter)[ /a-z]{0,10}ended[^0-9]{0,20}(?:31|30)(?:st)?[ ./-]*(?:dec\w*|12)[ ,./-]*(20\d\d)", t)
    fy = int(max(m)) if m else None
    if fy is None:
        return []
    first = {}
    for r in rows:
        first.setdefault(r["item"], r)
    rel = f"{pdf.parent.name}/{pdf.name}"
    out = []
    for col, year in [("current", fy), ("prior", fy - 1)]:
        def v(item, _col=col):
            r = first.get(item)
            return None if r is None or r["unit"] not in SCALE else r[_col] * SCALE[r["unit"]]

        ii, ie, nii = v("interest_income"), v("interest_expense"), v("net_interest_income")
        ok = ii is not None and ie is not None and nii is not None and _close(ii - abs(ie), nii, 0.005)
        conf = "HIGH" if col == "current" else "MEDIUM"
        for item, var in [("interest_income", "Interest Income"), ("interest_expense", "Interest Expense"),
                          ("net_interest_income", "Net Interest Income"),
                          ("income_tax_expense", "Income Tax Expense"), ("customer_deposits", "Customer Deposits"),
                          ("non_interest_income", "Non-Interest Income"), ("net_profit", "Net Profit"),
                          ("impairment_charges", "Impairment Charges"), ("net_loans",
                                                                         "Loans and Advances to Customers (Net)"),
                          ("total_shareholders_funds", "Total Equity")]:
            if v(item) is None:
                continue
            gate = "II-IE=NII (condensed, same column set)" if item in {"interest_income", "interest_expense",
                                                                           "net_interest_income"} else \
                "condensed statement (Phase 1 check)"
            out.append(Candidate(bank_id, year, var, _pos(v(item), var), rel, first[item]["page"],
                                 first[item]["label"], col, gate, ok if "NII" in gate else False, conf,
                                 notes=f"BoT condensed format, columns={first[item]['columns']}"))
    return out


ARITHMETIC_GATES = ("II-IE=NII", "gross - allowance")


def candidates_for(bank_id: str, folder: Path, root: Path | None = None) -> pd.DataFrame:
    """All candidates from every PDF of a bank: text layer as is, scanned PDFs through their OCR copy.

    OCR values keep only arithmetic gates (II - IE = NII; gross - allowance = net loans). Everything else from an
    OCR copy is flagged for manual verification until a cross-source gate (Phase 1, Tier 1 ratio) passes it.
    """
    from tzbank.extract.ocr import extraction_source

    rows = []
    for pdf in sorted(folder.glob("*.pdf")):
        src = extraction_source(pdf, root)
        is_ocr = src != pdf
        if not text_pages(src):
            continue
        n = pymupdf.open(src).page_count
        try:
            cands = condensed_candidates(bank_id, src) if n <= 6 else []
            if not any(c.variable == "Interest Income" for c in cands):
                # short PDFs laid out like full statements (and OCR copies) go through the full-report picker
                cands = full_report_candidates(bank_id, src) or cands
        except Exception as exc:  # noqa: BLE001 - inspection must continue past one bad file
            cands = [Candidate(bank_id, 0, "ERROR", None, pdf.name, 0, str(exc)[:80], "", "error", False, "")]
        for c in cands:
            c.extra = dict(c.extra or {}, ocr=is_ocr)
            if is_ocr and not c.gate.startswith(ARITHMETIC_GATES):
                c.gate_passed = False
                c.gate += " [OCR: manual verification needed]"
            if c.variable == "Impairment Charges" and c.gate == "on gated IS page":
                c.gate_passed = False  # impairment labels are ambiguous (charges vs write-backs); Phase 1 gate only
        rows += [asdict(c) for c in cands]
    df = pd.DataFrame(rows)
    if len(df):
        df["ocr"] = df["extra"].map(lambda e: bool((e or {}).get("ocr")))
    return df


PHASE1_CHECK = {"Net Interest Income": "net_interest_income", "Net Profit": "net_profit",
                "Customer Deposits": "customer_deposits", "Loans and Advances to Customers (Net)": "net_loans",
                "Total Equity": "total_equity", "Non-Interest Income": "non_interest_income",
                "Operating Expenses": "operating_expenses", "Impairment Charges": "impairment_charges",
                "Profit Before Tax": "profit_before_tax", "Total Assets": "total_assets"}


def apply_phase1_gate(c: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Compare each candidate with Phase 1 (same bank, year); pass lines within 1% that had no stronger gate."""
    p1 = canon[(canon["provenance"] == "phase1") & canon["value"].notna() & (canon["period_type"] == "FY")]
    lut = {(b, int(y), n): v for b, y, n, v in zip(p1["bank_id"], p1["fiscal_year"], p1["canonical_name"],
                                                     p1["value"], strict=False)}
    c = c.copy()
    c["phase1_value"] = [lut.get((b, int(y), PHASE1_CHECK.get(v, "")))
                         for b, y, v in zip(c["bank_id"], c["fiscal_year"], c["variable"], strict=False)]
    c["phase1_rel_diff"] = (c["value"].abs() - c["phase1_value"].abs()) / c["phase1_value"].abs()
    match = c["phase1_rel_diff"].abs() <= 0.01
    c.loc[~c["gate_passed"] & match, "gate"] = c.loc[~c["gate_passed"] & match, "gate"] + " -> matches Phase 1 (1%)"
    c.loc[match, "gate_passed"] = True
    return c


def apply_capital_gate(c: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 / RWA within 0.5 pp of the Phase 1 Tier 1 ratio for the same bank-year."""
    p1 = canon[(canon["provenance"] == "phase1") & (canon["canonical_name"] == "tier1_ratio") & canon["value"].notna()]
    lut = {(b, int(y)): v for b, y, v in zip(p1["bank_id"], p1["fiscal_year"], p1["value"], strict=False)}
    c = c.copy()
    cap = c["variable"].isin(["Total RWA", "Tier 1 Capital"])
    for i in c.index[cap]:
        implied = (c.at[i, "extra"] or {}).get("implied_tier1_ratio")
        ref = lut.get((c.at[i, "bank_id"], int(c.at[i, "fiscal_year"])))
        if implied is not None and ref is not None and abs(implied - ref) <= 0.005:
            c.at[i, "gate_passed"] = True
            c.at[i, "gate"] = f"Tier1/RWA {implied:.4f} vs Phase 1 ratio {ref:.4f}"
        elif implied is not None:
            c.at[i, "gate"] = f"Tier1/RWA {implied:.4f} vs Phase 1 ratio {ref}"
    return c
