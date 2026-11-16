"""Rule-based candidate picker over PDF text layers (prompt 02 section 1). Picks are candidates for review, never
written straight to the workbook: ``tzbank.extract.ledger`` holds the reviewed values with page references.

Each full annual report holds the current year (first number column) and the prior year (second column).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tzbank.extract.pdftext import Row, page_rows, page_unit, text_pages

SCALE = {"000": 1_000.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}


@dataclass
class Pick:
    item: str
    page: int
    label: str
    current: float | None
    prior: float | None
    unit: str | None
    rule: str


def _amounts(r: Row) -> list[float]:
    """Numbers that look like amounts (drop note references and years)."""
    out = []
    for n in r.numbers:
        if abs(n) < 100 and float(n).is_integer():
            continue  # note refs, line numbers
        if 1990 <= abs(n) <= 2035 and float(n).is_integer():
            continue  # years
        out.append(n)
    return out


def _page_text(rows: list[Row]) -> str:
    return " ".join(r.text for r in rows).lower()


IS_RULES = {
    "interest_income": r"^(total )?interest (and similar )?income( calculated using the effective interest( rate)?( method)?)?( \d+)?$",
    "interest_expense": r"^(total )?interest (and similar )?expenses?( \d+)?$",
    "net_interest_income": r"^net interest income( \d+)?$",
    "fee_commission_income": r"^(net )?fees? and commission (income|revenue)( \d+)?$",
    "non_interest_income": r"^total non[- ]interest income|^non[- ]interest income( \d+)?$",
    "operating_expenses": r"^total (operating )?expenses( \d+)?$|^operating expenses( \d+)?$",
    "impairment_charges": r"(credit )?impairment (losses|charges?|on loans)|expected credit loss|^impairment|"
                          r"^(net )?credit loss",
    "profit_before_tax": r"^\(?(loss|profit)\)?/?\(?(loss|profit)?\)? before (income )?tax",
    "income_tax_expense": r"^income tax|^tax(ation)? (charge|expense|credit)",
    "net_profit": r"^\(?(loss|profit)\)?/?\(?(loss|profit)?\)? for the (year|period)$|^net (profit|loss)",
}
BS_RULES = {
    "customer_deposits": r"^deposits? (from|due to|by) customers( [\d.a-z]+)?$|^customer deposits$",
    "total_equity": r"^total (equity|shareholders'? (funds|equity))$",
    "total_assets": r"^total assets$",
    "net_loans": r"^loans and advances to customers( [\d.a-z]+)?$",
}
CAP_RULES = {
    "rwa_total_adjusted": r"^total adjusted risk[- ]weighted assets",
    "rwa_total": r"^total risk[- ]weighted assets( including.*)?$",
    "tier1_capital": r"^total qualifying tier ?(1|i) capital|^tier ?(1|i) capital$|^core capital$",
}
DEPOSIT_TYPES = {
    "interest_expense_time_deposits": r"^(time|fixed|term)( and call)? deposits?( accounts)?$",
    "interest_expense_savings_deposits": r"^savings? (deposits?|accounts?)$",
    "interest_expense_current_accounts": r"^(current|demand) (accounts?|deposits?)$",
    "interest_expense_call_deposits": r"^call (deposits?|accounts?)$",
    "interest_expense_customer_deposits_total": r"^deposits? (from|due to|by) customers$|^customer deposits$",
    "interest_expense_borrowings": r"^(interest (expense )?on )?borrowings?( \d+)?$",
    "interest_expense_bank_deposits": r"^deposits? (from|due to|by) (other )?banks",
}


def _first(rows: list[Row], rx: re.Pattern, unit: str | None, page: int, item: str, rule: str) -> Pick | None:
    for r in rows:
        if rx.search(r.label.strip().lower()):
            a = _amounts(r)
            if a:
                return Pick(item, page, r.label[:80], a[0], a[1] if len(a) > 1 else None, unit, rule)
    return None


def statement_pages(pdf: Path) -> dict[str, list[int]]:
    """Candidate pages: income statement, balance sheet, capital note, interest-expense note."""
    out: dict[str, list[int]] = {"is": [], "bs": [], "cap": [], "ie": []}
    for p in text_pages(pdf):
        rows = page_rows(pdf, p)
        t = _page_text(rows)
        if re.search(r"interest (and similar )?income", t) and re.search(r"before (income )?tax", t) \
                and "net interest income" in t:
            out["is"].append(p)
        if "total assets" in t and re.search(r"deposits? (from|due to) customers|customer deposits", t) \
                and "total liabilities" in t:
            out["bs"].append(p)
        if re.search(r"risk[- ]weighted assets", t) and re.search(r"tier ?(1|i)|core capital", t):
            out["cap"].append(p)
        if re.search(r"interest (and similar )?expense", t) and re.search(r"savings|time deposits|fixed deposits|"
                                                                          r"current accounts", t):
            out["ie"].append(p)
    return out


def pick_document(pdf: Path) -> list[Pick]:
    """Best candidate per item from the first matching statement page (statements precede notes)."""
    pages = statement_pages(pdf)
    picks: list[Pick] = []
    for kind, rules in [("is", IS_RULES), ("bs", BS_RULES), ("cap", CAP_RULES), ("ie", DEPOSIT_TYPES)]:
        for item, pat in rules.items():
            rx = re.compile(pat, re.I)
            for p in pages[kind]:
                rows = page_rows(pdf, p)
                pk = _first(rows, rx, page_unit(pdf, p), p, item, kind)
                if pk:
                    picks.append(pk)
                    break
    return picks


def scaled(v: float | None, unit: str | None) -> float | None:
    """Full TZS from a value in the document's unit."""
    if v is None or unit not in SCALE:
        return None
    return v * SCALE[unit]
