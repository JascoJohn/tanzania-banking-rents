"""PDF inspection pass (prompt 02 section 1.1): locate each gap item in the held reports and dump candidate rows
for review. Run manually: ``python -m tzbank.extract.inspect <bank folder> [year ...]``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from tzbank.config import find_root, path
from tzbank.extract.pdftext import page_rows, page_unit, text_pages

# item -> label regex. Applied to row labels; 'scope' patterns restrict the pages searched.
ITEMS: dict[str, str] = {
    "interest_income": r"^(total )?interest (and similar )?income\b|^interest income calculated",
    "interest_expense": r"^(total )?interest (and similar )?expenses?\b",
    "net_interest_income": r"^net interest income",
    "interest_expense_deposits": r"deposits? (from|by|due to) customers|customer deposits|^(current|savings|time|"
                                 r"fixed|term|call)\b.*(deposit|account)|^deposits\b",
    "fees": r"fees? and commission|commission income",
    "non_interest_income": r"non[- ]interest income|other (operating )?income|foreign exchange|total operating income",
    "operating_expenses": r"operating expenses|total expenses|general and admin|employee benefit|staff costs",
    "impairment_charges": r"impairment|credit loss|provision for (loan|bad)",
    "profit_before_tax": r"(profit|loss) before (income )?tax",
    "income_tax_expense": r"^income tax|^tax(ation)? (charge|expense)",
    "net_profit": r"(profit|loss) for the (year|period)|net (profit|loss)|profit after tax",
    "gross_loans": r"gross (loans|carrying)|loans and advances.*gross|total gross",
    "loan_loss_allowance": r"allowance|expected credit loss|^less.*impairment|provision for impairment",
    "customer_deposits": r"^deposits? (from|due to) customers|^customer deposits",
    "total_equity": r"^total (equity|shareholders)|shareholders'? funds",
    "rwa_total": r"risk[- ]weighted",
    "tier1_capital": r"tier ?(1|i)\b|core capital",
}
IE_PAGE = re.compile(r"interest (and similar )?expense", re.I)


def dump(pdf: Path, items: dict[str, str] | None = None, min_numbers: int = 1) -> list[tuple]:
    """(item, page, unit, label, numbers) for rows matching each item with at least ``min_numbers`` numbers."""
    items = items or ITEMS
    out = []
    pages = text_pages(pdf)
    for p in pages:
        rows = page_rows(pdf, p)
        text = " ".join(r.text for r in rows)
        unit = page_unit(pdf, p)
        for item, pat in items.items():
            if item == "interest_expense_deposits" and not IE_PAGE.search(text):
                continue
            rx = re.compile(pat, re.I)
            for r in rows:
                if rx.search(r.label) and len([n for n in r.numbers if abs(n) >= 100 or 0 < abs(n) < 1]) >= min_numbers:
                    out.append((item, p, unit, r.label[:70], r.numbers[:6]))
    return out


def subsidiary_mentions(pdf: Path) -> list[tuple[int, str]]:
    """Pages mentioning subsidiaries/consolidation (D23 check)."""
    rx = re.compile(r"subsidiar|consolidated|the group\b", re.I)
    hits = []
    for p in text_pages(pdf):
        for r in page_rows(pdf, p):
            if rx.search(r.text):
                hits.append((p, r.text[:110]))
    return hits


def main(argv: list[str]) -> None:
    folder = path("documents", find_root()) / argv[0]
    years = set(argv[1:])
    for pdf in sorted(folder.glob("*.pdf")):
        if years and not any(y in pdf.name for y in years):
            continue
        print(f"######## {argv[0]}/{pdf.name}  text pages {len(text_pages(pdf))}")
        for item, p, unit, label, nums in dump(pdf):
            print(f"{item[:16]:16s} p{p:<4d} {unit or '?':7s} {label:70s} {nums}")


if __name__ == "__main__":
    main(sys.argv[1:])
