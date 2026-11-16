"""Parser for BoT-format condensed statements published on the DSE (1-5 page PDFs; BFI Financial Statements
Regulations 2001, First-Third Schedules). Lines read 'n | Label | current | prior'; amounts are in the unit stated
on the page (usually TZS million or TZS '000).
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from tzbank.extract.pdftext import parse_number

NUMTOK = re.compile(r"^\(?-?[\d,]+(\.\d+)?\)?%?$|^-$")
UNIT = re.compile(r"amounts? in (tzs |tshs? )?(million|'?000|thousand)|(tzs|tshs?)\s*['’]?\s*(000|million|m)\b|"
                  r"in (million|thousands) (of )?(tanzania )?shillings", re.I)

# BoT schedule label -> canonical item (first match wins, case-insensitive)
LABELS = [
    ("interest_income", r"^interest income$"),
    ("interest_expense", r"^interest expenses?$"),
    ("net_interest_income", r"^net interest income"),
    ("bad_debts_written_off", r"^bad debts written"),
    ("impairment_charges", r"^impairment loss(es)? on loans"),
    ("non_interest_income", r"^non[- ]interest income$"),
    ("fees_commissions", r"^fees? and commissions?$"),
    ("non_interest_expenses", r"^non[- ]interest expenses?$"),
    ("operating_income", r"^operating income|^operating profit"),
    ("income_tax_expense", r"^income tax provision|^income tax"),
    ("net_profit", r"^net income ?\(loss\)? after income tax|^net income/\(loss\) after"),
    ("customer_deposits", r"^customer deposits$|^deposits from customers$"),
    ("total_shareholders_funds", r"^total shareholders'? funds"),
    ("net_loans", r"^loans, advances and overdrafts"),
    ("total_assets", r"^total assets$"),
]


def page_unit(text: str) -> str | None:
    m = UNIT.search(text)
    if not m:
        return None
    g = (m.group(2) or m.group(4) or m.group(5) or "").lower().strip("'")
    return "million" if g.startswith("m") else "000"


NUM_IN_LINE = r"(\(?-?[\d][\d,]*(?:\.\d+)?\)?|-)"
LINE_LABELS = [
    ("interest_income", r"interest income"),
    ("interest_expense", r"interest expenses?"),
    ("net_interest_income", r"net interest income(?: \(1 minus 2\))?"),
    ("impairment_charges", r"impairment loss(?:es)? on loans and advances"),
    ("non_interest_income", r"non[- ]interest income"),
    ("non_interest_expenses", r"non[- ]interest expenses?"),
    ("income_tax_expense", r"income tax provision"),
    ("net_profit", r"net income ?/?\(?loss\)? after income tax"),
    ("customer_deposits", r"customer deposits"),
    ("total_shareholders_funds", r"total shareholders'? funds"),
]


def parse_lines(pdf: Path) -> list[dict]:
    """Fallback for OCR text layers: label followed by 2 or 4 numbers on the same text line."""
    out = []
    doc = pymupdf.open(pdf)
    for pi, page in enumerate(doc, start=1):
        text = page.get_text()
        unit = page_unit(text)
        for line in text.splitlines():
            low = line.lower()
            for item, lab in LINE_LABELS:
                m = re.search(r"(?:^|[^a-z])" + lab + r"\s*[:|]?\s*" + r"\s+".join([NUM_IN_LINE] * 2)
                              + r"(?:\s+" + NUM_IN_LINE + r"\s+" + NUM_IN_LINE + ")?", low)
                if not m or (item == "interest_income" and "net interest" in low[:m.start() + 15]):
                    continue
                nums = [parse_number(g) for g in m.groups() if g is not None]
                if any(v is None for v in nums):
                    continue
                cur, pri, cols = (nums[2], nums[3], "quarter+year") if len(nums) == 4 else (nums[0], nums[1], "year")
                out.append({"page": pi, "item": item, "label": line.strip()[:60], "current": cur, "prior": pri,
                            "unit": unit, "columns": cols, "raw": nums})
                break
    return out


def parse_condensed(pdf: Path) -> list[dict]:
    """Rows (page, item, label, current, prior, unit) from every text block of a condensed statement."""
    out = []
    doc = pymupdf.open(pdf)
    for pi, page in enumerate(doc, start=1):
        unit = page_unit(page.get_text())
        for b in page.get_text("blocks"):
            toks = [t.strip() for t in b[4].split("\n") if t.strip()]
            for i, t in enumerate(toks):
                if NUMTOK.match(t) or len(t) < 4:
                    continue
                nums = []
                j = i + 1
                while j < len(toks) and NUMTOK.match(toks[j]) and len(nums) < 4:
                    nums.append(parse_number(toks[j].rstrip("%")))
                    j += 1
                if len(nums) < 2:
                    continue
                lab = re.sub(r"\s+", " ", t).strip()
                # Q4 publications show current quarter, comparative quarter, current year, comparative year.
                cur, pri, cols = (nums[2], nums[3], "quarter+year") if len(nums) == 4 else (nums[0], nums[1], "year")
                for item, pat in LABELS:
                    if re.search(pat, lab, re.I):
                        out.append({"page": pi, "item": item, "label": lab, "current": cur, "prior": pri,
                                    "unit": unit, "columns": cols, "raw": nums})
                        break
    return out if any(r["item"] == "interest_income" for r in out) else (parse_lines(pdf) or out)
