"""Find document dates in report text (prompt 02 section 2.1; D16, D24).

- ``fs_board_approval_date``: the board's approval of the financial statements ("approved by the Board of Directors
  on ...", "authorised for issue ... on ..."). Integrated-report approvals are not used (D24).
- ``auditor_report_date``: the date on the independent auditor's report page.
- ``agm_notice_date``: a dated notice of the annual general meeting inside the report (evidence the report was
  public by then: an *actual* publication date).

Each hit carries the page and a short quotation so it can be checked. Ambiguous pages return nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pymupdf

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = (rf"(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?({MONTHS}),?\s+(\d{{4}})"
        rf"|({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}}))")
DATE_RX = re.compile(DATE, re.I)
APPROVAL_RX = re.compile(r"(approved|authori[sz]ed for issue)[^.]{0,120}?board of directors[^.]{0,40}?\bon\b\s*"
                         + DATE + r"|board of directors[^.]{0,60}?(approved|authori[sz]ed)[^.]{0,80}?\bon\b\s*" + DATE,
                         re.I | re.S)
AUDITOR_PAGE_RX = re.compile(r"independent auditor", re.I)
AGM_RX = re.compile(r"notice is hereby given[^.]{0,200}annual general meeting", re.I | re.S)


def _to_date(groups: tuple) -> pd.Timestamp | None:
    g = [x for x in groups if x]
    for i in range(len(g) - 2):
        a, b, c = g[i], g[i + 1], g[i + 2]
        for d, m, y in ((a, b, c), (b, a, c)):
            try:
                return pd.Timestamp(f"{int(d)} {m} {int(y)}")
            except (ValueError, TypeError):
                continue
    return None


def _ok(d: pd.Timestamp | None, lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
    """Inside the window and not a 31 December (period-end references are not signing dates)."""
    return d is not None and lo < d <= hi and not (d.month == 12 and d.day == 31)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def find_dates(pdf: Path, fiscal_year: int | None = None) -> dict[str, object]:
    """Dates with page and quote; only dates after the fiscal year end and within 15 months are kept."""
    doc = pymupdf.open(pdf)
    lo = pd.Timestamp(f"{fiscal_year}-12-31") if fiscal_year else pd.Timestamp("1990-01-01")
    hi = lo + pd.DateOffset(months=15) if fiscal_year else pd.Timestamp("2100-01-01")
    out: dict[str, object] = {}
    for i, page in enumerate(doc, start=1):
        t = _norm(page.get_text())
        if "fs_board_approval_date" not in out:
            m = APPROVAL_RX.search(t)
            if m:
                d = _to_date(m.groups())
                if _ok(d, lo, hi):
                    out |= {"fs_board_approval_date": d, "fs_board_approval_page": i,
                            "fs_board_approval_quote": m.group(0)[:160]}
        if "auditor_report_date" not in out and AUDITOR_PAGE_RX.search(t) and re.search(r"(?i)\bdate\b|dar es salaam",
                                                                                        t[-600:]):
            ds = [(_to_date(m.groups()), m.group(0)) for m in DATE_RX.finditer(t[-600:])]
            ds = [(d, q) for d, q in ds if _ok(d, lo, hi)]
            if ds:
                out |= {"auditor_report_date": ds[-1][0], "auditor_report_page": i,
                        "auditor_report_quote": ds[-1][1]}
        if "agm_notice_date" not in out and AGM_RX.search(t):
            ds = [(_to_date(m.groups()), m.group(0)) for m in DATE_RX.finditer(t)]
            ds = [(d, q) for d, q in ds if _ok(d, lo, hi)]
            if ds:
                # the notice is dated before the meeting: take the earliest date on the notice page
                d, q = min(ds, key=lambda x: x[0])
                out |= {"agm_notice_date": d, "agm_notice_page": i, "agm_notice_quote": q}
    doc.close()
    return out
