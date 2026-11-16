"""Publication dates (prompt 02 section 2; decisions D16, D24, D25; blueprint limitation L9).

Order of evidence for each source document:

1. **actual**: a dated publication notice, DSE announcement, dated web posting or dated AGM notice in the report
   (column ``Actual Publication Date`` with its evidence in ``How Publication Date Established``);
2. **estimated**: the regulatory deadline for the statement type and period (table ``RULES``). Never earlier than
   the rule without evidence.

Board approval and auditor dates are recorded separately (``FS Board Approval Date``, ``Auditor Report Date``) and are
never publication dates. The 15-day rule counts from the board's approval of the financial statements (D24).
Non-bank sources (market data, statistics, datasets) are dated by their own posting or release date (actual), else a
stated estimate. A date known only as a lower bound (e.g. a PDF's last-modified date) is typed ``lower_bound`` (D35):
kept for descriptive history, ineligible at any valuation date (``tzbank.pit.eligible``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Regulation provisions. Texts, commencement and periods: docs/PUBLICATION_DATE_RULES.md.
REG_2001_Q = "BFI (Financial Statements) Regulations 2001 (GN 103 of 2001), reg. 6"
REG_2001_A = "BFI (Financial Statements) Regulations 2001 (GN 103 of 2001), reg. 8"
REG_2014_Q = "BFI (Disclosures) Regulations 2014 (GN 289 of 2014), reg. 7(3)"
REG_2014_A = "BFI (Disclosures) Regulations 2014 (GN 289 of 2014), reg. 8(3)"
ASSUMED = "assumed_2001_deadlines_gn368_text_not_held"
GN368_NOTE = ("BFI (Publication of Financial Statements) Regulations 2008 (GN 368 of 2008, Gazette No. 50 of 12 Dec "
              "2008, notice 632; revoked by GN 289/2014 reg. 20) governed this period; text not held, the 2001 "
              "deadlines are assumed (D36)")

# D36: three periods. GN 368 of 2008 was published in Subsidiary Legislation Supplement No. 41 of 12 Dec 2008
# (commencement inferred from the gazette date). GN 289 of 2014 was published on 22 Aug 2014 and has no commencement
# clause (inferred to operate from publication, D27); quarters ending in 2014 take the longer 45-day deadline.
GN368_GAZETTED = pd.Timestamp("2008-12-12")
GN289_PUBLISHED = pd.Timestamp("2014-08-22")
GN289_FIRST_QUARTER_APPLIED = pd.Timestamp("2015-03-31")


@dataclass(frozen=True)
class Rule:
    name: str
    regulatory_basis: str
    flag: str = ""


def regime(period_end: pd.Timestamp) -> str:
    """'2001' (verified), 'gn368' (2001 deadlines assumed, D36) or '2014' (verified) for a period end."""
    if period_end < GN368_GAZETTED:
        return "2001"
    if period_end < GN289_PUBLISHED:
        return "gn368"
    return "2014"


def quarterly_rule(period_end: pd.Timestamp) -> tuple[pd.Timestamp, Rule]:
    """Quarter-end + 45 days (2001 reg. 6; assumed under GN 368, D36) before GN 289/2014 applies; + 30 days
    (2014 reg. 7(3)) from Q1 2015 (D25, D27)."""
    if period_end >= GN289_FIRST_QUARTER_APPLIED:
        return period_end + pd.Timedelta(days=30), Rule("quarter_end+30d", REG_2014_Q)
    d = period_end + pd.Timedelta(days=45)
    if period_end >= GN289_PUBLISHED:
        return d, Rule("quarter_end+45d", REG_2001_Q, "2014 quarter: GN 289 commencement inferred from gazette date "
                                                       "22 Aug 2014; longer deadline applied (conservative)")
    if regime(period_end) == "gn368":
        return d, Rule("quarter_end+45d", f"{ASSUMED}: {GN368_NOTE}", ASSUMED)
    return d, Rule("quarter_end+45d", REG_2001_Q)


def annual_rule(fye: pd.Timestamp, approval: pd.Timestamp | None) -> tuple[pd.Timestamp, Rule]:
    """Earlier of (FS board approval + 15 days) and (FYE + 105 days); FYE + 105 days without an approval date.

    Same deadlines in 2001 reg. 8 and 2014 reg. 8(3) (both verified from their texts). Annual statements for years
    ending 12 Dec 2008 - 21 Aug 2014 (FY2008-FY2013) fell under GN 368 of 2008, whose text is not held: the 2001
    deadlines are assumed and flagged (D36).
    """
    r = regime(fye)
    if r == "2014":
        basis, flag = REG_2014_A, ""
    elif r == "gn368":
        basis, flag = f"{ASSUMED}: {GN368_NOTE}", ASSUMED
    else:
        basis, flag = REG_2001_A, ""
    cap = fye + pd.Timedelta(days=105)
    if approval is not None and not pd.isna(approval):
        return min(approval + pd.Timedelta(days=15), cap), Rule("min(fs_approval+15d, fye+105d)", basis, flag)
    return cap, Rule("fye+105d", basis, flag)


BANK_STATEMENT_TYPES = {"Annual report", "Audited FS abstract", "Interim", "Condensed annual", "Quarterly"}


def resolve(row: pd.Series) -> dict[str, object]:
    """publication_date, publication_date_type, rule_applied, regulatory_basis, flag for one Sources row."""
    actual = pd.to_datetime(row.get("Actual Publication Date"), errors="coerce")
    doc_type = str(row.get("Document Type") or "")
    period_end = pd.to_datetime(row.get("Period End"), errors="coerce")
    if pd.notna(actual):
        return {"publication_date": actual, "publication_date_type": "actual", "rule_applied": "evidence",
                "regulatory_basis": "n/a (actual date)", "pubdate_flag": ""}
    if doc_type not in BANK_STATEMENT_TYPES:
        est = pd.to_datetime(row.get("Estimated Publication Date"), errors="coerce")
        if pd.notna(est):
            return {"publication_date": est, "publication_date_type": "estimated",
                    "rule_applied": f"stated estimate: {row.get('Estimate Basis') or 'see Date Evidence'}",
                    "regulatory_basis": "n/a (non-bank source; not a regulatory deadline)", "pubdate_flag": ""}
        lower = pd.to_datetime(row.get("Lower-Bound Publication Date"), errors="coerce")
        if pd.notna(lower):
            # D35: a lower bound (e.g. a PDF's last-modified date) is kept for descriptive history but is never
            # eligible at a valuation date (tzbank.pit.eligible) until an actual or estimated date replaces it
            return {"publication_date": lower, "publication_date_type": "lower_bound",
                    "rule_applied": f"lower bound (D35): {row.get('Estimate Basis') or 'see Date Evidence'}",
                    "regulatory_basis": "n/a (non-bank source; lower bound only)",
                    "pubdate_flag": "lower bound only: ineligible at valuation dates until confirmed (D35)"}
        return {"publication_date": pd.NaT, "publication_date_type": "missing", "rule_applied": "",
                "regulatory_basis": "", "pubdate_flag": "non-bank source without a dated posting"}
    if pd.isna(period_end):
        return {"publication_date": pd.NaT, "publication_date_type": "missing", "rule_applied": "",
                "regulatory_basis": "", "pubdate_flag": "no period end"}
    if doc_type in {"Interim", "Quarterly"}:
        d, rule = quarterly_rule(period_end)
    else:
        approval = pd.to_datetime(row.get("FS Board Approval Date"), errors="coerce")
        d, rule = annual_rule(period_end, approval)
    return {"publication_date": d, "publication_date_type": "estimated", "rule_applied": rule.name,
            "regulatory_basis": rule.regulatory_basis, "pubdate_flag": rule.flag}


def resolve_sources(sources: pd.DataFrame) -> pd.DataFrame:
    """Add the D16 columns to the Sources table (recomputed in Python; CLAUDE.md 3.3)."""
    res = pd.DataFrame([resolve(r) for _, r in sources.iterrows()], index=sources.index)
    out = sources.copy()
    for c in res.columns:
        out[c] = res[c]
    out["publication_date"] = pd.to_datetime(out["publication_date"])
    return out


def phase1_unmapped_date(fiscal_year: int) -> pd.Timestamp:
    """Estimated annual date for a Phase 1 row with no mappable document (prompt 02 section 2.6)."""
    d, _ = annual_rule(pd.Timestamp(f"{fiscal_year}-12-31"), None)
    return d


def attach_doc_dates(df: pd.DataFrame, root=None, doc_col: str = "doc_id") -> pd.DataFrame:
    """Add publication_date and publication_date_type from the resolved Sources table (data/interim) by Doc ID, so a
    panel can go through ``tzbank.pit.eligible`` (D35 lower bounds stay ineligible)."""
    from tzbank.config import find_root, path

    f = path("interim", root or find_root()) / "inputs_sources.csv"
    src = pd.read_csv(f, usecols=["Doc ID", "publication_date", "publication_date_type"])
    lut = src.drop_duplicates("Doc ID").set_index("Doc ID")
    out = df.copy()
    out["publication_date"] = pd.to_datetime(out[doc_col].map(lut["publication_date"]), errors="coerce")
    out["publication_date_type"] = out[doc_col].map(lut["publication_date_type"])
    return out
