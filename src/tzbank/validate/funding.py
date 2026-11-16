"""Prompt 02 section 5.2 checks: funding data and publication-date completeness (non-critical warnings).

Blueprint §2 defines cost of deposits = interest expense on deposits / average deposits.
"""

from __future__ import annotations

import pandas as pd

from tzbank.clean.normalise import PRIMARY_ENTITIES

MIX = ["deposits_current", "deposits_savings", "deposits_time", "deposits_other"]
CHECK_COLUMNS = ["check", "bank_id", "fiscal_year", "value", "status", "detail", "evidence_class"]


def _fy(canon: pd.DataFrame) -> pd.DataFrame:
    """Original FY values of the bank's own entity, one per bank-year-variable."""
    d = canon[(canon["period_type"] == "FY") & canon["entity"].isin(PRIMARY_ENTITIES) & ~canon["is_restated"]
              & canon["value"].notna() & canon["category"].isna()]
    d = d.sort_values(["provenance"]).drop_duplicates(["bank_id", "fiscal_year", "canonical_name"])
    return d.pivot(index=["bank_id", "fiscal_year"], columns="canonical_name", values="value")


def funding_checks(canon: pd.DataFrame) -> pd.DataFrame:
    """Cost of deposits in [0, 15%]; deposit mix sums to 100% +/- 1%; deposit interest <= total interest expense."""
    w = _fy(canon)
    rows = []
    if "interest_expense_deposits" in w and "customer_deposits" in w:
        w = w.sort_index()
        prev = w.groupby(level="bank_id")["customer_deposits"].shift(1)
        gap = w.index.get_level_values("fiscal_year").to_series(index=w.index).groupby(level="bank_id").diff()
        avg = ((w["customer_deposits"] + prev) / 2).where(gap == 1)
        cod = w["interest_expense_deposits"] / avg
        for (b, y), v in cod.dropna().items():
            rows.append({"check": "cost of deposits within 0-15%", "bank_id": b, "fiscal_year": y, "value": v,
                         "status": "PASS" if 0 <= v <= 0.15 else "WARN",
                         "detail": "interest expense on deposits / average customer deposits"})
    have = [c for c in MIX if c in w]
    if have and "customer_deposits" in w:
        s = w[have].sum(axis=1, min_count=2) / w["customer_deposits"]
        for (b, y), v in s.dropna().items():
            rows.append({"check": "deposit mix sums to 100% +/- 1%", "bank_id": b, "fiscal_year": y, "value": v,
                         "status": "PASS" if abs(v - 1) <= 0.01 else "WARN", "detail": "+".join(have)})
    if "interest_expense_deposits" in w and "interest_expense" in w:
        r = (w["interest_expense_deposits"] / w["interest_expense"]).dropna()
        for (b, y), v in r.items():
            rows.append({"check": "deposit interest <= total interest expense", "bank_id": b, "fiscal_year": y,
                         "value": v, "status": "PASS" if v <= 1.0005 else "WARN",
                         "detail": "interest_expense_deposits / interest_expense"})
    out = pd.DataFrame(rows, columns=CHECK_COLUMNS[:-1])
    out["evidence_class"] = "Derived"
    return out


def publication_date_checks(sources: pd.DataFrame) -> pd.DataFrame:
    """Every source has publication_date_type; estimated dates also carry rule_applied and regulatory_basis."""
    rows = []
    for _, s in sources.iterrows():
        t = s.get("publication_date_type")
        problems = []
        if pd.isna(t) or t in ("", "missing", "legacy"):
            problems.append(f"publication_date_type={t}")
        if t == "estimated" and (pd.isna(s.get("rule_applied")) or pd.isna(s.get("regulatory_basis"))):
            problems.append("estimated without rule_applied/regulatory_basis")
        rows.append({"check": "source has publication_date_type (and rule if estimated)", "bank_id": s["Doc ID"],
                     "fiscal_year": None, "value": None, "status": "WARN" if problems else "PASS",
                     "detail": "; ".join(problems), "evidence_class": "Derived"})
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def publication_date_changes(sources: pd.DataFrame, valuation_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Prompt 02 section 2.5: every document whose D16 date differs from its pre-D16 date, with eligibility changes."""
    col = "Legacy Publication Date (pre-D16)"
    if col not in sources.columns:
        return pd.DataFrame()
    s = sources.copy()
    old = pd.to_datetime(s[col], errors="coerce")
    new = pd.to_datetime(s["publication_date"], errors="coerce")
    changed = old.ne(new) & ~(old.isna() & new.isna())
    out = pd.DataFrame({"doc_id": s["Doc ID"], "old_date": old, "new_date": new,
                        "publication_date_type": s.get("publication_date_type"), "rule_applied": s.get("rule_applied"),
                        "days_moved": (new - old).dt.days})[changed]
    for name, d in valuation_dates.items():
        before, after = (old <= d)[changed], (new <= d)[changed]
        out[f"eligible_{name}_before"] = before.map({True: "Y", False: "N"})
        out[f"eligible_{name}_after"] = after.map({True: "Y", False: "N"})
    out["new_document"] = out["old_date"].isna()
    out["eligibility_changed"] = (
        (out.filter(like="_before").to_numpy() != out.filter(like="_after").to_numpy()).any(axis=1)
        & ~out["new_document"].to_numpy())
    out["evidence_class"] = "Derived"
    return out.reset_index(drop=True)
