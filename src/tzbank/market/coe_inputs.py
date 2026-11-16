"""Long-bond yield inputs for the cost of equity (decision D34; prompt 02d section 5; blueprint §2 historical cost of
equity, §9 valuation inputs).

For every year-end from 2012 and for both valuation dates (D1) three series are built from BoT Treasury bond auction
results (weighted average yield to maturity, WAYTM):

- ``d34_primary``: the average WAYTM of all auctions with tenor >= 15 years held in the final quarter before the date
  (the three calendar months ending on the date). Fallback, if there were none: the most recent such auction before
  that quarter.
- ``alt_longest_tenor_nearest_year_end`` (sensitivity; the prompt 02 rule): the longest tenor auctioned in the
  12 months to the date, its latest auction with a published result.
- ``alt_most_recent_long_tenor`` (sensitivity): the most recent auction with tenor >= 15 years on or before the date.

Auctions with no published result (BoT FAILURE) never enter. Every auction used is listed in
``data/processed/coe_long_bond_auctions_used.csv``. The BoT calendar's "auction number" is the bond number: reopenings
of one bond share it across auctions and years (e.g. bond 561, 25 years, auctioned in 2021, 2022 and twice in 2024),
so each auction is identified by ``bond_number``, ``issue_number`` (from the result title) and ``auction_date``, with
``bond_auction_seq`` (1 = first auction of the bond in the calendar) and ``calendar_sn`` (the calendar's own S/N).
FY2008-FY2011 precede the auction calendar and stay empty, flagged "needs BoT Monthly Economic Review". Only the
risk-free building block is assembled here; the cost of equity itself is built in chapter 5 (D8).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path, settings, valuation_dates
from tzbank.io import write_csv

LONG_TENOR = 15
SERIES = ["d34_primary", "alt_longest_tenor_nearest_year_end", "alt_most_recent_long_tenor"]
AUCTION_COLS = ["bond_number", "issue_number", "bond_auction_seq", "calendar_sn", "auction_date", "tenor_years",
                "coupon", "waytm", "weighted_average_price"]


def auction_table(auctions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Calendar joined to published results: one row per auction, with bond and issue numbers."""
    a = auctions.rename(columns={"auction_number": "bond_number", "sn": "calendar_sn"}).copy()
    a = a.sort_values(["auction_date", "bond_number"])
    a["bond_auction_seq"] = a.groupby("bond_number").cumcount() + 1
    res = results.rename(columns={"auction_number": "bond_number"})
    res = res[res["status"] == "SUCCESS"].copy()
    res["issue_number"] = pd.to_numeric(res["bond_title"].astype(str).str.extract(r"ISSUE\s*(\d+)", expand=False),
                                        errors="coerce")
    keep = ["bond_number", "auction_date", "issue_number", "waytm", "weighted_average_price"]
    out = a.merge(res.reindex(columns=keep), on=["bond_number", "auction_date"], how="left")
    if "calendar_sn" not in out:
        out["calendar_sn"] = pd.NA
    return out[out["tenor_years"].notna()].reset_index(drop=True)


def _single(series: str, pick: pd.Series, date: pd.Timestamp, method: str, note: str) -> tuple[dict, list[dict]]:
    row = {"series": series, "waytm": pick["waytm"], "method": method, "n_auctions": 1,
           "tenor_years": str(int(pick["tenor_years"])), "bond_number": str(int(pick["bond_number"])),
           "issue_number": "" if pd.isna(pick["issue_number"]) else str(int(pick["issue_number"])),
           "auction_date": pick["auction_date"].date().isoformat(),
           "days_before_date": int((date - pick["auction_date"]).days), "status": "OK", "note": note,
           "evidence_class": "Observed"}
    return row, [pick[AUCTION_COLS].to_dict()]


def d34_primary(a: pd.DataFrame, date: pd.Timestamp) -> tuple[dict, list[dict]]:
    """Average WAYTM of tenor >= 15 auctions in the final quarter to ``date``; else the most recent one before."""
    start = date - pd.DateOffset(months=3)
    first_day = (start + pd.Timedelta(days=1)).date()
    long = a[(a["tenor_years"] >= LONG_TENOR) & a["waytm"].notna() & (a["auction_date"] <= date)]
    q = long[long["auction_date"] > start].sort_values("auction_date")
    if len(q):
        row = {"series": "d34_primary", "waytm": q["waytm"].mean(), "method": "final-quarter average",
               "n_auctions": len(q), "tenor_years": "/".join(str(int(t)) for t in q["tenor_years"]),
               "bond_number": "/".join(str(int(b)) for b in q["bond_number"]),
               "issue_number": "/".join("" if pd.isna(i) else str(int(i)) for i in q["issue_number"]),
               "auction_date": "/".join(d.date().isoformat() for d in q["auction_date"]),
               "days_before_date": int((date - q["auction_date"].max()).days), "status": "OK",
               "note": f"{len(q)} auction(s) with tenor >= {LONG_TENOR} years between {first_day} and "
                       f"{date.date()}; simple average of WAYTM",
               "evidence_class": "Derived" if len(q) > 1 else "Observed"}
        return row, [r[AUCTION_COLS].to_dict() for _, r in q.iterrows()]
    before = long[long["auction_date"] <= start].sort_values("auction_date")
    if before.empty:
        return {"series": "d34_primary", "status": "no auction", "evidence_class": "Observed",
                "note": f"no auction with tenor >= {LONG_TENOR} years on or before {date.date()}"}, []
    pick = before.iloc[-1]
    return _single("d34_primary", pick, date, "fallback: most recent before the final quarter",
                   f"no auction with tenor >= {LONG_TENOR} years in the final quarter to {date.date()}; most recent "
                   f"such auction before it used")


def alt_longest(a: pd.DataFrame, date: pd.Timestamp) -> tuple[dict, list[dict]]:
    """Longest tenor auctioned in the 12 months to ``date``; its latest auction with a published result."""
    sub = a[(a["auction_date"] > date - pd.DateOffset(years=1)) & (a["auction_date"] <= date)]
    if sub.empty:
        return {"series": SERIES[1], "status": "no auction", "evidence_class": "Observed",
                "note": "no auction in the 12 months to the date"}, []
    longest = int(sub["tenor_years"].max())
    cand = sub[sub["tenor_years"] == longest].sort_values("auction_date")
    ok = cand[cand["waytm"].notna()]
    if ok.empty:
        return {"series": SERIES[1], "status": "yield not retrieved", "tenor_years": str(longest),
                "evidence_class": "Observed", "note": f"no {longest}-year auction in the window has a result"}, []
    pick = ok.iloc[-1]
    skipped = cand[cand["auction_date"] > pick["auction_date"]]
    note = (f"longest tenor auctioned in the 12 months to {date.date()}: {longest} years; latest such auction"
            + (f"; {len(skipped)} later {longest}-year auction(s) without a published result skipped "
               f"({', '.join(d.date().isoformat() for d in skipped['auction_date'])})" if len(skipped) else ""))
    return _single(SERIES[1], pick, date, "single auction", note)


def alt_recent(a: pd.DataFrame, date: pd.Timestamp) -> tuple[dict, list[dict]]:
    """Most recent auction with tenor >= 15 years and a published result, on or before ``date``."""
    sub = a[(a["tenor_years"] >= LONG_TENOR) & a["waytm"].notna() & (a["auction_date"] <= date)]
    if sub.empty:
        return {"series": SERIES[2], "status": "no auction", "evidence_class": "Observed",
                "note": f"no auction with tenor >= {LONG_TENOR} years"}, []
    pick = sub.sort_values("auction_date").iloc[-1]
    return _single(SERIES[2], pick, date, "single auction",
                   f"most recent auction with tenor >= {LONG_TENOR} years on or before {date.date()}")


def target_dates(first_year: int, last_year: int, vdates: dict[str, pd.Timestamp]) -> list[tuple[str, pd.Timestamp]]:
    out = [(f"FY{y}", pd.Timestamp(f"{y}-12-31")) for y in range(first_year, last_year + 1)]
    return out + [(f"valuation_{k}", pd.Timestamp(v)) for k, v in vdates.items()]


FLAG_SHORT = "FLAGGED: below the D34 15-year floor (no auction of 15 years or more before 2013)"


def short_tenor_fallback(a: pd.DataFrame, date: pd.Timestamp, bot: dict[int, dict] | None) -> tuple[dict, list[dict]]:
    """Prompt 02f 3: years with no auction of >= 15 years take the longest tenor available, flagged: the longest-tenor
    auction in the final quarter when the calendar holds one (2012), else the BoT rate-structure table's December
    yield for its longest bond tenor (2008-2011, Table A3.13)."""
    start = date - pd.DateOffset(months=3)
    q = a[(a["auction_date"] > start) & (a["auction_date"] <= date) & a["waytm"].notna()]
    if len(q):
        q = q[q["tenor_years"] == q["tenor_years"].max()].sort_values("auction_date")
        row, used = _single(SERIES[0], q.iloc[-1], date, "fallback: longest tenor auctioned in the final quarter",
                            f"{FLAG_SHORT}; {int(q['tenor_years'].iloc[0])}-year auction used")
        row["status"] = "OK (flagged)"
        return row, used
    b = (bot or {}).get(date.year)
    if b is None:
        return {"series": SERIES[0], "status": "needs BoT Monthly Economic Review", "evidence_class": "Observed",
                "note": "no auction held and no BoT rate-structure value for December"}, []
    return {"series": SERIES[0], "waytm": b["waytm"], "method": "fallback: BoT rate structure (A3.13), December, "
            "longest bond tenor shown", "n_auctions": 0, "tenor_years": str(b["tenor"]), "status": "OK (flagged)",
            "note": f"{FLAG_SHORT}; {b['tenor']}-year Treasury bond rate from {b['source']}, {b['page']}",
            "evidence_class": "Observed"}, []


def term_spread_alternative(rows: list[dict], bot: dict[int, dict] | None) -> list[dict]:
    """D41 sensitivity: for flagged short-tenor years, the 10-year value plus the mean 15-year-minus-10-year spread
    (BoT rate structure, December values, over the years where both tenors are printed)."""
    spreads = [(v["spread_15_10"]) for v in (bot or {}).values() if v.get("spread_15_10") is not None]
    if not spreads:
        return []
    m = sum(spreads) / len(spreads)
    out = []
    for r in rows:
        if r["series"] == SERIES[0] and r.get("status") == "OK (flagged)":
            out.append(r | {"series": "d41_term_spread_adjusted", "waytm": r["waytm"] + m,
                            "method": "D41 sensitivity: flagged 10-year value + mean 15y-10y spread",
                            "note": f"mean 15-year minus 10-year spread {m:.4%} over {len(spreads)} years with both "
                                    "tenors (BoT rate structure A3.13, December)", "evidence_class": "Derived"})
    return out


def coe_inputs_annual(auctions: pd.DataFrame, results: pd.DataFrame, first_year: int = 2008, last_year: int = 2025,
                      vdates: dict[str, pd.Timestamp] | None = None,
                      bot_fallback: dict[int, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per date and series, one row per auction used)."""
    a = auction_table(auctions, results)
    first_auction = a["auction_date"].min().date() if len(a) else "n/a"
    rows, used = [], []
    for label, date in target_dates(first_year, last_year, vdates or {}):
        for fn in (d34_primary, alt_longest, alt_recent):
            if date.year < 2012:
                r, u = {"series": fn.__name__, "status": "needs BoT Monthly Economic Review",
                        "evidence_class": "Observed",
                        "note": f"precedes the BoT auction calendar held (first auction {first_auction})"}, []
            else:
                r, u = fn(a, date)
            if fn is d34_primary and r.get("status") in {"no auction", "needs BoT Monthly Economic Review"}:
                r, u = short_tenor_fallback(a, date, bot_fallback)
            r["series"] = {"d34_primary": SERIES[0], "alt_longest": SERIES[1], "alt_recent": SERIES[2]}.get(
                r["series"], r["series"])
            rows.append({"date_label": label, "date": date.date().isoformat(), "year": date.year, **r})
            used += [{"date_label": label, "date": date.date().isoformat(), "series": r["series"], **x} for x in u]
    rows += term_spread_alternative(rows, bot_fallback)
    cols = ["date_label", "date", "year", "series", "waytm", "method", "n_auctions", "tenor_years", "bond_number",
            "issue_number", "auction_date", "days_before_date", "status", "note", "evidence_class"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    u = pd.DataFrame(used)
    if len(u):
        u["auction_date"] = pd.to_datetime(u["auction_date"]).dt.date.astype(str)
        u["evidence_class"] = "Observed"
    return out, u


def comparison(coe: pd.DataFrame) -> pd.DataFrame:
    """D34 primary against the old rule, by date (basis points)."""
    p = coe.pivot(index=["date_label", "date"], columns="series", values="waytm")
    p = p.reindex(columns=SERIES).reset_index()
    p["primary_minus_old_rule_bp"] = (p[SERIES[0]] - p[SERIES[1]]) * 1e4
    p["evidence_class"] = "Derived"
    return p


def write_coe_inputs(root: Path | None = None) -> list[Path]:
    """data/processed/coe_inputs_annual.csv, coe_long_bond_auctions_used.csv, coe_long_bond_comparison.csv."""
    root = root or find_root()
    au = pd.read_csv(path("interim", root) / "bond_auctions.csv", parse_dates=["auction_date"])
    rf = path("interim", root) / "bond_auction_results.csv"
    res = pd.read_csv(rf, parse_dates=["auction_date"]) if rf.exists() and rf.stat().st_size > 5 else pd.DataFrame(
        columns=["auction_number", "auction_date", "waytm", "weighted_average_price", "bond_title", "status"])
    last = settings(root)["auction_years"][1]
    from tzbank.market.bot_tables import long_bond_fallback

    coe, used = coe_inputs_annual(au, res, 2008, last, valuation_dates(root), long_bond_fallback(root))
    p = path("processed", root)
    return [write_csv(coe, p / "coe_inputs_annual.csv", require_evidence=True),
            write_csv(used, p / "coe_long_bond_auctions_used.csv", require_evidence=True),
            write_csv(comparison(coe), p / "coe_long_bond_comparison.csv", require_evidence=True)]
