"""BoT Treasury bond auction calendar (prompt 01 section 3.5; blueprint §2 historical cost of equity, §6 data audit).

The calendar lists auction number, title and date only; yields must be retrieved from each auction result page.
Titles are free text (e.g. '13.00% 20-YEAR TREASURY BOND D', '15.49%-20 YEAR TREASURY BOND', '12.75 15-YEAR ...',
'10.08% 7-TREASURY BOND'); tenor and coupon are parsed with tolerant patterns and never imputed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path, settings
from tzbank.io import write_csv

_COUPON = re.compile(r"^\s*(\d{1,2}(?:\.\d+)?)\s*(%)?")
_TENOR = re.compile(r"(\d{1,2})\s*-?\s*(?:YEARS?\b|-?\s*TREASURY)", re.I)


def parse_title(title: str) -> dict[str, object]:
    """Parse coupon (decimal), tenor (years) and flags from an auction title."""
    t = str(title).strip()
    rest = t
    coupon = None
    m = _COUPON.match(t)
    if m:
        after = t[m.end():]
        # '%' marks a coupon; without it, a leading number followed by another number is a coupon
        # ('12.75 15-YEAR'), while one followed by '-YEAR' is the tenor ('5-YEAR TREASURY BOND').
        spaced = t[m.end(1):m.end()].strip() == "" and m.end() > m.end(1)
        if m.group(2) or (spaced and re.match(r"\d", after)):
            coupon = float(m.group(1)) / 100.0
            rest = after
    tm = _TENOR.search(rest)
    tenor = int(tm.group(1)) if tm else None
    return {
        "tenor_years": tenor,
        "coupon": coupon,
        "coupon_in_title": coupon is not None,
        "fixed_rate_label": bool(re.search(r"FIXED RATE", t, re.I)),
        "title_suffix_d": bool(re.search(r"\sD\s*$", t)),
    }


def read_auctions(file: Path | None = None) -> pd.DataFrame:
    """Parse the auction calendar into one row per auction."""
    raw = pd.read_excel(file or path("auctions"), header=None, dtype=object)
    hdr = raw.index[raw[1].astype(str).str.strip().eq("Auction Number")][0]
    df = raw.iloc[hdr + 1:, :4].copy()
    df.columns = ["sn", "auction_number", "title", "auction_date"]
    df = df[df["auction_number"].notna()]
    df["auction_number"] = pd.to_numeric(df["auction_number"]).astype(int)
    df["sn"] = pd.to_numeric(df["sn"]).astype(int)
    df["title"] = df["title"].astype(str).str.strip()
    df["auction_date"] = pd.to_datetime(df["auction_date"].astype(str).str.strip(), format="%d-%b-%Y")
    parsed = pd.DataFrame([parse_title(t) for t in df["title"]], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    df["evidence_class"] = "Observed"
    df["doc_id"] = "BOT-AUCTION-CALENDAR"
    return df.sort_values(["auction_date", "auction_number"]).reset_index(drop=True)


def last_auction_before(df: pd.DataFrame, tenor_years: int, date: str | pd.Timestamp) -> pd.Series:
    """The last auction of a given tenor on or before ``date``."""
    d = pd.Timestamp(date)
    sub = df[(df["tenor_years"] == tenor_years) & (df["auction_date"] <= d)]
    if sub.empty:
        raise LookupError(f"No {tenor_years}-year auction on or before {d.date()}")
    return sub.sort_values(["auction_date", "auction_number"]).iloc[-1]


def yields_needed(df: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """For each year, the longest-tenor auction held that year that is closest to 31 December.

    Blueprint §2: the annual COE uses the longest-tenor government bond yield each year. The second block lists, for
    each year, the last auction of every tenor >= 10 years (alternatives if the longest tenor's yield is unusable).
    """
    rows = []
    for y in range(first, last + 1):
        sub = df[(df["auction_date"].dt.year == y) & df["tenor_years"].notna()]
        if sub.empty:
            rows.append({"year": y, "selection": "longest_tenor_closest_to_year_end", "auction_number": None,
                         "note": "No auction with a parsed tenor in this year"})
            continue
        ye = pd.Timestamp(f"{y}-12-31")
        longest = int(sub["tenor_years"].max())
        pick = sub[sub["tenor_years"] == longest].sort_values("auction_date").iloc[-1]
        rows.append({"year": y, "selection": "longest_tenor_closest_to_year_end", **_row(pick, ye),
                     "note": f"Longest tenor auctioned in {y}: {longest} years"})
        for ten in sorted(t for t in sub["tenor_years"].unique() if t >= 10 and t != longest):
            alt = sub[sub["tenor_years"] == ten].sort_values("auction_date").iloc[-1]
            rows.append({"year": y, "selection": "alternative_long_tenor_last_in_year", **_row(alt, ye), "note": ""})
    out = pd.DataFrame(rows)
    out["yield_status"] = "TO COLLECT"
    out["yield_source_needed"] = "BoT auction result page (weighted average yield to maturity)"
    out["evidence_class"] = "Observed"
    return out


def _row(r: pd.Series, ye: pd.Timestamp) -> dict[str, object]:
    return {"auction_number": int(r["auction_number"]), "auction_date": r["auction_date"], "tenor_years":
            int(r["tenor_years"]), "coupon": r["coupon"], "title": r["title"],
            "days_before_year_end": int((ye - r["auction_date"]).days)}


def write_auctions(root: Path | None = None) -> list[Path]:
    """Write data/interim/bond_auctions.csv and reports/data_audit_auction_yields_needed.csv."""
    root = root or find_root()
    df = read_auctions(path("auctions", root))
    first, last = settings(root)["auction_years"]
    need = yields_needed(df, first, last)
    return [write_csv(df, path("interim", root) / "bond_auctions.csv"),
            write_csv(need, path("reports", root) / "data_audit_auction_yields_needed.csv")]
