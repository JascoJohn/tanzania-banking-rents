"""BoT Treasury bond auction results (prompt 02 section 3; blueprint section 2, historical cost of equity).

``fetch`` (network, run manually, not part of ``run all``) saves each auction's result exactly as served by the
public endpoint the BoT TBonds page itself calls (POST /TBonds/AuctionSummaries) to
data/raw/bot/auctions/<auction>_<days>.json, plus the list page. ``parse`` (offline, in ``run all``) turns the saved
files into data/interim/bond_auction_results.csv. Requests are paced and identify the project; nothing bypasses
any access control.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

BASE = "https://www.bot.go.tz/"
UA = "Mozilla/5.0 (compatible; tanzania-banking-rents research; private academic use)"
FIELDS = {
    "REDEMPTION DATE": "redemption_date",
    "WEIGHTED AVERAGE PRICE (WAP) FOR SUCCESSFUL BIDS": "weighted_average_price",
    "WEIGHTED AVERAGE YIELD-TO-MATURITY": "waytm_pct",
    "WEIGHTED AVERAGE COUPON YIELD": "coupon_yield_pct",
    "AMOUNT OFFERED TZS(000,000)": "amount_offered_tzs_mn",
    "TOTAL TENDERED TZS(000,000)": "amount_tendered_tzs_mn",
    "SUCCESSFUL BIDS TZS(000,000)": "amount_successful_tzs_mn",
    "NUMBER OF BIDS RECEIVED": "bids_received",
    "NUMBER OF SUCCESSFUL BIDS": "bids_successful",
    "MINIMUM SUCCESSFUL PRICE / 100": "minimum_successful_price",
}


def raw_dir(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / "bot" / "auctions"


def _get(url: str, data: dict[str, str] | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https host)
        return r.read()


def fetch(root: Path | None = None, pause: float = 1.0, refetch: bool = False) -> list[Path]:
    """Download the TBonds list page and every auction result not yet saved. Returns files written."""
    d = raw_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    listing = _get(BASE + "TBonds")
    (d / "TBonds_list.html").write_bytes(listing)
    values = list(dict.fromkeys(re.findall(rb'id="showSummaryDetails" value="(\d+_\d+)"', listing)))
    out = [d / "TBonds_list.html"]
    for v in values:
        f = d / f"{v.decode()}.json"
        if f.exists() and not refetch:
            continue
        no, days = v.decode().split("_")
        resp = _get(BASE + "TBonds/AuctionSummaries", {"au_no": no, "au_days": days})
        json.loads(resp)  # must be JSON; raises otherwise
        f.write_bytes(resp)
        out.append(f)
        time.sleep(pause)
    return out


def _num(s: object) -> float | None:
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def parse(root: Path | None = None) -> pd.DataFrame:
    """One row per saved auction result (competitive column)."""
    rows = []
    for f in sorted(raw_dir(root).glob("*.json")):
        j = json.loads(f.read_bytes())
        if j.get("message") != "SUCCESS":
            # BoT returns FAILURE with an empty summary: no result published. The response does not say why
            # (cancelled, rejected or not recorded); the auction is kept for traceability and has no yield.
            rows.append({"auction_number": int(f.stem.split("_")[0]), "au_days_key": int(f.stem.split("_")[1]),
                         "bond_title": j.get("bondTitle"),
                         "auction_date": pd.to_datetime(j.get("BondDate"), format="%B %d, %Y", errors="coerce"),
                         "isin": j.get("ISIN"), "file": f"bot/auctions/{f.name}", "status": j.get("message"),
                         "status_note": "BoT endpoint returned FAILURE with no summary: no result published "
                                        "(reason not stated; excluded from yield series)"})
            continue
        items = {i["itemDesc"].strip(): i for i in j.get("tbondSummary", [])}
        no = j["tbondSummary"][0]["auNo"] if j.get("tbondSummary") else int(f.stem.split("_")[0])
        r = {"auction_number": int(no), "au_days_key": int(f.stem.split("_")[1]), "bond_title": j.get("bondTitle"),
             "auction_date": pd.to_datetime(j.get("BondDate"), format="%B %d, %Y", errors="coerce"),
             "isin": j.get("ISIN"), "file": f"bot/auctions/{f.name}", "status": "SUCCESS"}
        for desc, col in FIELDS.items():
            v = items.get(desc, {}).get("itemValue")
            r[col] = pd.to_datetime(v, format="%d/%m/%Y", errors="coerce") if col == "redemption_date" else _num(v)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["waytm"] = df["waytm_pct"] / 100
    df["coupon_yield"] = df["coupon_yield_pct"] / 100
    df["evidence_class"] = "Observed"
    df["doc_id"] = "BOT-TBOND-RESULTS"
    return df.sort_values(["auction_date", "auction_number"]).reset_index(drop=True)


def write_results(root: Path | None = None) -> list[Path]:
    """data/interim/bond_auction_results.csv (empty frame with a note if nothing is saved yet)."""
    root = root or find_root()
    df = parse(root)
    return [write_csv(df, path("interim", root) / "bond_auction_results.csv")]


if __name__ == "__main__":
    for p in fetch():
        print(p)
