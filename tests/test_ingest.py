"""Ingest: crosswalk completeness, auction parsing, Phase 1 read-only reading, banks.yaml (prompt 01 section 12)."""

import pandas as pd
import pytest

from tzbank.clean.crosswalk import CROSSWALK_COLUMNS, load_crosswalk, unmapped
from tzbank.config import banks
from tzbank.ingest.auctions import last_auction_before, parse_title, read_auctions
from tzbank.ingest.inputs_workbook import SHEETS, load_interim
from tzbank.ingest.phase1 import read_bank_universe, read_phase1


@pytest.fixture(scope="module")
def auctions():
    return read_auctions()


def test_crosswalk_complete(root):
    cw = load_crosswalk(root)
    assert list(cw.columns[: len(CROSSWALK_COLUMNS)]) == CROSSWALK_COLUMNS
    p1 = read_phase1()
    sheets = {s: load_interim(s, root) for s in SHEETS}
    assert unmapped(cw, p1, sheets).empty


def test_crosswalk_rows_are_mapped_or_explained(root):
    cw = load_crosswalk(root)
    assert set(cw["status"]) <= {"mapped", "excluded", "needed_not_held"}
    assert (cw.loc[cw["status"] == "mapped", "canonical_name"] != "").all()
    assert (cw.loc[cw["status"] != "mapped", "exclusion_reason"] != "").all()
    assert set(cw.loc[cw["status"] == "mapped", "evidence_class"]) <= {"Observed", "Derived"}


def test_every_auction_has_tenor(auctions):
    assert len(auctions) == 377
    assert auctions["tenor_years"].notna().all()
    assert set(auctions["tenor_years"]) == {2, 5, 7, 10, 15, 20, 25}


def test_coupon_extracted_for_every_title_that_states_one(auctions):
    """Three titles state no coupon at all; every other title yields a coupon. No coupon is imputed."""
    no_coupon = auctions[auctions["coupon"].isna()]
    assert sorted(no_coupon["auction_number"]) == [327, 355, 393]
    assert not no_coupon["title"].str.contains("%").any()
    assert auctions["coupon"].between(0.05, 0.20).sum() == len(auctions) - 3


@pytest.mark.parametrize("title,tenor,coupon", [
    ("13.00% 20-YEAR TREASURY BOND D", 20, 0.13), ("12.75 15-YEAR TREASURY BOND", 15, 0.1275),
    ("15.49%-20 YEAR TREASURY BOND", 20, 0.1549), ("10.08% 7-TREASURY BOND", 7, 0.1008),
    ("10.08%7-YEAR TREASURY BOND", 7, 0.1008), ("9.18% 5- YEAR TREASURY BOND", 5, 0.0918),
    ("5-YEAR TREASURY BOND", 5, None), ("15-YEAR FIXED RATE TREASURY BOND", 15, None),
])
def test_parse_title(title, tenor, coupon):
    p = parse_title(title)
    assert p["tenor_years"] == tenor
    assert (p["coupon"] is None) if coupon is None else abs(p["coupon"] - coupon) < 1e-12


def test_last_20y_before_historical_date_is_693(auctions):
    r = last_auction_before(auctions, 20, "2025-12-31")
    assert r["auction_number"] == 693
    assert r["auction_date"] == pd.Timestamp("2025-12-17")


def test_last_15y_before_historical_date_is_688(auctions):
    assert last_auction_before(auctions, 15, "2025-12-31")["auction_number"] == 688


def test_banks_yaml_matches_phase1_universe():
    u = read_bank_universe()
    cfg = {b["bank_id"]: b for b in banks()}
    assert sorted(cfg) == [f"B0{i}" for i in range(1, 8)]
    for _, r in u.iterrows():
        b = cfg[r["Bank ID"]]
        assert (b["name"], b["ticker"], b["listing_year"]) == (r["Bank Name"], r["Ticker"], int(r["Listing Year"]))


def test_phase1_long_has_all_raw_sheets():
    p1 = read_phase1()
    assert {"RAW_CAPITAL", "RAW_CREDIT_RISK", "RAW_LIQUIDITY", "RAW_PROFITABILITY", "RAW_BALANCE_SHEET", "RAW_MARKET",
            "RAW_OWNERSHIP", "RAW_GOVERNANCE", "RAW_DISCLOSURE", "RAW_DATA"} <= set(p1["sheet"])
    assert set(p1["bank_id"]) == {f"B0{i}" for i in range(1, 8)}


def test_publication_dates_recomputed_not_cached(root):
    """Every Historical Financials row with a Doc ID gets the D16-resolved Sources date for that Doc ID."""
    hf = load_interim("Historical Financials", root)
    src = load_interim("Sources", root).set_index("Doc ID")["publication_date"]  # D16, recomputed in Python
    with_doc = hf[hf["Doc ID"].notna()]
    assert (with_doc["publication_date"] == with_doc["Doc ID"].map(src)).all()
    assert "Eligible: Historical" not in hf.columns  # the workbook's formula column is not read
