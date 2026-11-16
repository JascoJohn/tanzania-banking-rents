"""Prompt 02d: deposit-interest reconciliation, unit and outlier checks, D34 long-bond yields, D35 lower bounds,
inclusion/NPS adjudication. Hand-checked values throughout."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.ingest.adjudication import apply, resolve
from tzbank.market.coe_inputs import alt_longest, alt_recent, auction_table, coe_inputs_annual, d34_primary
from tzbank.pit import eligible
from tzbank.pit.pubdates import resolve as resolve_pubdate
from tzbank.validate.plausibility import cost_of_deposits, deposit_checks, outlier_checks, unit_checks

# --- D34 long-bond yield -------------------------------------------------------------------------------------------

AUCTIONS = pd.DataFrame({
    "sn": [5, 4, 3, 2, 1], "auction_number": [686, 674, 688, 689, 693],
    "auction_date": pd.to_datetime(["2025-09-24", "2025-10-01", "2025-11-12", "2025-11-26", "2025-12-17"]),
    "tenor_years": [25, 10, 15, 5, 20], "coupon": [0.1375, 0.135, 0.1275, 0.1075, 0.13]})
RESULTS = pd.DataFrame({
    "auction_number": [686, 674, 688, 689, 693], "auction_date": AUCTIONS["auction_date"],
    "bond_title": [f"X BOND NUMBER {n} ISSUE {i}" for n, i in [(686, 17), (674, 3), (688, 9), (689, 1), (693, 4)]],
    "waytm": [0.13187, 0.13, 0.120794, 0.11, 0.120193], "weighted_average_price": 100.0, "status": "SUCCESS"})


def test_d34_primary_final_quarter_average():
    a = auction_table(AUCTIONS, RESULTS)
    row, used = d34_primary(a, pd.Timestamp("2025-12-31"))
    # tenor >= 15 in 1 Oct - 31 Dec 2025: bonds 688 (12.0794%) and 693 (12.0193%); the 25-year of 24 Sep is outside
    assert row["waytm"] == pytest.approx((0.120794 + 0.120193) / 2)
    assert row["n_auctions"] == 2 and row["bond_number"] == "688/693" and len(used) == 2
    assert row["method"] == "final-quarter average"
    # the old rule picks the stale 25-year auction (13.19%): the 117 bp gap the owner flagged
    old, _ = alt_longest(a, pd.Timestamp("2025-12-31"))
    assert old["bond_number"] == "686" and old["waytm"] == pytest.approx(0.13187)
    assert (old["waytm"] - 0.120193) * 1e4 == pytest.approx(116.77, abs=0.01)
    recent, _ = alt_recent(a, pd.Timestamp("2025-12-31"))
    assert recent["bond_number"] == "693"


def test_d34_fallback_and_identifiers():
    a = auction_table(AUCTIONS, RESULTS)
    row, used = d34_primary(a, pd.Timestamp("2025-10-31"))  # quarter 1 Aug - 31 Oct holds only the 25-year
    assert row["bond_number"] == "686" and row["method"] == "final-quarter average"
    row, used = d34_primary(a[a["bond_number"] != 686], pd.Timestamp("2025-10-31"))
    assert row["status"] == "no auction"
    late = a[a["bond_number"] == 686]
    row, _ = d34_primary(late, pd.Timestamp("2026-03-31"))  # nothing in Jan-Mar 2026: most recent before it
    assert row["method"].startswith("fallback") and row["bond_number"] == "686"
    assert a.set_index("bond_number").loc[686, "issue_number"] == 17
    assert a.set_index("bond_number").loc[693, "bond_auction_seq"] == 1


def test_reopened_bond_keeps_its_number_and_counts_auctions():
    au = pd.DataFrame({"sn": [3, 2, 1], "auction_number": [561, 561, 600],
                       "auction_date": pd.to_datetime(["2021-04-21", "2024-11-13", "2024-12-01"]),
                       "tenor_years": [25, 25, 15], "coupon": [0.1595, 0.1595, 0.12]})
    res = pd.DataFrame({"auction_number": [561, 561, 600], "auction_date": au["auction_date"],
                        "bond_title": ["B NUMBER 561 ISSUE 1"] * 2 + ["B NUMBER 600 ISSUE 2"],
                        "waytm": [0.163, 0.159, 0.15], "weighted_average_price": 100.0, "status": "SUCCESS"})
    a = auction_table(au, res)
    assert a[a["bond_number"] == 561]["bond_auction_seq"].tolist() == [1, 2]
    c, used = coe_inputs_annual(au, res, 2024, 2024)
    p = c[c["series"] == "d34_primary"].iloc[0]
    assert p["bond_number"] == "561/600" and p["waytm"] == pytest.approx((0.159 + 0.15) / 2)


def test_coe_outputs(root):
    p = path("processed", root)
    coe = pd.read_csv(p / "coe_inputs_annual.csv")
    assert "auction_number" not in coe.columns and {"bond_number", "issue_number"} <= set(coe.columns)
    assert coe["evidence_class"].notna().all()
    assert coe["note"].dropna().nunique() > 5  # a reason per row, not one boilerplate note
    prim = coe[coe["series"] == "d34_primary"].set_index("date_label")
    assert prim.loc["FY2025", "waytm"] == pytest.approx((0.120794 + 0.120193) / 2, abs=1e-6)
    assert prim.loc["valuation_historical", "waytm"] == prim.loc["FY2025", "waytm"]
    assert {"valuation_current", "valuation_historical"} <= set(prim.index)
    used = pd.read_csv(p / "coe_long_bond_auctions_used.csv")
    d34 = used[(used["series"] == "d34_primary") & (used["date_label"] != "FY2012")]  # FY2012: flagged fallback (02f)
    assert (d34["tenor_years"] >= 15).all()


# --- deposit interest, unit and outlier checks -----------------------------------------------------------------------

def _wide(rows):
    return pd.DataFrame(rows)


def test_deposit_checks_flag_partial_capture():
    # CRDB-like series: deposit share ~0.99 except FY2013 (0.27); cost of deposits drops 2.2% -> 0.7%
    rows = [{"bank_id": "B", "fiscal_year": y, "interest_expense": ie, "interest_expense_deposits": d,
             "customer_deposits": dep}
            for y, ie, d, dep in [(2011, 35.0, 34.8, 1600.0), (2012, 55.5, 55.25, 2300.0), (2013, 69.3, 18.6, 2600.0),
                                  (2014, 95.5, 88.1, 3000.0)]]
    flags = deposit_checks(_wide(rows))
    got = {(f["check"], f["fiscal_year"]) for f in flags}
    assert ("deposit share of interest expense vs own median", 2013) in got
    assert ("cost of deposits changes more than 2x y/y", 2013) in got  # 0.76% vs 2.84%: ratio 0.27
    assert ("cost of deposits changes more than 2x y/y", 2014) in got  # back up: ratio 4.0
    cod = cost_of_deposits(_wide(rows))
    assert cod.iloc[1] == pytest.approx(55.25 / 1950.0)


def test_unit_checks():
    rows = [{"bank_id": "M", "fiscal_year": 2017, "net_profit": 314_738_000.0, "total_assets": 21.37e9,
             "total_equity": 3.98e9, "interest_income": 2.99e9},
            {"bank_id": "M", "fiscal_year": 2018, "net_profit": 930_000.0, "total_assets": 15.92e9,
             "total_equity": -532e6, "interest_income": 3.34e9},
            {"bank_id": "M", "fiscal_year": 2019, "net_profit": 409e6, "total_assets": 1.678e9,  # unit slip x10
             "total_equity": 3.0e9, "interest_income": 3.30e9}]
    got = {(f["check"], f["fiscal_year"]) for f in unit_checks(_wide(rows))}
    assert ("total equity <= 0", 2018) in got
    assert ("net_profit changes more than 100x y/y", 2018) in got  # 930k vs 314.7m: ratio 0.003
    assert ("net_profit changes more than 100x y/y", 2019) in got
    assert ("interest income / total assets above 30%", 2019) in got  # 3.30 / 1.678 = 197%
    assert ("net profit / total assets outside +/-15%", 2019) in got  # 409m / 1.678bn = 24%
    assert ("net profit / total assets outside +/-15%", 2017) not in got


def test_outlier_checks():
    w = _wide([{"bank_id": "X", "fiscal_year": 2020, "total_equity": 1.0, "interest_expense_deposits": 1.0,
                "customer_deposits": 5.0},
               {"bank_id": "X", "fiscal_year": 2021, "total_equity": -1.0, "interest_expense_deposits": 1.0,
                "customer_deposits": 5.0}])
    panel = pd.DataFrame({"bank_id": ["X", "X"], "fiscal_year": [2020, 2021], "roe": [0.6, 0.1],
                          "nii_to_avg_assets": [0.05, 0.2], "cost_to_income": [2.5, 0.5]})
    got = {(f["check"], f["fiscal_year"]) for f in outlier_checks(w, panel)}
    assert got == {("|ROE| > 50%", 2020), ("cost-to-income > 2", 2020), ("NII / average assets > 15%", 2021),
                   ("equity <= 0", 2021), ("cost of deposits outside 0-15%", 2021)}  # 1 / 5 = 20%


def test_outlier_list_complete_and_verified(root):
    o = pd.read_csv(path("validation", root) / "outliers.csv")
    assert o["evidence_class"].notna().all()
    assert {"known case: MKCB FY2019 loss", "known case: MKCB FY2021 loss",
            "known case: MBP FY2013 cost-to-income 1.86"} <= set(o["check"])
    mcb = o[(o["bank_id"] == "B06") & (o["check"] == "cost-to-income > 2")]
    assert set(mcb["fiscal_year"]) == {2017, 2018, 2019, 2020} and (mcb["status"] == "verified").all()
    real = o[~o["check"].str.startswith("known case")]
    assert real["inputs"].str.contains("<").all()  # every input carries its source reference
    acc = pd.read_csv(path("processed", root) / "bank_year_flags.csv")
    assert not acc.set_index(["bank_id", "fiscal_year"]).loc[("B04", 2013), "accepted_for_analysis"]


# --- corrected values in the processed panel -----------------------------------------------------------------------

def test_crdb_deposit_interest_corrected(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv").set_index(["bank_id", "fiscal_year"])
    # CRDB AR FY2013 p.140 / FY2014 p.135 (TZS m): current + savings + fixed; + inter-bank + subordinated = total
    assert w.loc[("B01", 2013), "interest_expense_deposits"] == 3_036e6 + 15_592e6 + 47_464e6
    assert w.loc[("B01", 2014), "interest_expense_deposits"] == 8_519e6 + 19_907e6 + 59_629e6
    assert 66_092 + 2_918 + 267 == 69_277 and 88_055 + 6_124 + 1_333 == 95_512
    # NMB FY2021 (2021/2022 reports, amounts printed one row below their labels)
    assert w.loc[("B02", 2021), "interest_expense_deposits"] == (62_486 + 27_698 + 9_413) * 1e6
    share = (w["interest_expense_deposits"] / w["interest_expense"]).loc["B01"].dropna()
    assert (share.loc[2009:2021] > 0.6).all()


def test_deposit_reconciliation_report(root):
    d = pd.read_csv(path("validation", root) / "deposit_interest_reconciliation.csv")
    dep = d[d["variable"].str.contains("Deposits|Accounts")]
    assert dep["status"].isin(["reconciled", "reconciled via another report", "reconciled (layout verified)",
                               "corrected"]).all()  # nothing accepted by the weak check alone
    c13 = d[(d["row_id"] == "HF-1154")].iloc[0]
    assert c13["status"] == "corrected" and c13["reconciled_value"] == 66_092e6


def test_weak_gate_never_admits_focus_deposit_interest():
    from tzbank.extract.ledger import select

    c = pd.DataFrame([{"bank_id": "B01", "fiscal_year": 2013, "variable": "Interest Expense on Customer Deposits",
                       "value": 18_628e6, "file": "CRDB/2013.pdf", "page": 140, "label": "sum", "column": "current",
                       "gate": "deposit interest <= total interest expense", "gate_passed": True,
                       "confidence": "HIGH"}])
    ledger, review = select(c)
    assert ledger.empty or not ((ledger["bank_id"] == "B01") & (ledger["fiscal_year"] == 2013)).any()
    assert len(review) == 1


def test_crdb_fy2009_2010_equity_and_opex(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv").set_index(["bank_id", "fiscal_year"])
    assert w.loc[("B01", 2010), "total_equity"] == 240_921_695e3  # FY2010 AR p.33
    assert w.loc[("B01", 2009), "total_equity"] == 207_774_139e3
    assert 2_305_224_620 - 2_064_302_925 == 240_921_695
    assert w.loc[("B01", 2010), "operating_expenses"] == (63_360_635 + 46_180_800) * 1e3  # FY2010 AR p.32
    long = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False)
    r = long[(long["bank_id"] == "B01") & (long["fiscal_year"] == 2010) & (long["canonical_name"] == "total_equity")]
    assert set(r["value"]) == {240_921_695e3, 233_511e6} and r["is_restated"].sum() == 1  # D21
    p = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv").set_index(["bank_id", "fiscal_year"])
    assert pd.notna(p.loc[("B01", 2010), "roe"])  # CRDB ROE now starts in FY2010


# --- D35 lower bounds ----------------------------------------------------------------------------------------------

def test_lower_bound_dates_ineligible():
    row = pd.Series({"Document Type": "Statistical report", "Lower-Bound Publication Date": "2024-01-30",
                     "Estimate Basis": "PDF modification date"})
    r = resolve_pubdate(row)
    assert r["publication_date_type"] == "lower_bound" and r["publication_date"] == pd.Timestamp("2024-01-30")
    df = pd.DataFrame({"publication_date": pd.to_datetime(["2024-01-30", "2024-01-30"]),
                       "publication_date_type": ["lower_bound", "actual"], "period_end": pd.NaT})
    with pytest.warns(UserWarning):
        out = eligible(df, "2025-12-31")
    assert out["publication_date_type"].tolist() == ["actual"]


def test_nps_sources_typed_lower_bound(root):
    s = pd.read_csv(path("interim", root) / "inputs_sources.csv")
    nps = s[s["Doc ID"].astype(str).str.startswith("BOT-NPSAR-")]
    assert len(nps) == 4 and (nps["publication_date_type"] == "lower_bound").all()
    fir = s[s["Doc ID"].astype(str).str.startswith("BOT-FIR-")]
    assert (fir["publication_date_type"] == "estimated").all()  # file-name IDs stay as posting evidence
    panel = pd.read_csv(path("processed", root) / "nps_panel.csv")
    assert (panel["publication_date_type"].dropna() == "lower_bound").all()


# --- adjudication ----------------------------------------------------------------------------------------------------

def test_adjudication_codes_and_panel_untouched():
    assert resolve("", 89.0, 72.0) == (89.0, "open: panel value used until the owner decides")
    assert resolve("report", 89.0, 72.0)[0] == 72.0
    assert resolve("panel", 89.0, 72.0)[0] == 89.0
    assert pd.isna(resolve("exclude", 89.0, 72.0)[0])
    assert resolve("75", 89.0, 72.0)[0] == 75.0
    with pytest.raises(ValueError):
        resolve("maybe", 1.0, 2.0)
    panel = pd.DataFrame({"series_id": ["S", "S"], "year": [2024, 2025], "value": [10.0, 20.0]})
    table = pd.DataFrame([{"panel": "nps", "series": "S", "year": 2025, "adjudicated_value": 19.0,
                           "item_id": "ADJ-X", "status": "decided: report value"}])
    out = apply(panel, "nps", table)
    assert out["value"].tolist() == [10.0, 20.0] and out["adjudicated_value"].tolist() == [10.0, 19.0]


def test_adjudication_report(root):
    t = pd.read_csv(path("reports", root) / "inclusion_nps_adjudication.csv")
    assert len(t) == 4 and "owner_decision" in t.columns
    eft = t[t["item_id"] == "ADJ-04"].iloc[0]
    assert eft["panel_value_used"] == 21_075_755 and eft["adjudicated_value"] == 20_595_123
    nps = pd.read_csv(path("processed", root) / "nps_panel.csv")
    row = nps[(nps["series_id"] == "03_TACH|Table C1|Volume of transactions") & (nps["year"] == 2024)].iloc[0]
    assert row["value"] == 21_075_755 and row["adjudicated_value"] == 20_595_123  # both vintages visible
    decided = t[t["item_id"].isin(["ADJ-01", "ADJ-02", "ADJ-03"])]  # filled by the owner in prompt 02e (D37)
    assert decided["owner_decision"].notna().all() and decided["status"].str.startswith("decided").all()
