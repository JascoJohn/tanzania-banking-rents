"""Prompt 02e: D36 date rule, D37 adjudications, D38 filled checklists, D39 supplements, new-report rows."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.ingest.manual_checklist import CANON, comparative_year, gate, page_number, unit_scale
from tzbank.pit.pubdates import ASSUMED, REG_2001_A, REG_2001_Q, REG_2014_A, annual_rule, quarterly_rule, regime

# --- D36: three-period publication-date rule --------------------------------------------------------------------------


def test_gn368_three_periods():
    assert regime(pd.Timestamp("2008-09-30")) == "2001"
    assert regime(pd.Timestamp("2008-12-31")) == "gn368"
    assert regime(pd.Timestamp("2014-06-30")) == "gn368"
    assert regime(pd.Timestamp("2014-09-30")) == "2014"
    d, r = quarterly_rule(pd.Timestamp("2008-09-30"))
    assert d == pd.Timestamp("2008-11-14") and r.regulatory_basis == REG_2001_Q and r.flag == ""
    d, r = quarterly_rule(pd.Timestamp("2010-06-30"))
    assert d == pd.Timestamp("2010-08-14") and r.flag == ASSUMED  # 45 days assumed under GN 368
    d, r = quarterly_rule(pd.Timestamp("2014-09-30"))
    assert d == pd.Timestamp("2014-11-14") and "2014 quarter" in r.flag  # D27 unchanged
    assert annual_rule(pd.Timestamp("2007-12-31"), None)[1].regulatory_basis == REG_2001_A
    _, r = annual_rule(pd.Timestamp("2013-12-31"), pd.Timestamp("2014-03-20"))
    assert r.flag == ASSUMED and r.regulatory_basis.startswith(ASSUMED)
    assert annual_rule(pd.Timestamp("2014-12-31"), None)[1].regulatory_basis == REG_2014_A


def test_gn368_rule_changes_no_eligibility(root):
    s = pd.read_csv(path("processed", root) / "sources.csv")
    flagged = s[s["pubdate_flag"].astype(str) == ASSUMED]
    assert len(flagged) >= 10
    assert (pd.to_datetime(flagged["publication_date"]) < pd.Timestamp("2015-01-01")).all()  # before both cut-offs
    assert "owner_instruction_unverified" not in " ".join(s["regulatory_basis"].astype(str))
    assert (s["Doc ID"] == "REG-GAZ-2008-50").any()


# --- D37 adjudications ---------------------------------------------------------------------------------------------


def test_d37_adjudications_applied(root):
    t = pd.read_csv(path("reports", root) / "inclusion_nps_adjudication.csv").set_index("item_id")
    assert t.loc["ADJ-01", "adjudicated_value"] == 21.76
    assert t.loc["ADJ-02", "adjudicated_value"] == 72
    assert t.loc["ADJ-03", "adjudicated_value"] == 45_785
    assert t.loc["ADJ-04", "adjudicated_value"] == 20_595_123
    assert t["status"].str.startswith("decided").all()


# --- D38 filled checklists -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("unit, scale", [("TZS'000", 1e3), ("TZS' Millions", 1e6), ("TZS' Million", 1e6),
                                         ("TZS million", 1e6), ("TZS", 1.0), ("TZS bn?", None)])
def test_unit_normalisation(unit, scale):
    assert unit_scale(unit) == scale


def test_page_text_parsing():
    assert page_number("68 (printed p.66)") == 68
    assert page_number("PDF p.1") == 1
    assert page_number("Note 10(a)") is None
    assert comparative_year("50 (from 2013 report comparative)") == 2013
    assert comparative_year("150 (from 2016 comparative)") == 2016
    assert comparative_year("150 (Note 28)") is None


def _conv(rows):
    base = {"value": 1.0, "file": "X/2020.pdf", "page": 1, "value_found_on_page": True}
    return pd.DataFrame([base | r for r in rows])


def test_checklist_gates():
    wide = pd.DataFrame([{"bank_id": "B02", "fiscal_year": 2013, "customer_deposits": 2_577.946e9,
                          "interest_income": 100.0, "interest_expense": 30.0, "net_interest_income": 70.0,
                          "tier1_ratio": 0.209, "net_loans": 90.0},
                         {"bank_id": "B02", "fiscal_year": 2012, "deposits_time": 87.651e9}])
    rows = [{"bank_id": "B02", "fiscal_year": 2013, "variable": v, "value_full": x}
            for v, x in [("Customer Deposits: Current Accounts", 1_313.982e9),
                         ("Customer Deposits: Savings", 1_191.448e9), ("Customer Deposits: Time", 72.516e9)]]
    g = gate(_conv(rows), wide)
    assert g["accepted"].all() and g["gate_results"].str.contains("deposit mix").all()
    bad = rows[:2] + [{"bank_id": "B02", "fiscal_year": 2013, "variable": "Customer Deposits: Time",
                       "value_full": 172.516e9}]
    g = gate(_conv(bad), wide)
    assert not g["accepted"].any() and g["reason"].str.contains("deposit mix").all()
    # Tier 1 / RWA vs reported ratio; failing ratio rejects both rows
    g = gate(_conv([{"bank_id": "B02", "fiscal_year": 2013, "variable": "Tier 1 Capital", "value_full": 199.65e9},
                    {"bank_id": "B02", "fiscal_year": 2013, "variable": "Total RWA", "value_full": 954.04e9}]), wide)
    assert g["accepted"].all()
    g = gate(_conv([{"bank_id": "B02", "fiscal_year": 2013, "variable": "Tier 1 Capital", "value_full": 150e9},
                    {"bank_id": "B02", "fiscal_year": 2013, "variable": "Total RWA", "value_full": 954.04e9}]), wide)
    assert not g["accepted"].any()
    # deposit interest above the total interest expense
    g = gate(_conv([{"bank_id": "B05", "fiscal_year": 2013, "variable": "Interest Expense on Customer Deposits",
                     "value_full": 40.0}]), wide.assign(bank_id="B05"))
    assert not g["accepted"].iloc[0]
    # year-on-year jump flagged, not rejected (time deposits 72.5bn vs 87.7bn: no flag; 4x: flag)
    g = gate(_conv([{"bank_id": "B02", "fiscal_year": 2013, "variable": "Customer Deposits: Time",
                     "value_full": 360e9}]), wide.drop(columns="customer_deposits"))
    assert g["accepted"].iloc[0] and "flagged" in g["yoy_flag"].iloc[0]
    assert set(CANON) >= {"Customer Deposits: Time", "Tier 1 Capital", "Total RWA"}


def test_checklist_rows_written(root):
    rows = pd.read_csv(path("interim", root) / "manual_checklist_rows.csv")
    assert len(rows) == 62 and rows["accepted"].all()
    assert (rows["value_full"] >= 1e6).all()  # units normalised to full TZS
    r = rows[(rows["bank_id"] == "B02") & (rows["fiscal_year"] == 2012)].iloc[0]
    assert r["doc_id"] == "NMB-AR-FY2013" and r["confidence"] == "MEDIUM"  # 'from 2013 report comparative'
    r = rows[(rows["bank_id"] == "B07") & (rows["variable"] == "Interest Expense on Customer Deposits")].iloc[0]
    assert r["value_is_sum_of_printed_lines"] == "265,380 + 340,268" and int(r["page"]) == 23
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv").set_index(["bank_id", "fiscal_year"])
    assert w.loc[("B02", 2013), "deposits_current"] == 1_313_982e6
    ml = pd.read_csv(path("reports", root) / "manual_check_list.csv")
    filled = set(zip(rows["bank_id"], rows["variable"], rows["fiscal_year"], strict=True))
    assert not filled & set(zip(ml["bank_id"], ml["variable"], ml["fiscal_year"], strict=True))


# --- new-report rows ----------------------------------------------------------------------------------------------


def test_crdb_fy2025_full_report_rows(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv").set_index(["bank_id", "fiscal_year"])
    r = w.loc[("B01", 2025)]
    assert r["interest_expense_deposits"] == (46_130 + 40_446 + 196_032) * 1e6  # Note 13, p.287
    assert r["interest_expense_borrowings"] == (116_714 + 52_134 + 16_357) * 1e6
    assert r["deposits_current"] + r["deposits_savings"] + r["deposits_time"] == r["customer_deposits"]
    # Note 13 reconciles: conventional lines + lease + Islamic distribution = audited total
    assert 577_455 + 5_819 + 7_898 == r["interest_expense"] / 1e6
    long = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False)
    t = long[(long["bank_id"] == "B01") & (long["fiscal_year"] == 2024)
             & (long["canonical_name"] == "interest_expense_deposits_time")]
    assert {126_488e6, 125_457e6} <= set(t["value"]) and t["is_restated"].sum() >= 1  # D21 restatement row
    chk = pd.read_csv(path("validation", root) / "crdb_fy2025_fullreport_check.csv")
    assert (chk["status"] == "not printed as such on the cited page").sum() <= 5  # listed for the owner


def test_crdb_fy2010_cost_to_income_and_mkcb_fy2021(root):
    p = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv").set_index(["bank_id", "fiscal_year"])
    assert p.loc[("B01", 2010), "cost_to_income"] == pytest.approx(109_541_435 / (125_005_122 + 70_530_048))
    long = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False)
    npf = long[(long["bank_id"] == "B05") & (long["fiscal_year"] == 2021) & (long["canonical_name"] == "net_profit")]
    assert set(npf["value"]) >= {1_148e6, -1_859_510e3}
    assert npf.loc[npf["value"] == -1_859_510e3, "is_restated"].all()


# --- D39 supplements ---------------------------------------------------------------------------------------------


def test_supplement_gates(root):
    d = pd.read_csv(path("reports", root) / "inclusion_nps_code_extraction_review.csv")
    assert "owner_check" in d.columns and d["evidence_class"].notna().all()
    assert not d.loc[d["gate_result"].str.startswith(("FAIL", "NOT")), "used"].any()  # failures never used
    t1 = d[(d["report"] == 2022) & (d["table"] == "Table 1")]
    assert len(t1) == 48 and t1["gate_result"].eq("PASS").all()
    atm = t1[t1["row_label"].str.contains("ATM") & (t1["column"] == "Dec-22")]
    assert atm["value"].iloc[0] == 1_885
    t32 = d[(d["report"] == 2025) & (d["table"] == "Table 3.2") & (d["column"] == "2024/25 number")]
    parts = t32[t32["row_label"] != "Total Loans"]["value"].sum()
    assert parts == t32[t32["row_label"] == "Total Loans"]["value"].iloc[0] == 16_256
    h12 = d[(d["table"] == "Table H12") & d["used"]]
    assert h12.groupby(["series", "row_label", "column"]).size().max() == 1  # one used vintage per series-year
    unl = d[d["gate_result"].str.contains("D33")]
    assert len(unl) == 9 and not unl["used"].any()
    psp = d[(d["report"] == 2025) & (d["table"] == "Annex O")]
    assert len(psp) >= 70 and psp["used"].sum() >= 70


def test_supplements_merged_below_owner(root):
    nps = pd.read_csv(path("processed", root) / "nps_panel.csv", low_memory=False)
    code = nps[nps["source_precedence"].astype(str).str.startswith("code-extracted")]
    owner = nps[nps["source_precedence"] == "owner panel"]
    assert len(code) > 0 and (code["extraction_method"] == "automated").all()
    assert not set(code["canonical_name"]) & set(owner["canonical_name"])  # never replaces an owner series
    cw = pd.read_csv(path("crosswalk", root), dtype=str)
    assert set(code["canonical_name"]) <= set(cw["canonical_name"])
