"""Prompt 02f: D40 merge, BoT report extraction gates, processed BoT tables, COE back to 2008, PDF-stage cache."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.extract.bot_reports import gates
from tzbank.ingest.bot import TEMPLATE_COLUMNS
from tzbank.ingest.bot_database import freq_class, merge, norm_label, topic_of, unit_of
from tzbank.market.coe_inputs import short_tenor_fallback
from tzbank.reporting.pdf_stage import fingerprint

# --- D40 merge ------------------------------------------------------------------------------------------------------


def test_label_normalisation_matches_v1_and_v2():
    assert norm_label("dom Treasury bills rate | 91 days") == norm_label("Domestic currency | Treasury bills rates | "
                                                                         "91 days")
    assert norm_label("dom overall lending rate | short-term (up to 1 year)") == norm_label(
        "Domestic currency | Overall lending rate | Short-term (up to 1year)")
    assert norm_label("dom treasury bonds rate | 10 years") == norm_label("Domestic currency | Treasury bonds rates | "
                                                                          "10-years")
    assert topic_of("Weighted average interest rate structure (A3.13)")[0] == topic_of(
        "Interest Rate Structure, Weighted Averages")[0] == "interest_rates"
    assert topic_of("Treasury bills auction - actual sales by investor category (A2.2)")[0] != topic_of(
        "Treasury Bills Outstanding by Investor Category")[0]
    assert freq_class("Quarter-end") == "quarterly" and freq_class("End-June stock/ratio") == "fy_june"
    assert unit_of("% per annum") == ("ratio", 0.01) and unit_of("Millions of TZS") == ("TZS million", 1.0)


def _rows(wb, vals):
    return pd.DataFrame([{"workbook": wb, "topic": "interest_rates", "category": "rates", "label": lab,
                          "source_label": lab, "period_end": pd.Timestamp(d), "frequency": "Monthly", "value": v,
                          "unit": "Percent", "source_report": wb, "file": "f", "table": "A3.13", "page": 1, "flag": "",
                          "group": "g"} for lab, d, v in vals])


def test_v2_supersedes_v1_and_conflicts_listed():
    v1 = _rows("v1", [("dom savings deposit rate", "2014-12-31", 3.1), ("dom savings deposit rate", "2015-06-30", 3.2),
                      ("dom savings deposit rate", "2010-12-31", 2.5)])
    v2 = _rows("v2", [("Domestic currency | Savings deposit rate", "2014-12-31", 3.1),
                      ("Domestic currency | Savings deposit rate", "2015-06-30", 3.4)])
    merged, diff = merge(v1, v2)
    assert len(merged) == 3  # two shared keys from v2, one v1-only row kept
    assert set(merged.loc[merged["precedence"] == "v1 only", "value"]) == {2.5}
    assert merged.loc[merged["period_end"] == "2015-06-30", "value"].iloc[0] == 3.4
    assert diff["status"].str.startswith("CONFLICT").sum() == 1 and (diff["status"] == "same value").sum() == 1


def test_merged_database_outputs(root):
    d = pd.read_csv(path("validation", root) / "bot_v1_v2_diff.csv")
    assert len(d) > 500 and d["evidence_class"].notna().all()
    assert d["status"].str.startswith("CONFLICT").sum() < 0.15 * len(d)
    db = pd.read_csv(path("interim", root) / "bot_research_database.csv", low_memory=False)
    assert list(db.columns) == [*TEMPLATE_COLUMNS, "evidence_class"]
    assert not db.duplicated(["series_id", "period_end"]).any()
    assert (db.loc[db["unit"] == "ratio", "value"].abs() <= 1.5).mean() > 0.99  # percentages became ratios
    long = pd.read_csv(path("interim", root) / "bot_long.csv", low_memory=False)
    assert long["file"].astype(str).str.contains("D40").sum() == len(db)


# --- BoT report extraction gates (D39) ---------------------------------------------------------------------------


def test_licensed_banks_gates():
    recs = [("Commercial banks", ["35", "13", "22", "6", "29"]), ("Microfinance banks", ["3", "0", "3", "0", "3"]),
            ("Community banks", ["2", "2", "0", "1", "1"]), ("Development finance banks", ["2", "2", "0", "0", "2"]),
            ("Total", ["42", "17", "25", "7", "35"])]
    cols = ["Number", "Domestic", "Foreign", "Listed", "Not listed"]
    g = gates("Licensed Banks", cols, recs)
    assert all(v.startswith("PASS") for v in g.values())
    bad = [*recs[:4], ("Total", ["43", "17", "25", "7", "35"])]
    assert any(v.startswith("FAIL") for v in gates("Licensed Banks", cols, bad).values())
    junk = [*recs, ("website is under construction", ["1", "1", "0", "0", "1"])]
    assert gates("Licensed Banks", cols, junk)[(5, 0)].startswith("FAIL")


def test_margin_gate():
    recs = [("One- year deposit rate", ["8.43", "8.03"]), ("One- year lending rate", ["15.80", "16.73"]),
            ("Margin between one year lending and deposit rates", ["7.37", "8.70"])]
    g = gates("Selected Bank's Interest Rates", ["2020/21", "2021/22"], recs)
    assert g[(2, 0)].startswith("PASS") and g[(2, 1)].startswith("PASS")
    recs[2] = ("Margin between one year lending and deposit rates", ["7.37", "9.70"])
    assert gates("Selected Bank's Interest Rates", ["2020/21", "2021/22"], recs)[(2, 1)].startswith("FAIL")


def test_code_extracted_failures_never_used(root):
    e = pd.read_csv(path("interim", root) / "bot_code_extracted.csv")
    assert not e.loc[~e["gate_result"].astype(str).str.startswith("PASS"), "used"].astype(bool).any()
    assert e["used"].astype(bool).sum() > 300
    for t in ("rates_annual", "sector_totals"):
        df = pd.read_csv(path("processed", root) / f"{t}.csv")
        code = df[df["precedence"].str.startswith("code")]
        # the D42 policy-rate series may carry the 2024 CBR read from report text (prompt 03b); it stays labelled
        ok = code["series_id"].str.startswith("BOTCODE_") | (
            code["series_id"].eq("POLICY_RATE_D42") & code["precedence"].str.contains("D42"))
        assert ok.all()


# --- processed tables and COE --------------------------------------------------------------------------------------


def test_processed_bot_tables(root):
    p = path("processed", root)
    for t in ("rates_annual", "rates_quarterly", "sector_totals", "credit_by_activity"):
        df = pd.read_csv(p / f"{t}.csv")
        assert len(df) > 100 and df["evidence_class"].notna().all()
        assert not df.duplicated(["series_id", "period_end"]).any()
    q = pd.read_csv(p / "rates_quarterly.csv")
    assert pd.to_datetime(q["period_end"]).dt.month.isin([3, 6, 9, 12]).all()
    a = pd.read_csv(p / "rates_annual.csv")
    owner = a[a["precedence"].str.startswith("owner")]
    assert pd.to_datetime(owner["period_end"]).dt.month.eq(12).all()
    tb = owner[owner["series_id"].str.contains("TREASURY_BILL_91")]
    assert tb["year"].min() <= 2008 and tb["year"].max() >= 2024 and (tb["value"] < 0.3).all()
    c = pd.read_csv(p / "credit_by_activity.csv")
    assert c["year"].nunique() >= 20


def test_coe_back_to_2008(root):
    c = pd.read_csv(path("processed", root) / "coe_inputs_annual.csv")
    p = c[c["series"] == "d34_primary"].set_index("year")
    for y in range(2008, 2013):
        assert p.loc[y, "status"] == "OK (flagged)" and "FLAGGED" in p.loc[y, "note"]
        assert p.loc[y, "tenor_years"] in ("10", 10)
    assert p.loc[2008, "waytm"] == pytest.approx(0.195)
    assert p.loc[2012, "waytm"] == pytest.approx(0.158154)
    assert p.loc[2013, "status"] == "OK"


def test_short_tenor_fallback_prefers_final_quarter_auction():
    a = pd.DataFrame({"bond_number": [321], "issue_number": [1], "bond_auction_seq": [1], "calendar_sn": [1],
                      "auction_date": pd.to_datetime(["2012-11-28"]), "tenor_years": [10], "coupon": [0.1144],
                      "waytm": [0.158154], "weighted_average_price": [78.0]})
    row, used = short_tenor_fallback(a, pd.Timestamp("2012-12-31"), {2012: {"waytm": 0.1582, "tenor": 10,
                                                                            "source": "s", "page": "p"}})
    assert row["waytm"] == 0.158154 and len(used) == 1
    row, used = short_tenor_fallback(a.iloc[0:0], pd.Timestamp("2010-12-31"),
                                     {2010: {"waytm": 0.1359, "tenor": 10, "source": "s", "page": "p"}})
    assert row["waytm"] == 0.1359 and row["status"] == "OK (flagged)" and not used


# --- PDF stage cache ---------------------------------------------------------------------------------------------


def test_fingerprint_changes_with_inputs(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    one = fingerprint([f], [pdf], [])
    assert one == fingerprint([f], [pdf], [])
    f.write_text("x\n2\n")
    assert fingerprint([f], [pdf], []) != one


# --- 02e fixes -----------------------------------------------------------------------------------------------------


def test_02e_fixes(root):
    hf = pd.read_csv(path("interim", root) / "inputs_historical_financials.csv", low_memory=False)
    cites = hf.set_index("Row ID").loc[["HF-0143", "HF-0726", "HF-0727"], "Page / Note"]
    assert cites.str.contains(r"PDF p\.\d+").all()
    long = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False)
    pbt = long[(long["bank_id"] == "B05") & (long["fiscal_year"] == 2021) & (long["canonical_name"] == "profit_before_tax")]
    assert set(pbt["value"]) >= {1_148e6, -2_706_838e3}
    assert pbt.loc[pbt["value"] == 1_148e6, "is_restated"].eq(False).all()
