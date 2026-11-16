"""Prompt 05: chapter 6 (funding architecture) and the SMR / deposit-insurance text extraction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch06_funding import asym_beta, beta_ols, smr_annual
from tzbank.config import path


def test_preflight_deposit_interest_coverage(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str})
    for b in ("B01", "B02"):
        d = w[(w["bank_id"] == b) & w["customer_deposits"].notna() & (w["fiscal_year"] >= 2009)]
        assert d["interest_expense_deposits"].notna().all(), b  # every year with deposits has deposit interest


def test_smr_events_chain_and_bot_long(root):
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    s = p[(p["kind"] == "SMR") & (p["scope"] == "private deposit liabilities")].sort_values("event_date")
    assert list(s["value"]) == [0.10, 0.08, 0.10, 0.08, 0.07, 0.06]
    assert s["gate_result"].str.startswith("PASS").all()
    assert s["pdf_page"].notna().all() and s["quote"].str.len().gt(20).all()
    b = pd.read_csv(path("interim", root) / "bot_long.csv", low_memory=False)
    r = b[(b["file"] == "data/interim/bot_policy_text.csv") & (b["category"] == "reserve_requirements")]
    assert len(r) == len(p[p["kind"] == "SMR"]) and r["page_table"].str.startswith("text, p.").all()
    assert (b["category"] == "deposit_insurance").any()
    # cross-check with BoT Table A3.15 (owner database, D40): private-deposit ratio 10% in 2009-14, as in the text
    a315 = b[b["series_name"].astype(str).str.contains("Minimum reserve ratio - total private deposits")]
    assert len(a315) > 20 and np.allclose(a315["value_numeric"].astype(float), 0.10)


def test_smr_annual_time_weighting():
    ev = pd.DataFrame({"event_date": pd.to_datetime(["2000-01-15", "2017-04-15", "2019-07-15"]),
                       "value": [0.10, 0.08, 0.07]})
    a = smr_annual(ev, [2016, 2017, 2019])
    assert a.loc[2016, "smr_avg"] == pytest.approx(0.10)
    # 2017: 10% for Jan 1 - Apr 14 (104 days), 8% for 261 days
    assert a.loc[2017, "smr_avg"] == pytest.approx((0.10 * 104 + 0.08 * 261) / 365)
    assert a.loc[2019, "smr_year_end"] == 0.07


def test_net_franchise_below_gdfa_every_year_and_bands(root):
    f = pd.read_csv(path("tables", root) / "ch06_franchise.csv", dtype={"bank_id": str})
    ok = f["net_franchise_central"].notna()
    assert ok.sum() > 50
    assert (f.loc[ok, "net_franchise_central"] < f.loc[ok, "gdfa"]).all()
    assert (f.loc[ok, "net_franchise_high"] < f.loc[ok, "gdfa"]).all()
    assert np.allclose(f.loc[ok, "gdfa_pct_assets"] * 0 + 1, 1) and f.loc[ok, "gdfa_pct_assets"].notna().mean() > 0.9
    for unit in ("", "_pct_equity", "_roe_pts", "_pct_assets"):  # every net figure carries its band, ordered low <= central <= high
        lo, c, hi = (f.loc[ok, f"net_franchise_{k}{unit}"] for k in ("low", "central", "high"))
        both = lo.notna() & hi.notna()
        assert (lo[both] <= c[both] + 1e-12).all() and (c[both] <= hi[both] + 1e-12).all()
    assert f["measure_label"].str.contains("proxy").all()


def test_reconciliation_with_40_funding_term(root):
    f = pd.read_csv(path("tables", root) / "ch06_franchise.csv", dtype={"bank_id": str})
    r = f["reconciliation_residual"].dropna()
    assert len(r) > 40 and (r.abs() < 1e-10).all()
    c = pd.read_csv(path("tables", root) / "ch04_4_0_components.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    x = f.set_index(["bank_id", "fiscal_year"]).loc[("B02", 2025)]
    k = (1 - c.loc[("B02", 2025), "tax_rate"]) * c.loc[("B02", 2025), "ta_e"] * c.loc[("B02", 2025), "attr_share"]
    assert x["funding_term_40_roe"] == pytest.approx(-c.loc[("B02", 2025), "c_fund"] * c.loc[("B02", 2025), "ibl_ta"] * k)


def test_deposit_beta_rates_dated_within_period(root):
    r = pd.read_csv(path("tables", root) / "ch06_rates_used.csv")
    r = r[r["tbill"].notna()]
    assert (pd.to_datetime(r["tbill_first_date"]).dt.year == r["year"]).all()
    assert (pd.to_datetime(r["tbill_last_date"]).dt.year == r["year"]).all()
    p = r[r["policy_rate"].notna()]
    assert (pd.to_datetime(p["policy_date"]).dt.year == p["year"]).all()


def test_beta_hand_cases():
    dr = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005])
    o = beta_ols(0.3 * dr + 0.001, dr)
    assert o["beta"] == pytest.approx(0.3) and o["lo"] <= 0.3 <= o["hi"]
    dr = np.array([0.01, 0.02, 0.015, 0.005, -0.02, -0.01, -0.015, -0.005])
    dc = np.where(dr > 0, 0.5 * dr, 0.2 * dr)
    a = asym_beta(dc, dr)
    assert a["beta_up"] == pytest.approx(0.5) and a["beta_down"] == pytest.approx(0.2)
    assert asym_beta(dc[:5], dr[:5])["status"].startswith("not estimated")


def test_ch06_manifest_note_hypotheses(root):
    m = json.loads((path("manifests", root) / "ch06.json").read_text(encoding="utf-8"))
    assert m["chapter"] == "06"
    for t in m["tables"] + m["figures"]:
        assert (root / t["path"]).exists()
    h = pd.read_csv(path("tables", root) / "ch06_hypotheses.csv")
    assert list(h["hypothesis"]) == ["H6.1", "H6.2", "H5.3 (revisited)"]
    assert h["verdict"].str.match(r"^(supported|rejected|inconclusive)").all()
    note = (path("notes", root) / "ch06.md").read_text(encoding="utf-8")
    for head in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                 "## Evidence class", "## Limitations", "## Cross-cutting explanations"):
        assert head in note
    assert "nan%" not in note and "proxy" in note
