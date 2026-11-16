"""Prompt 11: chapter 12, USD returns and implementation (blueprint chapter 12; D60)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch12_usd_implementation import fee_per_side, vdates, window
from tzbank.config import path, settings
from tzbank.extract.bot_fx import align


def _t(root, name):
    return pd.read_csv(path("tables", root) / name)


def test_fx_alignment_hand_case():
    seed = {2010: 1395.7, 2011: 1557.4}
    tabs = [{"file": "a", "page": 1, "eop": [1453.5, 1566.7, 1571.6], "avg": [1395.7, 1557.4, 1571.7], "pub": None},
            {"file": "b", "page": 1, "eop": [9.0, 9.0, 9.0], "avg": [1500.0, 1600.0, 1700.0], "pub": None}]
    placed, left = align(tabs, seed)
    assert [t["file"] for t in placed] == ["a"] and placed[0]["start"] == 2010
    assert [t["file"] for t in left] == ["b"]


def test_fx_series_known_values(root):
    f = pd.read_csv(path("interim", root) / "bot_fx_eop.csv").set_index("year")["tzs_per_usd"]
    assert f[2010] == pytest.approx(1453.5) and f[2015] == pytest.approx(2148.5) and f[2022] == pytest.approx(2308.9)
    yrs = list(f.index)
    assert yrs == list(range(yrs[0], yrs[-1] + 1))
    assert (f.diff().dropna() > -0.05 * f.shift().dropna()).all()  # no implausible jumps from misalignment


def test_usd_translation_and_breakeven_identities(root):
    a = _t(root, "ch12_returns_annual.csv").dropna(subset=["usd_return"])
    assert np.allclose(a["usd_return"], (1 + a["tzs_return"]) * a["fx_start"] / a["fx_end"] - 1)
    assert np.allclose(a["tzs_return"], (a["price_end"] + a["dividends"]) / a["price_start"] - 1)
    from tzbank.chapters.ch12_usd_implementation import usd_total_return

    e = _t(root, "ch12_expected_usd.csv")
    dv = _t(root, "ch09_dividends_12m.csv")
    w = settings(root)["ch12"]["div_wht_foreign"]
    for r in e.itertuples():
        divs = [(x.dps / x.price, x.years_from_valuation)
                for x in dv[(dv["valuation"] == r.valuation) & (dv["bank"] == r.bank)].itertuples()]
        assert usd_total_return(r.tzs_price_return, divs, w, r.breakeven_depreciation) == pytest.approx(r.usd_hurdle)
    assert np.allclose(e["expected_usd_return_after_cost"],
                       e["expected_usd_return"] - e["round_trip_cost"] / (1 + e["depreciation"]))


def test_usd_return_rebuilt_from_components(root):
    """Prompt 24: the USD total return equals the target price and each dividend translated at the rate on its own
    date (FX_t = FX_0 (1 + d)^t), withholding on dividends only, divided by the price in USD at the start."""
    e = _t(root, "ch12_expected_usd.csv")
    v = _t(root, "ch09_valuations.csv").set_index(["valuation", "bank"])
    dv = _t(root, "ch09_dividends_12m.csv")
    w = settings(root)["ch12"]["div_wht_foreign"]
    fx0 = 2500.0  # any starting rate: the return does not depend on it
    for r in e.itertuples():
        p0, p1 = v.loc[(r.valuation, r.bank), "price"], v.loc[(r.valuation, r.bank), "target_price_12m"]
        usd_in = p0 / fx0
        usd_out = p1 / (fx0 * (1 + r.depreciation))
        for x in dv[(dv["valuation"] == r.valuation) & (dv["bank"] == r.bank)].itertuples():
            usd_out += x.dps * (1 - w) / (fx0 * (1 + r.depreciation) ** x.years_from_valuation)
        assert usd_out / usd_in - 1 == pytest.approx(r.expected_usd_return, abs=1e-12)
        assert r.tzs_price_return == pytest.approx(p1 / p0 - 1)
        assert r.excess_over_hurdle == pytest.approx(r.expected_usd_return - r.usd_hurdle)


def test_usd_hand_case():
    from tzbank.chapters.ch12_usd_implementation import usd_total_return

    # price 100 -> 90, one dividend of 4 paid half-way, 5% withholding, 3.8% depreciation
    want = 90 / 1.038 / 100 + 4 * 0.95 / 1.038 ** 0.5 / 100 - 1
    assert usd_total_return(-0.10, [(0.04, 0.5)], 0.05, 0.038) == pytest.approx(want)
    assert usd_total_return(-0.10, [], 0.05, 0.0) == pytest.approx(-0.10)


def test_fee_schedule_hand_cases(root):
    fees = {**settings(root)["ch12"]["fees"], "mode": "schedule"}  # the schedule itself; the default mode is the cap
    assert fee_per_side(1e8, {**fees, "mode": "cap"}) == pytest.approx(fees["max_total_per_side"])
    # TZS 5m: 1.7% brokerage x 1.18 + 0.396% market fees = 2.402%, above the DSE cap of 2.3768% -> capped
    assert fee_per_side(5e6, fees) == pytest.approx(min(0.017 * 1.18 + 0.00396, fees["max_total_per_side"]))
    assert fee_per_side(5e6, fees) == pytest.approx(0.023768)
    # TZS 100m: (10m x 1.7% + 40m x 1.5% + 50m x 0.8%) / 100m x 1.18 + 0.396%
    assert fee_per_side(1e8, fees) == pytest.approx((170e3 + 600e3 + 400e3) / 1e8 * 1.18 + 0.00396)
    assert fee_per_side(1e3, fees) <= fees["max_total_per_side"] + 1e-12


def test_hurdle_uses_the_coe_components(root):
    """D60 amended: hurdle = rf_US + beta (ERP + CRP) + LP with the register's beta and LP (issue 33)."""
    from tzbank.chapters.ch12_usd_implementation import hurdle, macro_at

    reg = pd.read_csv(root / "config" / "assumption_register.csv")
    for val, vd in vdates(root).items():
        for bank in ("NMB", "CRDB"):
            x = reg[(reg["valuation"] == val) & (reg["bank"] == bank)].set_index("symbol")["base"]
            rf = macro_at(root, vd, "macro_us_10_year_treasury_yield")[0]
            erp = macro_at(root, vd, "macro_mature_market_implied_erp")[0]
            crp = macro_at(root, vd, "macro_tanzania_country_risk_premium")[0]
            assert hurdle(root, val, bank)[0] == pytest.approx(rf + x["beta"] * (erp + crp) + x["LP"])
    assert "em_premium" not in settings(root)["ch12"]


def test_dividend_withholding_deducted(root):
    e = _t(root, "ch12_expected_usd.csv")
    wht = settings(root)["ch12"]["div_wht_foreign"]
    assert np.allclose(e["dividend_wht"], wht * e["expected_dividend_yield"])
    assert np.allclose(e["expected_tzs_return_12m"], e["tzs_price_return"] + e["expected_dividend_yield"]
                       - e["dividend_wht"])
    assert np.allclose(e["expected_tzs_return_gross"], e["tzs_price_return"] + e["expected_dividend_yield"])


def test_open_issue_parameters_exist(root):
    """Every settings key named in docs/OPEN_ISSUES.md exists (D55 deferred-fix rule; ch12 review C)."""
    import re

    s = settings(root)
    text = (root / "docs" / "OPEN_ISSUES.md").read_text(encoding="utf-8")
    keys = set(re.findall(r"`((?:capital_rules|valuation|part1|ch12|persistence)\.[a-z_]+)`", text))
    assert keys
    missing = [k for k in keys if k.split(".")[1] not in s.get(k.split(".")[0], {})]
    assert missing == [] or missing == ["ch12.em_premium"], missing
    for grp in ("## (i) Data collection", "## (ii) Owner decisions", "## (iii) Counsel / publication", "## Resolved"):
        assert grp in text


def test_capacity_uses_only_data_to_each_date(root):
    for val, vd in vdates(root).items():
        assert window(root, vd)["date"].max() <= vd
        liq = _t(root, "ch12_liquidity.csv")
        assert pd.Timestamp(liq.loc[liq["valuation"] == val, "window_end"].max()) <= vd


def test_capacity_days_formula(root):
    c = _t(root, "ch12_capacity.csv")
    assert (c["days_to_build_or_exit"] == np.ceil(c["position_tzs"] / c["daily_trade_tzs"])).all()
    assert np.allclose(c["round_trip_cost"], 2 * (c["fee_per_side"] + c["impact_per_side"]))
    assert (c.groupby(["valuation", "bank", "participation"])["days_to_build_or_exit"].apply(
        lambda s: s.is_monotonic_increasing)).all()


def test_market_access_rows_have_source_and_access_date(root):
    a = _t(root, "ch12_market_access.csv")
    assert len(a) >= 10
    for c in ("source_url", "source_title", "access_date", "source_date"):
        assert a[c].notna().all() and (a[c].astype(str).str.strip() != "").all(), c
    assert a["source_url"].str.startswith("http").all()
    assert pd.to_datetime(a["access_date"]).notna().all()
    assert a["status"].str.contains("for counsel review").all()
    topics = set(a["topic"])
    assert {"foreign_ownership", "cmsa", "custody_settlement", "tax", "repatriation", "execution"} <= topics


def test_depreciation_scenarios_point_in_time(root):
    d = _t(root, "ch12_depreciation.csv")
    assert set(d.loc[d["valuation"] == "HIST", "scenario"]) == {"none", "ppp", "history", "stress"}
    # the realised 2026 move is known only at 31 Oct 2026 (prompt 17 B8 sensitivity, never the base)
    cur = d[d["valuation"] == "CURR"]
    assert set(cur["scenario"]) == {"none", "ppp", "history", "stress", "realised_ytd", "realised_ytd_annualised"}
    assert not cur.loc[cur["scenario"].str.startswith("realised"), "base"].any()
    assert (d.groupby("valuation")["base"].sum() == 1).all()
    fx = pd.read_csv(path("interim", root) / "bot_fx_eop.csv")
    assert (pd.to_datetime(fx["publication_date"]) <= vdates(root)["HIST"]).all()  # all usable at 31 Dec 2025


def test_ch12_manifest_note_and_no_legal_conclusions(root):
    m = json.loads((path("manifests", root) / "ch12.json").read_text(encoding="utf-8"))
    assert m["run_log_stage"] == "usd" and m["hypotheses"] == ["H12.1", "H12.2"]
    t = (path("notes", root) / "ch12.md").read_text(encoding="utf-8")
    assert "for counsel review" in t and "no legal conclusions" in t
    assert "nan%" not in t
    h = _t(root, "ch12_hypotheses.csv")
    assert set(h["hypothesis"]) == {"H12.1", "H12.2"} and (h["evidence_class"] == "Interpretation").all()
