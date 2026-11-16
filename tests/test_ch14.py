"""Prompt 14: market behaviour and valuation extensions (chapter 14 page)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.config import path
from tzbank.reporting.findings import latest_pack


def _t(root, n):
    return pd.read_csv(path("tables", root) / f"ch14_{n}.csv")


def test_return_attribution_adds_up(root):
    a = _t(root, "return_attribution")
    assert {"NMB", "CRDB"} <= set(a["bank"])
    ok = a.dropna(subset=["eps_growth_log"])
    assert np.allclose(ok["eps_growth_log"] + ok["pe_change_log"] + ok["dividends_log"], ok["log_total_return"])
    b = a.dropna(subset=["bvps_growth_log"])  # CRDB FY2009 book value not held: since-listing BVPS route missing
    assert np.allclose(b["bvps_growth_log"] + b["pb_change_log"] + b["dividends_log"], b["log_total_return"])
    assert np.allclose(ok["log_price_change"], ok["eps_growth_log"] + ok["pe_change_log"])  # CRDB FY2009 EPS missing
    assert (a["dividends_log"] >= 0).all() and a["split_flag"].str.len().gt(20).all()


def test_attribution_hand_check():
    """Two years, one dividend: TR = (P1 + D0) / P0; the components add to ln TR."""
    p0, p1, d0, e0, e1 = 100.0, 150.0, 5.0, 10.0, 12.0
    tr = np.log((p1 + d0) / p0)
    eps_g, pe = np.log(e1 / e0), np.log((p1 / e1) / (p0 / e0))
    assert eps_g + pe + (tr - np.log(p1 / p0)) == pytest.approx(tr)


def test_event_windows(root):
    e = _t(root, "event_study")
    assert set(e["window"]) == {"[-20,-1]", "[0,+20]", "[-5,-1]", "[0,+5]", "[-1,+1]"}
    assert e.groupby(["event", "event_date", "bank", "window"]).size().eq(1).all()
    assert (e[e["window"] == "[0,+20]"]["days"] <= 21).all()
    assert e["benchmark"].str.startswith("all equities ex NMB and CRDB").all()
    assert (e["event_day"] >= e["event_date"]).all()  # first trading day on or after the event date
    # CAR = raw return - (alpha x days + beta x market return); t = CAR / (sd x sqrt(days))
    ok = e.dropna(subset=["car_market", "alpha", "beta"])
    assert np.allclose(ok["car_market"], ok["raw_log_return"] - ok["alpha"] * ok["days"] - ok["beta"]
                       * ok["market_log_return"], atol=1e-9)


def test_car_hand_check():
    """Market model: AR_t = r_t - (a + b m_t); CAR over three days."""
    r, m, a, b = np.array([0.01, -0.02, 0.03]), np.array([0.0, -0.01, 0.01]), 0.001, 1.2
    assert float((r - (a + b * m)).sum()) == pytest.approx(0.02 - 0.003 - 1.2 * 0.0)


def test_variance_ratio_definition():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(size=4000))
    q = 5
    vr = r.rolling(q).sum().dropna().var() / (q * r.var())
    assert vr == pytest.approx(1.0, abs=0.1)  # a random walk has VR near 1


def test_microstructure_is_descriptive(root):
    m = _t(root, "microstructure")
    assert (m["reading"] == "descriptive").all()
    assert {"autocorrelation", "variance ratio", "block vs board", "price clustering", "52-week high",
            "turnover spikes"} <= set(m["test"])


def test_seasonality_tests_reported(root):
    s = _t(root, "seasonality")
    tests = s[s["reading"].str.startswith("test")]
    assert {"jan", "dec"} <= set(tests["month"].astype(str))
    assert tests[["coefficient", "hac_se", "t_stat", "p_value"]].notna().all().all()
    assert np.allclose(tests["t_stat"], tests["coefficient"] / tests["hac_se"])
    assert (tests["years"] >= 8).all()  # 2015-2026
    means = s[s["reading"] == "descriptive"]
    assert set(means["month"].astype(int)) == set(range(1, 13))


def test_split_study(root):
    sp = _t(root, "split_study")
    assert set(sp["window_days"]) == {5, 20, 60}
    d = sp[sp["metric"] == "trading_day_share"]
    assert np.allclose(d["did_vs_crdb"], (d["nmb_post"] - d["nmb_pre"]) - (d["crdb_post"] - d["crdb_pre"]))
    reg = sp[sp["metric"].str.startswith("DiD regression")]
    assert len(reg) == 3 and reg["reading"].str.contains("Newey-West").all()
    o = _t(root, "split_objectives")
    assert {"improve affordability", "improve tradability", "broaden retail participation"} == set(o["stated_objective"])
    aff = o[o["stated_objective"] == "improve affordability"].iloc[0]
    assert aff["pre_60d"] / aff["post_60d"] == pytest.approx(10, rel=0.15)


def test_roll_and_corwin_schultz_hand_checks():
    from tzbank.chapters.ch14_market_data import corwin_schultz, roll_spread

    # bid-ask bounce around a constant value: prices alternate 99, 101 -> negative autocovariance
    px = pd.Series([99.0, 101.0] * 50)
    assert roll_spread(px) == pytest.approx(2 * np.sqrt(np.log(101 / 99) ** 2), rel=0.05)
    r = roll_spread(pd.Series(np.exp(np.linspace(0, 1, 50))))  # constant drift: zero covariance, no spread
    assert np.isnan(r) or r < 1e-6
    hi, lo = pd.Series([101.0] * 20), pd.Series([99.0] * 20)
    s = corwin_schultz(hi, lo)
    d = np.log(101 / 99)  # constant range: alpha = ln(H / L), S = 2(e^a - 1) / (1 + e^a)
    assert s == pytest.approx(2 * (np.exp(d) - 1) / (1 + np.exp(d)), rel=1e-6)


def test_stale_filter(root):
    from tzbank.chapters.ch14_market_data import panel

    p = panel(root)
    assert (p.loc[p["stale"], "lr_nonstale"].isna()).all()
    assert (p.loc[p["volume"].fillna(0) == 0, "stale"]).all()
    m = _t(root, "microstructure")
    v = set(m.loc[m["test"] == "autocorrelation", "version"])
    assert any(x.startswith("unfiltered (affected by stale prices") for x in v)
    assert any(x.startswith("stale-filtered") for x in v)


def test_comovement_and_proxy_check(root):
    c = _t(root, "comovement")
    assert c["correlation"].between(-1, 1).all()
    pc = _t(root, "index_proxy_check")
    dsei = pc[(pc["proxy"] == "all equities (DSEI-like)") & (pc["dse_index"] == "DSEI")].iloc[0]
    assert dsei["days"] > 200 and dsei["daily_return_correlation"] > 0.5


def test_data_quality_reconciliation(root):
    q = _t(root, "data_quality")
    eq = q[q["check"].str.endswith("share of common dates equal")]
    assert (eq["value"] == 1.0).all()
    m = q[q["check"] == "match rate at the 5% tolerance (the owner's flag rule)"]
    assert (m["value"] == 1.0).all()


def test_quarterly_fundamentals(root):
    f = _t(root, "quarterly_fundamentals")
    assert {"NMB", "CRDB"} <= set(f["bank"])
    assert f["cost_of_risk_annualised"].dropna().between(-0.05, 0.15).all()
    assert f["nim_reported"].dropna().between(0.01, 0.3).all()


def test_valuation_extensions(root):
    g = _t(root, "implied_growth")
    solved = g[g["status"].str.startswith("solved")]
    assert (solved["implied_g"] <= solved["g_upper_bound_guard"] + 1e-9).all()
    uns = g[~g["status"].str.startswith("solved")]
    above = uns[uns["status"].str.startswith("price above")]
    assert (above["value_at_upper_bound"] < above["price"]).all()  # even the guard-limit g leaves value below price
    gr = _t(root, "roe_lambda_grid")
    for _, x in gr.groupby(["bank", "lambda"]):
        assert x.sort_values("roe_star")["value_per_share"].is_monotonic_increasing
    ln = _t(root, "price_lines_scenarios")
    w = ln.pivot_table(index=["valuation", "bank"], columns="line", values="price_line")
    assert (w["bear-case value per share"] < w["base-case value per share"]).all()
    assert (w["base-case value per share"] < w["bull-case value per share"]).all()
    r = _t(root, "real_roe_coe")
    assert np.allclose(r["real_roe"], (1 + r["roe"]) / (1 + r["inflation_imf"]) - 1)
    y = _t(root, "yield_minus_tbill").dropna()
    assert np.allclose(y["yield_minus_tbill"], y["dividend_yield"] - y["tbill"])


def test_annual_report_items_page_checked(root):
    a = _t(root, "annual_report_items")
    obs = a[a["status"] == "observed"]
    assert len(obs) >= 40 and (obs["page_check"] == "value found on the cited page").all()
    nd = a[a["status"].astype(str).str.startswith("NOT")]
    assert {"top20_depositors_share", "top20_borrowers_share"} <= set(nd["variable"])
    s = a[a["variable"] == "salary_backed_share_of_loans"]
    assert s["value"].between(0, 1).all()


def test_catalysts_and_thesis_breaks(root):
    c = _t(root, "catalysts")
    assert {"past (dated)", "expected"} <= set(c["status"])
    t = _t(root, "thesis_breaks")
    assert t.groupby("bank").size().ge(5).all() and set(t["status"]) <= {"BREACHED", "not breached"}


def test_ch14_manifest_and_pack(root):
    m = json.loads((path("manifests", root) / "ch14.json").read_text(encoding="utf-8"))
    assert m["run_log_stage"] == "market_ext" and len(m["tables"]) == 22
    p = latest_pack(root)
    pack = pd.read_csv(p, low_memory=False)
    ch = pack[pack["chapter"].astype(str) == "14"]
    assert len(ch) > 100 and (ch["dashboard_page"] == "dashboard/pages/14_market_extensions.py").all()


# --- prompt 15b --------------------------------------------------------------------------------------------------------
def test_comovement_frequencies_and_dimson(root):
    c = _t(root, "comovement")
    f = set(c["frequency"])
    assert {"daily, both non-stale", "weekly (all days)", "monthly (all days)", "daily, Dimson (lags -1, 0, +1)"} <= f
    d = c[c["frequency"] == "daily, Dimson (lags -1, 0, +1)"].iloc[0]
    assert d["correlation"] == pytest.approx(d["cross_corr_lag_minus1"] + d["cross_corr_lag0"]
                                             + d["cross_corr_lag_plus1"])
    r = _t(root, "rolling_correlation")
    assert set(r["window"]) == {"60 trading days (daily returns)", "12 months (monthly returns)"}
    assert r["correlation"].between(-1, 1).all()


def test_dimson_hand_check():
    """y_t = 0.5 x_t + 0.3 x_t-1: the Dimson beta (lags -1..+1) recovers 0.8; the contemporaneous slope alone 0.5."""
    from tzbank.chapters.ch14_market_data import dimson

    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=5000))
    y = 0.5 * x + 0.3 * x.shift(1) + 0.01 * pd.Series(rng.normal(size=5000))
    assert dimson(y, x)["beta_sum"] == pytest.approx(0.8, abs=0.01)


def test_results_events_confirmed_only(root):
    e = _t(root, "results_events")
    inc = e[e["include"].astype(bool)]
    assert inc["date_basis"].str.startswith("confirmed").all()
    assert not e.loc[~e["include"].astype(bool), "date_basis"].str.startswith("confirmed").any()
    assert set(inc["window"]) == {"[0,+5]", "[0,+20]"} and inc[["car_market", "car_t_stat"]].notna().all().all()
    assert inc.groupby(["event_bank", "event_date"])["bank"].nunique().eq(2).all()  # event bank and the other bank


def test_bear_screen_buy_lines(root):
    from tzbank.chapters.ch09_valuation import DOWN

    pl = _t(root, "price_lines_scenarios")
    b = pl[pl["line"].str.startswith("bear-screen")]
    assert len(b) == 4 and set(b["valuation"]) == {"HIST", "CURR"}
    bear = pl[pl["line"] == "bear-case value per share"].set_index(["valuation", "bank"])["price_line"]
    for r in b.itertuples():
        assert bear[(r.valuation, r.bank)] / r.price_line - 1 == pytest.approx(DOWN)
    pack = pd.read_csv(latest_pack(root), low_memory=False)
    assert pack["id"].str.contains("BEAR_SCREEN", case=False).any()
