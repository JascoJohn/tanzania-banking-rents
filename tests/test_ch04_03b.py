"""Prompt 03b: chapter 4.4-4.7 (transactions, operating leverage, risk, business-model map), the D42 correction and
the shared n/a formatter."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch04.common import num, pct
from tzbank.chapters.ch04.dupont_attribution import components
from tzbank.chapters.ch04.operating_leverage import leverage, scale_elasticity
from tzbank.chapters.ch04.risk import risk
from tzbank.chapters.ch04.transactions import bank_metrics
from tzbank.config import path


def _w():
    return pd.DataFrame({
        "bank_id": ["B02", "B02"], "fiscal_year": [2024, 2025], "total_assets": [1000.0, 1200.0],
        "fee_income_total": [20.0, 24.0], "fee_income_digital": [5.0, 6.0], "fx_income": [4.0, 5.5],
        "non_interest_income": [30.0, 36.0], "kpi_customers": [100.0, 120.0], "kpi_agents": [10.0, 12.0],
        "net_interest_income": [70.0, 84.0], "operating_expenses": [-40.0, -46.0], "staff_costs": [20.0, 23.0],
        "kpi_employees": [50.0, 60.0], "net_profit": [38.5, 48.3], "impairment_charges": [-5.0, -6.0],
        "gross_loans": [600.0, 700.0], "stage3_loans_gross": [18.0, 21.0], "loan_loss_allowance": [-15.0, -20.0],
        "interest_earning_assets": [800.0, 980.0], "rwa_total": [700.0, 800.0]})


# --- 4.4 transactions --------------------------------------------------------------------------------------------


def test_fee_and_fx_ratios_hand_case():
    r = bank_metrics(_w()).set_index("fiscal_year").loc[2025]
    assert r["fee_to_avg_assets"] == pytest.approx(24 / 1100)
    assert r["fx_to_avg_assets"] == pytest.approx(5.5 / 1100)
    assert r["fee_share_of_nonii"] == pytest.approx(24 / 36)
    assert r["digital_share_of_fees"] == pytest.approx(0.25)
    assert r["fee_per_customer"] == pytest.approx(0.2)
    assert r["fee_per_customer_low"] < r["fee_per_customer"] < r["fee_per_customer_high"]
    assert r["fee_per_customer_confidence"].startswith("MEDIUM")


def test_market_context_is_labelled_market_level(root):
    m = pd.read_csv(path("tables", root) / "ch04_4_4_market_context.csv")
    assert len(m) > 0 and (m["level"] == "market-level (all institutions; not bank-level)").all()
    assert "bank_id" not in m.columns
    assert m["evidence_class"].notna().all()


# --- 4.5 operating leverage --------------------------------------------------------------------------------------


def test_incremental_margin_and_cost_ratios_hand_case():
    r = leverage(_w()).set_index("fiscal_year").loc[2025]
    assert r["cost_to_income"] == pytest.approx(46 / 120)
    assert r["opex_to_avg_assets"] == pytest.approx(46 / 1100)
    # (Delta income - Delta OpEx) / Delta income = (20 - 6) / 20
    assert r["incremental_margin"] == pytest.approx(0.7)
    assert r["staff_cost_per_employee"] == pytest.approx(23 / 60)
    assert r["kpi_confidence"].startswith("MEDIUM")


def test_scale_elasticity_hand_case():
    years = np.arange(2010, 2020)
    ta = 1000 * 1.1 ** np.arange(10)
    w = pd.DataFrame({"bank_id": "B99", "fiscal_year": years, "total_assets": ta,
                      "operating_expenses": 0.05 * ((ta + np.r_[np.nan, ta[:-1]]) / 2) ** 0.8})
    e = scale_elasticity(w).iloc[0]
    assert e["elasticity"] == pytest.approx(0.8, abs=1e-9)
    assert e["n"] == 9 and e["reading"].startswith("scale economies")


def test_45_and_46_terms_equal_40_tree(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    w = w.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    c = components(w).set_index(["bank_id", "fiscal_year"])
    lev = pd.read_csv(path("tables", root) / "ch04_4_5_operating_leverage.csv", dtype={"bank_id": str})
    rk = pd.read_csv(path("tables", root) / "ch04_4_6_risk.csv", dtype={"bank_id": str})
    j = lev.set_index(["bank_id", "fiscal_year"]).join(c[["opex_ta", "imp_ta"]], how="inner").dropna(
        subset=["opex_ta", "opex_to_avg_assets"])
    assert len(j) > 40 and np.allclose(j["opex_to_avg_assets"], j["opex_ta"], rtol=0, atol=1e-12)
    j = rk.set_index(["bank_id", "fiscal_year"]).join(c[["imp_ta"]], how="inner").dropna(
        subset=["imp_ta", "impairment_to_avg_assets"])
    assert len(j) > 40 and np.allclose(j["impairment_to_avg_assets"], j["imp_ta"].abs(), rtol=0, atol=1e-12)


def test_kpi_ratio_columns_flagged_medium(root):
    t = pd.read_csv(path("tables", root) / "ch04_4_4_transactions.csv")
    for k in ("customer", "agent"):
        s = t[t[f"fee_per_{k}"].notna()]
        assert (s[f"fee_per_{k}_confidence"].str.startswith("MEDIUM")).all()
    lev = pd.read_csv(path("tables", root) / "ch04_4_5_operating_leverage.csv")
    kpi_cols = [c for c in lev.columns if c.endswith(("_per_employee", "_per_branch", "_per_agent"))]
    has = lev[kpi_cols].notna().any(axis=1)
    assert has.any() and (lev.loc[has, "kpi_confidence"].str.startswith("MEDIUM")).all()


# --- 4.6 risk ----------------------------------------------------------------------------------------------------


def test_risk_adjusted_nim_hand_case():
    r = risk(_w()).set_index("fiscal_year").loc[2025]
    assert r["cost_of_risk"] == pytest.approx(6 / 650)
    assert r["nim"] == pytest.approx(84 / 890)
    assert r["risk_adjusted_nim"] == pytest.approx(84 / 890 - 6 / 650)
    assert r["npl_ratio"] == pytest.approx(0.03)
    assert r["coverage"] == pytest.approx(20 / 21)
    assert r["return_on_rwa"] == pytest.approx(48.3 / 750)


# --- 4.7 business-model map (reads both files independently) ------------------------------------------------------


def test_map_agrees_with_40_attribution(root):
    m = pd.read_csv(path("tables", root) / "ch04_business_model_map.csv")
    rk = pd.read_csv(path("tables", root) / "ch04_4_0_subsection_ranking.csv")
    rk = rk[rk["comparison"] == "NMB - CRDB"]
    assert not m["contradiction_flag"].any(), m[m["contradiction_flag"]][["dimension", "item"]]
    assert m["evidence_for"].fillna("").str.strip().ne("").all()  # every claim has supporting evidence
    assert m["evidence_class"].notna().all()
    for _, r in m.iterrows():  # independent re-check of named winners against the 4.0 ranking
        if r["winner"] not in ("NMB", "CRDB"):
            continue
        s = rk[rk["subsection"] == r["attribution_subsection"]]
        if r["attribution_window"] == "2015-2018":
            s = s[s["fiscal_year"].between(2015, 2018)]
        v = s["contribution"].mean()
        assert abs(v) >= 0.005 and (v > 0) == (r["winner"] == "NMB"), r["dimension"]
        assert r["attribution_mean_gap_pp"] == pytest.approx(round(v * 100, 2), abs=0.011)
    dims = set(m["dimension"])
    assert all(any(d.startswith(f"question {i}:") for d in dims) for i in range(1, 9))
    assert {"hypothesis H4.1", "hypothesis H4.2"} <= dims


# --- D42 and formatter ---------------------------------------------------------------------------------------------


def test_d42_policy_rate_discount_then_cbr(root):
    r = pd.read_csv(path("processed", root) / "rates_annual.csv")
    p = r[r["series_id"] == "POLICY_RATE_D42"].set_index("year")
    assert p.loc[2024, "value"] == pytest.approx(0.06)
    assert "Central Bank Rate" in p.loc[2024, "series_name"]
    assert p.loc[p.index <= 2023, "series_name"].str.contains("discount rate").all()
    assert p.loc[p.index >= 2024, "series_name"].str.contains("Central Bank Rate").all()


def test_formatter_never_nan():
    assert pct(float("nan")) == "n/a (not held)" and pct(None, reason="x") == "n/a (x)"
    assert pct(0.1234, 1) == "12.3%" and pct(0.01, 2, sign=True) == "+1.00%"
    assert num(np.nan).startswith("n/a") and num(1234.5) == "1,234"


def test_notes_never_print_nan_and_ch04_index(root):
    for f in path("notes", root).glob("ch04*.md"):
        t = f.read_text(encoding="utf-8")
        assert "nan%" not in t and "nan pp" not in t, f.name
    idx = json.loads((path("manifests", root) / "ch04.json").read_text(encoding="utf-8"))
    subs = [s["subsection"] for s in idx["subsections"]]
    assert subs == [f"4.{i}" for i in range(8)]
    for s in idx["subsections"]:
        assert (root / s["manifest"]).exists(), s["manifest"]
