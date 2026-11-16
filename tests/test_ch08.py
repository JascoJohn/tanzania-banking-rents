"""Prompt 07: chapter 8 (credit risk and capital efficiency)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch08_credit import capital_efficiency, credit_history, h81, stress_lens, ttc_table
from tzbank.config import path


def _w():
    yrs = list(range(2009, 2026))
    n = len(yrs)
    return pd.DataFrame({
        "bank_id": "B02", "fiscal_year": yrs, "impairment_charges": [10.0] * n,
        "gross_loans": [1000.0 + 100 * i for i in range(n)], "stage3_loans_gross": [50.0] * n,
        "loan_loss_allowance": [40.0] * n, "profit_before_tax": [60.0] * n, "net_profit": [42.0] * n,
        "total_assets": [3000.0] * n, "rwa_total": [2000.0] * n})


def test_cost_of_risk_hand_case_and_regime():
    h = credit_history(_w()).set_index("fiscal_year")
    # 2018: 10 / ((1000 + 900 + 1000 + 800) / 2) with loans 1900 (2018) and 1800 (2017)
    assert h.loc[2018, "cost_of_risk"] == pytest.approx(10 / ((1900 + 1800) / 2))
    assert h.loc[2017, "accounting_regime"] == "IAS 39" and h.loc[2018, "accounting_regime"] == "IFRS 9"
    assert h.loc[2018, "npl_ratio"] == pytest.approx(50 / 1900)
    assert h.loc[2018, "coverage"] == pytest.approx(40 / 50)
    assert np.isnan(h.loc[2009, "cost_of_risk"])  # no opening loans: no average, no imputation


def test_ttc_cycle_includes_stress_and_grid():
    h = credit_history(_w())
    t = ttc_table(h)
    p = t[t["definition_type"] == "primary"].iloc[0]
    assert (p["start"], p["end"]) == (2010, 2025) and p["covers_2015_18"]
    g = t[t["definition_type"] == "boundary grid"]
    assert len(g) == 24 and (g["start"] <= 2015).all() and (g["end"] >= 2018).all()
    x = h[h["fiscal_year"].between(2010, 2025)]
    assert p["ttc_mean"] == pytest.approx(x["cost_of_risk"].mean())
    assert p["ttc_pooled"] == pytest.approx(x["impairment"].sum() / x["avg_gross_loans"].sum())


def test_risk_adjusted_rorwa_hand_case():
    w = _w()
    h = credit_history(w)
    t = ttc_table(h)
    cb = pd.DataFrame({"bank_id": "B02", "fiscal_year": w["fiscal_year"], "rwa_total": 2000.0,
                       "capital_basis": "Bank-solo (D2)"})
    e = capital_efficiency(w, cb, h, t).set_index("fiscal_year")
    ttc = t.loc[t["definition_type"] == "primary", "ttc_mean"].iloc[0]
    al = (1900 + 1800) / 2
    assert e.loc[2018, "return_on_rwa"] == pytest.approx(42 / 2000)
    assert e.loc[2018, "risk_adjusted_rorwa"] == pytest.approx((60 + 10 - ttc * al) / 2000)
    assert e.loc[2018, "rwa_density"] == pytest.approx(2000 / 3000)


def test_ifrs9_break_not_smoothed(root):
    h = pd.read_csv(path("tables", root) / "ch08_credit_history.csv", dtype={"bank_id": str})
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    for b in ("B01", "B02"):
        x = w[w["bank_id"] == b].set_index("fiscal_year")
        raw = x.loc[2018, "impairment_charges"] / ((x.loc[2018, "gross_loans"] + x.loc[2017, "gross_loans"]) / 2)
        r = h[(h["bank_id"] == b) & (h["fiscal_year"] == 2018)].iloc[0]
        assert r["cost_of_risk"] == pytest.approx(raw)  # as computed: no transition add-back, no smoothing
        assert r["ifrs9_transition_year"] and r["accounting_regime"] == "IFRS 9"
    tr = pd.read_csv(path("tables", root) / "ch08_ifrs9_transition.csv", dtype={"bank_id": str})
    v = tr.set_index(["bank_id", "item"])["value"]
    assert v[("B01", "ifrs9_adjustment_retained_earnings")] == pytest.approx(-21_132e6)
    assert v[("B01", "allowance_increase")] == pytest.approx(21_345e6)
    assert v[("B02", "equity_impact_net")] == pytest.approx(-11_560e6)
    assert v[("B02", "fair_value_reserve_adjustment")] == pytest.approx(826e6)
    assert set(tr["evidence_class"]) <= {"Observed", "Derived"}


def test_ttc_sensitivity_table(root):
    t = pd.read_csv(path("tables", root) / "ch08_ttc_cost_of_risk.csv", dtype={"bank_id": str})
    for b in ("B01", "B02"):
        x = t[t["bank_id"] == b]
        assert (x["definition_type"] == "primary").sum() == 1
        g = x[x["definition_type"].isin(["primary", "boundary grid"])]
        assert len(g) == 25 and g["covers_2015_18"].all()
        assert {"stress only FY2015-18", "IAS 39 regime FY2010-17", "IFRS 9 regime FY2018-25",
                "primary cycle excluding FY2018"} <= set(x["definition"])


def test_stress_lens_reproduces_tree_at_actual(root):
    s = pd.read_csv(path("tables", root) / "ch08_stress_lens.csv", dtype={"bank_id": str})
    a = s[s["cost_of_risk_scenario"] == "actual"]
    assert np.allclose(a["roe_scenario"], a["roe_tree"])
    assert np.allclose(a["roe_tree"], a["roe_reported"], atol=1e-9)
    st = s[s["cost_of_risk_scenario"] == "2015-18 mean (own)"]
    assert (st["evidence_class"] == "Scenario").all()
    # a higher cost of risk can only lower ROE when everything else is held
    m = st.merge(a, on=["bank_id", "fiscal_year"], suffixes=("", "_a"))
    assert ((m["cost_of_risk"] > m["cost_of_risk_a"]) <= (m["roe_scenario"] < m["roe_scenario_a"])).all()


def test_stress_lens_hand_case():
    from tzbank.chapters.ch04.dupont_attribution import roe_full

    x = {"iea_ta": 0.9, "y_iea": 0.1, "ibl_ta": 0.8, "c_fund": 0.03, "nonii_ta": 0.02, "opex_ta": 0.03,
         "imp_ta": 0.01, "other_ta": 0.0, "tax_rate": 0.3, "ta_e": 6.0, "attr_share": 1.0}
    base = roe_full(x)
    y = dict(x, imp_ta=0.02)
    assert roe_full(y) - base == pytest.approx(-0.01 * 0.7 * 6.0)


def test_h81_bootstrap_reproducible():
    w = _w()
    w.loc[w["fiscal_year"].between(2015, 2018), "impairment_charges"] = 60.0
    h = credit_history(w)
    t = ttc_table(h)
    boot = {"reps": 400, "block": 3, "seed": [5, 8]}
    a = h81(h.assign(bank_id="B02"), t, boot, {"B02": "NMB", "B01": "CRDB"})
    b = h81(h.assign(bank_id="B02"), t, boot, {"B02": "NMB", "B01": "CRDB"})
    pd.testing.assert_frame_equal(a, b)
    r = a[a["current_definition"] == "FY2025"].iloc[0]
    assert r["gap"] < 0 and r["gap_lo_block"] <= r["gap"] <= r["gap_hi_block"] + 1e-12


def test_ch08_outputs_manifest_and_note(root):
    m = json.loads((path("manifests", root) / "ch08.json").read_text(encoding="utf-8"))
    assert m["chapter"] == "08"
    for t in m["tables"] + m["figures"]:
        assert (root / t["path"]).exists()
        if "data" in t:
            assert (root / t["data"]).exists()
    for t in m["tables"]:
        df = pd.read_csv(root / t["path"])
        assert "evidence_class" in df.columns and df["evidence_class"].notna().all()
    h = pd.read_csv(path("tables", root) / "ch08_hypotheses.csv")
    assert list(h["hypothesis"]) == ["H8.1", "H8.2"]
    p = pd.read_csv(path("tables", root) / "ch08_panel_summary.csv", dtype={"bank_id": str})
    assert len(p) >= 5 and p["coverage_flag"].notna().all()  # seven-bank panel where RWA is held, flagged
    note = (path("notes", root) / "ch08.md").read_text(encoding="utf-8")
    for head in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                 "## Evidence class", "## Limitations", "## Cross-cutting explanations", "IFRS 9"):
        assert head in note
    assert "nan%" not in note


def test_stress_lens_function_on_toy_panel():
    w = _w()
    w["interest_income"] = 300.0
    w["interest_expense"] = 100.0
    w["non_interest_income"] = 50.0
    w["operating_expenses"] = 120.0
    w["profit_before_tax"] = 300.0 - 100 + 50 - 120 - 10
    w["net_profit"] = w["profit_before_tax"] * 0.7
    w["profit_attributable"] = w["net_profit"]
    w["equity_attributable"] = 500.0
    w["total_equity"] = 500.0
    w["interest_earning_assets"] = 2700.0
    w["customer_deposits"] = 2200.0
    w["borrowings"] = 100.0
    w2 = pd.concat([w, w.assign(bank_id="B01")], ignore_index=True)
    h = credit_history(w2)
    s = stress_lens(w2, h, ttc_table(h))
    a = s[(s["cost_of_risk_scenario"] == "actual")]
    assert np.allclose(a["roe_scenario"], a["roe_reported"])
