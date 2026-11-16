"""Prompt 03a: chapter 4.0-4.3 (extended DuPont, Shapley attribution, revenue, funding, deployment) and the NMB/CRDB
detail lines."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.chapters.ch04.dupont_attribution import FULL, REDUCED, components, roe_full, roe_reduced, shapley
from tzbank.chapters.ch04.funding import funding
from tzbank.chapters.ch04.profitability import dupont_panel
from tzbank.config import path
from tzbank.extract.detail_lines import NOTES, categorise, reconcile

# --- 4.0 DuPont tree ---------------------------------------------------------------------------------------------


def _w():
    return pd.DataFrame({
        "bank_id": ["B02", "B02"], "fiscal_year": [2024, 2025], "total_assets": [1000.0, 1200.0],
        "equity_attributable": [150.0, 170.0], "total_equity": [155.0, 176.0], "interest_income": [90.0, 110.0],
        "interest_expense": [20.0, 26.0], "interest_earning_assets": [800.0, 980.0],
        "customer_deposits": [700.0, 820.0], "borrowings": [50.0, 70.0], "non_interest_income": [30.0, 36.0],
        "operating_expenses": [40.0, 46.0], "impairment_charges": [5.0, 6.0], "profit_before_tax": [55.0, 69.0],
        "net_profit": [38.5, 48.3], "profit_attributable": [38.0, 47.8]})


def test_tree_reconstructs_roe_hand_case():
    c = components(_w())
    r = c.iloc[0]
    assert r["tree"] == "full"
    assert r["roe_reported"] == pytest.approx(47.8 / 160.0)  # attributable profit / average attributable equity
    assert r["roe_tree"] == pytest.approx(r["roe_reported"], abs=1e-14)
    assert r["other_ta"] == pytest.approx((69 - (110 - 26 + 36 - 46 - 6)) / 1100)  # reconciling items = 1/1100


def test_tree_reconstructs_reported_roe_every_bank_year(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", low_memory=False)
    c = components(w).set_index(["bank_id", "fiscal_year"])
    p = dupont_panel(w).set_index(["bank_id", "fiscal_year"])["roe"]
    ok = c["roe_tree"].notna()
    assert ok.mean() > 0.9 and (c.loc[ok, "reconstruction_error"].abs() < 1e-12).all()
    assert (c.loc[~ok, "tree"] == "not computable (a component is missing)").all()
    both = c[ok].join(p, how="inner")
    assert len(both) > 60
    assert ((both["roe_tree"] - both["roe"]).abs() < 1e-12).all()  # same ROE as the chapter 4 starter panel


def test_shapley_sums_to_gap_and_hand_case():
    f = lambda x: x["a"] * x["b"]  # noqa: E731
    sv = shapley(f, {"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 5.0}, ["a", "b"])
    # hand-checked: a: ((3-1)*2 + (3*5-1*5))/2 = (4+10)/2 = 7; b: ((5-2)*1 + (3*5-3*2))/2 = (3+9)/2 = 6
    assert sv["a"] == pytest.approx(7.0) and sv["b"] == pytest.approx(6.0)
    assert sum(sv.values()) == pytest.approx(f({"a": 3, "b": 5}) - f({"a": 1, "b": 2}))
    base = {n: 0.1 for n in FULL} | {"ta_e": 8.0, "attr_share": 1.0, "tax_rate": 0.3}
    tgt = {n: 0.12 for n in FULL} | {"ta_e": 7.0, "attr_share": 0.99, "tax_rate": 0.28}
    sv = shapley(roe_full, base, tgt, FULL)
    assert sum(sv.values()) == pytest.approx(roe_full(tgt) - roe_full(base), abs=1e-12)
    base_r = {n: 0.05 for n in REDUCED} | {"ta_e": 9.0, "tax_rate": 0.3, "attr_share": 1.0}
    tgt_r = {n: 0.04 for n in REDUCED} | {"ta_e": 7.5, "tax_rate": 0.25, "attr_share": 1.0}
    assert sum(shapley(roe_reduced, base_r, tgt_r, REDUCED).values()) == pytest.approx(
        roe_reduced(tgt_r) - roe_reduced(base_r), abs=1e-12)


def test_attribution_outputs_sum(root):
    t = path("tables", root)
    gap = pd.read_csv(t / "ch04_4_0_gap_attribution.csv")
    s = gap.groupby("fiscal_year").agg(c=("contribution", "sum"), g=("gap", "first"))
    assert len(s) >= 10 and ((s["c"] - s["g"]).abs() < 1e-12).all()
    chg = pd.read_csv(t / "ch04_4_0_change_attribution.csv")
    s = chg.groupby(["bank_id", "comparison", "tree"]).agg(c=("contribution", "sum"), g=("roe_change", "first"))
    assert ((s["c"] - s["g"]).abs() < 1e-12).all()
    assert {"2019 -> 2025", "2015-18 (stress) -> 2019-25"} <= set(chg["comparison"])
    assert chg.loc[~chg["bank_id"].isin(["B01", "B02"]), "flag"].str.startswith("reduced").all()


# --- detail lines ------------------------------------------------------------------------------------------------


def test_note_reconcile_and_categories():
    lines = [("term loans", [300.0, 250.0], 1), ("overdrafts", [100.0, 90.0], 2), ("treasury bills", [50.0, 40.0], 3),
             ("treasury bonds", [70.0, 60.0], 4), ("placements with other banks", [5.0, 4.0], 5),
             ("total interest income", [525.0, 444.0], 6)]
    comps, total = reconcile(lines, 0)
    assert total == 525.0 and len(comps) == 5
    cats = categorise(comps, NOTES["interest_income"])
    assert cats == {"Interest Income: Loans and Advances": 400.0, "Interest Income: Government Securities": 120.0,
                    "Interest Income: Placements and Interbank": 5.0}
    assert sum(cats.values()) == total
    sub = [("term loans", [300.0], 1), ("overdrafts", [100.0], 2), ("loans and advances", [400.0], 3),
           ("bills", [50.0], 4), ("total", [450.0], 5)]
    comps, total = reconcile(sub, 0)
    assert total == 450.0 and [c for c, _ in comps] == ["term loans", "overdrafts", "bills"]  # subtotal skipped


def test_detail_components_sum_to_totals(root):
    d = pd.read_csv(root / "data" / "extraction" / "detail_lines.csv")
    for kind, total_var in [("interest_income", "Interest Income: Note Total"),
                            ("fee_income", "Fee and Commission Income")]:
        k = d[d["note_kind"] == kind]
        tot = k[k["variable"] == total_var].set_index(["bank_id", "fiscal_year"])["value"]
        parts = k[k["variable"] != total_var].groupby(["bank_id", "fiscal_year"])["value"].sum()
        assert len(tot) >= 15
        both = pd.concat([tot, parts], axis=1, keys=["t", "p"]).dropna()
        assert ((both["p"] - both["t"]).abs() <= 0.001 * both["t"]).all()
    ii = d[d["variable"] == "Interest Income: Note Total"]
    assert ((ii["value"] / ii["audited_total"] - 1).abs() <= 0.02).all()  # gate against audited interest income


def test_wide_detail_columns_sum(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", low_memory=False)
    f = w[w["bank_id"].isin(["B01", "B02"]) & w["interest_income_loans"].notna()]
    parts = f[["interest_income_loans", "interest_income_securities", "interest_income_placements",
               "interest_income_other"]].fillna(0).sum(axis=1)
    # printed note lines are rounded to TZS '000 or million: allow 0.1% (the extraction gate)
    assert len(f) >= 15 and ((parts - f["interest_income_note_total"]).abs()
                             <= 0.001 * f["interest_income_note_total"]).all()


# --- 4.2 funding -----------------------------------------------------------------------------------------------


def test_gdfa_proxy_hand_case():
    w = pd.DataFrame({"bank_id": ["B02", "B02"], "fiscal_year": [2024, 2025], "customer_deposits": [800.0, 1000.0],
                      "interest_expense_deposits": [10.0, 18.0], "interest_expense": [15.0, 25.0],
                      "borrowings": [50.0, 50.0], "total_assets": [1100.0, 1300.0], "total_equity": [150.0, 170.0],
                      "deposits_current": [300.0, 400.0], "deposits_savings": [300.0, 350.0],
                      "deposits_time": [200.0, 250.0]})
    f = funding(w, pd.Series({2025: 0.10})).iloc[1]
    assert f["cost_of_deposits"] == pytest.approx(18 / 900)
    assert f["gdfa_proxy_tzs"] == pytest.approx((0.10 - 18 / 900) * 900)
    assert f["casa_share"] == pytest.approx(0.75)
    assert "proxy" in f["measure_label"]


def test_ch04_manifests_and_notes(root):
    import json

    for sub in ("4_0", "4_1", "4_2", "4_3"):
        m = json.loads((path("manifests", root) / f"ch04_{sub}.json").read_text(encoding="utf-8"))
        assert m["tables"] and m["notes"]
        for tb in m["tables"]:
            assert pd.read_csv(root / tb["path"])["evidence_class"].notna().all()
        txt = (root / m["notes"][0]).read_text(encoding="utf-8")
        for h in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                  "## Evidence class", "## Limitations", "Cross-cutting explanations", "## Interpretation"):
            assert h in txt, (sub, h)
