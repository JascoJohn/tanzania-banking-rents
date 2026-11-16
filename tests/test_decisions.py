"""Prompt 18 owner-decision tables (tzbank.chapters.ch13_decisions): identities against the register base case and
hand-checked values. Reads the generated outputs (run ``python -m tzbank run market_ext`` first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzbank.config import find_root, path

ROOT = find_root()
TAB = path("tables", ROOT)


def _t(name: str) -> pd.DataFrame:
    f = TAB / f"ch13_dec_{name}.csv"
    if not f.exists():
        pytest.skip(f"{f.name} not generated")
    return pd.read_csv(f)


def test_every_table_has_evidence_class():
    files = sorted(TAB.glob("ch13_dec_*.csv"))
    assert len(files) >= 21
    for f in files:
        d = pd.read_csv(f)
        assert "evidence_class" in d and d["evidence_class"].notna().all(), f.name


def test_weight_zero_is_the_register_base_case():
    w = _t("interp_weights")
    v = pd.read_csv(TAB / "ch09_valuations.csv")
    for r in v.itertuples():
        x = w[(w["valuation"] == r.valuation) & (w["bank"] == r.bank) & (w["weight_on_B"] == 0)]
        assert x["value_per_share"].iloc[0] == pytest.approx(r.value_per_share, rel=1e-6)
        assert x["rating"].iloc[0] == r.rating  # the registered ratings are unchanged


def test_er_star_is_the_linear_mixture_and_value_rises_with_weight():
    w = _t("interp_weights")
    assert np.allclose(w["er_star"], (1 - w["weight_on_B"]) * w["er_star_A"] + w["weight_on_B"] * w["er_star_B"])
    for _, g in w.groupby(["valuation", "bank"]):
        assert g.sort_values("weight_on_B")["value_per_share"].is_monotonic_increasing


def test_crossing_weight_brackets_the_price():
    w, c = _t("interp_weights"), _t("crossing_weights")
    for r in c.dropna(subset=["crossing_weight"]).itertuples():
        g = w[(w["valuation"] == r.valuation) & (w["bank"] == r.bank)].sort_values("weight_on_B")
        below = g[g["weight_on_B"] <= r.crossing_weight]["value_per_share"].iloc[-1]
        above = g[g["weight_on_B"] >= r.crossing_weight]["value_per_share"].iloc[0]
        assert below <= r.price <= above


def test_mixture_probability_hand_check():
    m = _t("mc_conditional")
    r = m[m["weight_on_B"] == 0.25].iloc[0]
    assert r["prob_value_above_price_mixture"] == pytest.approx(
        0.75 * r["prob_value_above_price_given_A"] + 0.25 * r["prob_value_above_price_given_B"])


def test_configuration_three_is_the_base_and_two_is_weight_one():
    cfg, w = _t("configurations"), _t("interp_weights")
    for (val, bank), g in cfg.groupby(["valuation", "bank"]):
        ww = w[(w["valuation"] == val) & (w["bank"] == bank)].set_index("weight_on_B")["value_per_share"]
        assert g[g["configuration"].str.startswith("3.")]["value_per_share"].iloc[0] == pytest.approx(ww[0.0])
        assert g[g["configuration"].str.startswith("2.")]["value_per_share"].iloc[0] == pytest.approx(ww[1.0])


def test_bank_coe_register_row_reproduces_register_coe():
    b = _t("bank_coe")
    reg = b[b["basis"] == "register (common)"]
    assert np.allclose(reg["coe"], reg["coe_register"], atol=1e-9)
    # hand check: COE = rf + beta x (ERP + CRP) + LP, CURR evidence line y_long 0.1084, DS 0.0358, ERP 0.0420,
    # CRP 0.0557, beta 0.604, LP 0.015 -> 0.14660 (register 0.146548; the evidence line is rounded)
    assert 0.1084 - 0.0358 + 0.604 * (0.0420 + 0.0557) + 0.015 == pytest.approx(0.146548, abs=1e-4)


def test_terminal_growth_identity():
    t = _t("terminal_growth")
    assert np.allclose(t["g_implied_b_x_roe"], t["g_register"], atol=1e-9)


def test_surplus_drag_headline_and_bounds_order():
    s = _t("surplus_drag")
    for _, g in s.groupby("bank"):
        r = g.set_index(g["role"].str.split(" ").str[0])["drag_share_of_profit"]
        assert r["lower"] <= r["headline"] <= r["upper"]


def test_analyst_register_split_restatement():
    reg = pd.read_csv(ROOT / "data" / "raw" / "analyst_reports" / "register.csv")
    pre = reg[reg["per_share_basis_reported"].str.startswith("pre-split")]
    assert np.allclose(pre["fair_value_post_split"], pre["fair_value_reported"] / 10)
    assert reg["evidence_class"].eq("Observed").all()


def test_monitoring_rows_carry_r_labels():
    m = _t("monitoring")
    assert sorted(m["monitor_row"].dropna().loc[lambda s: s != ""]) == [f"r{i}" for i in range(10)]


def test_rating_rule_switch_points_match_chapter_9():
    r = _t("rating_rule").set_index(["valuation", "bank"])
    assert r.loc[("CURR", "NMB"), "coe_change_to_switch_down"] == pytest.approx(-0.015)
    assert r.loc[("CURR", "CRDB"), "coe_change_to_switch_down"] == pytest.approx(-0.025)  # D93 midpoint COE (was -0.0225)
    assert (r.loc[[("CURR", "NMB"), ("CURR", "CRDB")], "rating"] == "Sell").all()
