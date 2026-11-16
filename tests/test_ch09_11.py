"""Prompt 09b: chapters 9-11 (valuations, implied expectations, scenarios)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch09_valuation import rating
from tzbank.config import path


def test_rating_rule_hand_cases():
    assert rating(0.25, 0.15, -0.10)[0] == "Buy"  # > COE + 5 pp, downside within 25%
    assert rating(0.25, 0.15, -0.30)[0] == "Hold"  # failed downside test: Buy -> Hold, never Sell
    assert rating(0.05, 0.15, -0.10)[0] == "Sell"  # < COE - 5 pp
    assert rating(0.16, 0.15, -0.90)[0] == "Hold"
    assert rating(0.20, 0.15, -0.10)[0] == "Hold"  # exactly COE + 5 pp is not a Buy


def test_every_published_value_is_stamped(root):
    reg = pd.read_csv(root / "config" / "assumption_register.csv")
    ver = str(reg["version"].iloc[0])
    assert ver == "v1.3" and reg["status"].str.startswith("APPROVED v1.3").all()
    for ch in ("09", "10", "11"):
        m = json.loads((path("manifests", root) / f"ch{ch}.json").read_text(encoding="utf-8"))
        assert m["register_version"] == ver and len(m["input_hash"]) == 16
        for t in m["tables"] + m["figures"]:
            df = pd.read_csv(root / t["path"] if "data" not in t else root / t["data"])
            assert (df["register_version"] == ver).all() and (df["input_hash"] == m["input_hash"]).all()
            assert df["evidence_class"].notna().all()
        note = (path("notes", root) / f"ch{ch}.md").read_text(encoding="utf-8")
        assert m["input_hash"] in note and ver in note and "nan" not in note.replace("financ", "")


def test_valuations_mechanics(root):
    v = pd.read_csv(path("tables", root) / "ch09_valuations.csv").set_index(["valuation", "bank"])
    assert (v["identity_residual"].abs() < 1e-9).all()
    assert (v["rim_ddm_gap"] < 0.10).all()
    assert v.loc[("CURR", "NMB"), "shares"] == pytest.approx(5e9)  # post-split at 31 Oct 2026
    assert v.loc[("HIST", "NMB"), "shares"] == pytest.approx(5e8)
    assert v.loc[("HIST", "NMB"), "value_per_share_post_split_basis"] == pytest.approx(
        v.loc[("HIST", "NMB"), "value_per_share"] / 10)
    for r in v.itertuples():
        exp = (r.target_price_12m + r.expected_dps_12m - r.price) / r.price
        assert r.expected_total_return == pytest.approx(exp)
        assert r.rating == rating(r.expected_total_return, r.coe, r.bear_downside)[0]
        assert r.bear_value_per_share < r.value_per_share
    o = pd.read_csv(path("tables", root) / "ch09_override_log.csv")
    assert o.empty and {"override_rating", "rationale"} <= set(o.columns)


def test_sensitivity_grid_includes_coe_200bp(root):
    g = pd.read_csv(path("tables", root) / "ch09_sensitivity_grids.csv")
    v = pd.read_csv(path("tables", root) / "ch09_valuations.csv").set_index(["valuation", "bank"])
    for (val, bank), x in g[g["grid"] == "coe_x_g"].groupby(["valuation", "bank"]):
        r = v.loc[(val, bank), "coe"]
        assert np.isclose(x["coe"].min(), r - 0.02) and np.isclose(x["coe"].max(), r + 0.02)
        assert len(x) == 81
    assert {"coe_x_g", "coe_x_roe_star", "nim_x_cor"} == set(g["grid"])


def test_historical_run_is_point_in_time(root):
    p = pd.read_csv(path("tables", root) / "ch09_pit_verification.csv")
    assert p["eligible"].isin(["yes", "PASS"]).all()
    assert (p["input"].str.startswith("test:")).sum() == 4


def test_implied_and_rerating(root):
    imp = pd.read_csv(path("tables", root) / "ch10_implied.csv")
    assert imp["additional_assumption"].str.len().gt(50).all()  # D12: every conclusion states its assumption
    s = pd.read_csv(path("tables", root) / "ch10_iso_surface.csv")
    assert set(s["valuation"]) == {"HIST", "CURR"} and len(s) == 4 * 20 * 41
    r = pd.read_csv(path("tables", root) / "ch10_rerating.csv")
    tot = r["earnings_and_time_log"] + r["discount_rate_log"] + r["residual_log"]
    assert np.allclose(tot, r["market_total_return_log"])
    mv = np.log((r["market_cap_curr"] + r["dividends_paid"]) / r["market_cap_hist"])
    assert np.allclose(mv, r["market_total_return_log"])
    b = pd.read_csv(path("tables", root) / "ch10_bridge.csv")
    assert (b.loc[b["step"].str.startswith("residual"), "tzs"].abs() < 1).all()  # sequential bridge, reported residual
    h = pd.read_csv(path("tables", root) / "ch10_hypotheses.csv")
    assert h["hypothesis"].iloc[0] == "H10.1" and h["verdict"].iloc[0] in {"supported", "rejected", "inconclusive"}


def test_implied_coe_reproduces_the_price(root):
    from tzbank.banks.engine import project
    from tzbank.valuation import models as vm
    from tzbank.valuation.context import build_ctx

    ctx = build_ctx(root)
    c = ctx.cases[("CURR", "B02", "base")]
    ic = vm.implied_coe(c.op, c.dr, c.price * c.shares, c.vd, lag_months=c.lag_months)
    assert vm.value_at(project(c.op, c.dr), ic, c.vd, "rim", lag_months=c.lag_months) == pytest.approx(c.price * c.shares, rel=1e-6)


def test_scenarios_monte_carlo_and_tested_call(root):
    sc = pd.read_csv(path("tables", root) / "ch11_scenarios.csv")
    for _, g in sc.groupby(["valuation", "bank"]):
        g = g.set_index("scenario")["value_per_share"]
        assert g["bear"] < g["base"] < g["bull"]
    mc = pd.read_csv(path("tables", root) / "ch11_monte_carlo.csv")
    assert (mc["draws"] == 10_000).all()
    assert (mc["p5"] <= mc["p50"]).all() and (mc["p50"] <= mc["p95"]).all()
    assert mc["prob_value_above_price"].between(0, 1).all()
    assert set(mc["variant"]) == {"base-centred (primary)", "bull-centred (sensitivity)"}
    bull = mc[mc["variant"].str.startswith("bull")]
    assert (bull["er_shift"] == 0).all() and len(bull) == 4  # unshifted sensitivity kept
    t = pd.read_csv(path("tables", root) / "ch11_tested_call.csv")
    for r in t.itertuples():
        assert r.actual_total_return == pytest.approx(
            (r.actual_price_31oct2026_same_basis + r.dividend_paid_per_share) / r.hist_price - 1)
    nmb = t.set_index("bank").loc["NMB"]
    assert nmb["actual_price_31oct2026_same_basis"] == pytest.approx(1820)  # post-split basis (owner review D)
    assert nmb["hist_price"] == pytest.approx(841) and nmb["basis"].startswith("post-split")
    assert (mc["persistence_replicates_kept"] > 0.5).all()


def test_ordering_rule_and_amended_h91(root):
    o = pd.read_csv(path("outputs", root) / "valuation" / "ordering_check.csv")
    assert not (o["status"] == "FAIL").any()
    ex = o[o["status"].astype(str).str.startswith("EXEMPT")]
    assert set(ex["symbol"]) <= {"g_L+g_D", "g", "pi_max"}  # only the policy drivers may be exempt (D56)
    sc = o[o["check"].astype(str).str.startswith("scenario")]
    assert len(sc) == 4 and (sc["status"] == "PASS").all()
    r = pd.read_csv(root / "config" / "assumption_register.csv")
    er = r[r["symbol"] == "ER_star"]
    assert (er["bear"] <= er["base"].clip(upper=0) + 1e-12).all() and (er["bear"] <= er["base"] + 1e-12).all()
    h = pd.read_csv(path("tables", root) / "ch09_hypotheses.csv")
    assert "P10-P90" in h["rule"].iloc[0]


def test_monte_carlo_reproducible():
    from tzbank.banks.engine import Drivers, Opening
    from tzbank.valuation import uncertainty as un

    op = Opening(2024, 100.0, 120.0, 40.0, 0.0, 20.0, 0.0, 140.0)
    dr = Drivers(g_L=[0.1, 0.1], g_D=[0.1, 0.1], NIM=[0.05] * 2, phi=[0.01] * 2, CIR=[0.5] * 2, CoR=[0.01] * 2,
                 rho=[0.5] * 2, rate=[0.1] * 2, rate0=0.1, omega=0.0, theta=0.0, l_min=0.2, s_W=0.02, tax=0.3, nu=0.0,
                 delta=0.0, k_star=0.1, pi_max=0.5, roe_star=0.15, g=0.05, payout_star=2 / 3, lam=0.5, fade_years=2)
    c = un.Case("B02", "HIST", pd.Timestamp("2024-12-31"), op, dr, 0.12, 10.0, 2.0, {"rim": 0.6, "ddm": 0.4})
    b = {k: (-0.01, 0.01) for k in un.MC_VARS}
    a1 = un.monte_carlo(c, b, {"NIM_rate": 0.6, "CoR_growth": -0.4}, 50, [1, 2])
    a2 = un.monte_carlo(c, b, {"NIM_rate": 0.6, "CoR_growth": -0.4}, 50, [1, 2])
    pd.testing.assert_frame_equal(a1, a2)
    assert np.corrcoef(a1["NIM"], a1["rate"])[0, 1] > 0.2


def test_coe_minus_g_guard_holds_and_stops(root):
    """D64 amended: COE - terminal g >= 4 pp in every scenario; a violation stops the run."""
    from tzbank.valuation import register as rg
    from tzbank.valuation.approve import load_register
    from tzbank.valuation.run import CoeGrowthGuardError, case

    g = pd.read_csv(path("tables", root) / "ch09_coe_g_guard.csv")
    assert len(g) == 12 and (g["coe_minus_g"] >= 0.04 - 1e-12).all() and (g["status"] == "PASS").all()
    reg = load_register(root)
    _, inputs = rg.build(root)
    bad = reg.copy()
    m = (bad["valuation"] == "HIST") & (bad["bank"] == "NMB") & (bad["symbol"] == "g")
    bad.loc[m, ["base", "bear", "bull"]] = 0.13
    with pytest.raises(CoeGrowthGuardError):
        case(bad, inputs, "HIST", "B02", "base", root)


def test_terminal_g_rule(root):
    """D64 amended: terminal g = IMF final-year Tanzanian inflation + 3%; bear / bull -/+ 1 pp; fade from nominal GDP."""
    reg = pd.read_csv(root / "config" / "assumption_register.csv")
    mac = pd.read_csv(path("processed", root) / "macro.csv", low_memory=False)
    for val, vd in (("HIST", "2025-12-31"), ("CURR", "2026-10-31")):
        m = mac[mac["As At (Valuation Date)"].astype(str).str[:10] == vd].set_index("canonical_name")["value_numeric"]
        x = reg[(reg["valuation"] == val) & (reg["symbol"] == "g")]
        assert np.allclose(x["base"], m["macro_tanzania_inflation_forecast_medium_term"] + 0.03)
        assert np.allclose(x["bear"], x["base"] - 0.01) and np.allclose(x["bull"], x["base"] + 0.01)
        f = reg[(reg["valuation"] == val) & (reg["symbol"] == "g_fade")]
        assert np.allclose(f["base"], m["macro_tanzania_nominal_gdp_growth_forecast"])


def test_h91_reading_per_bank_and_date(root):
    """Pre-flight 12a A: every bank and date is classified by price vs the MC P10-P90 range (D57 verdict at 31 Dec
    2025; 31 Oct 2026 reported as a supplementary reading)."""
    mc = pd.read_csv(path("tables", root) / "ch11_monte_carlo.csv")
    mc = mc[mc["variant"].str.startswith("base")]
    h = pd.read_csv(path("tables", root) / "ch09_hypotheses.csv").iloc[0]
    lab = {"HIST": "31 Dec 2025", "CURR": "31 Oct 2026"}
    for r in mc.itertuples():
        out = not r.p10 <= r.price <= r.p90
        word = ("supported" if out else "rejected") if r.valuation == "HIST" else (
            "material difference" if out else "no material difference")
        assert f"{r.bank} {lab[r.valuation]} {word}" in h["verdict_by_bank_date"]
    hist = mc[mc["valuation"] == "HIST"]
    outs = [not (r.p10 <= r.price <= r.p90) for r in hist.itertuples()]
    assert h["verdict"] == ("supported" if all(outs) else "rejected" if not any(outs) else "inconclusive")
