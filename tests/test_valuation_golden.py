"""Prompt 09a golden test: a hand-built small bank with RIM, DDM and the target-price identity worked out by hand
(VALUATION_SPEC §4, §6, §10).

Bank at 31 Dec 2024: net loans 100, deposits 120, liquid assets 40, no wholesale funding, equity 20, no NCI, total
assets 140 (theta = omega = 0). Drivers (both years): g_L = g_D = 10%, l_min = 20%, NIM 5%, phi 1%, CIR 50%, CoR 1%,
tax 30%, nu 0, rho 50%, delta 0, k* 10%, payout cap 50%, rate path flat. Two explicit years, no fade, then terminal with
ROE* 15%, g 5%, payout* = 1 - g / ROE* = 2/3; r = 12%.

Hand derivation, year 1 (funding surplus, so W = 0, TA_1 = D_1 + E_1 = 132 + E_1, LA_1 = TA_1 - L_1 = 22 + E_1):
  avg IEA = avg TA = (140 + 132 + E_1) / 2;  revenue = (NIM + phi) avg TA = 0.03 (272 + E_1)
  OpEx = 0.5 revenue = 0.015 (272 + E_1);  Imp = 1% x (100 + 110) / 2 = 1.05
  NP_1 = 0.7 (0.015 (272 + E_1) - 1.05) = 2.121 + 0.0105 E_1
  the payout cap binds (capital room is large), so Div_1 = 0.5 NP_1 and E_1 = 20 + 0.5 NP_1
  => E_1 = (20 + 0.5 x 2.121) / (1 - 0.5 x 0.0105)
Year 2 (L_2 = 121, D_2 = 145.2, TA_2 = 145.2 + E_2):
  NP_2 = 0.7 (0.015 (277.2 + E_1 + E_2) - 1.155) = 0.0105 (277.2 + E_1 + E_2) - 0.8085
  E_2 = E_1 + 0.5 NP_2 => E_2 = (E_1 (1 + 0.00525) + 0.00525 x 277.2 - 0.40425) / (1 - 0.00525)
RIM_0 = 20 + (NP_1 - 0.12 x 20)/1.12 + (NP_2 - 0.12 E_1)/1.12^2 + (0.15 - 0.12) E_2 / (0.07 x 1.12^2)
DDM_0 = Div_1/1.12 + Div_2/1.12^2 + (0.15 - 0.05) E_2 / (0.07 x 1.12^2)
Identity: value at 31 Dec 2025 (cum Div_1) = V_0 x 1.12.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tzbank.banks.engine import Drivers, EngineCheckError, Opening, project, require
from tzbank.valuation import coe, models

R = 0.12


def _case(**kw):
    op = Opening(year=2024, loans=100.0, deposits=120.0, liquid=40.0, wholesale=0.0, equity_attr=20.0,
                 nci_equity=0.0, total_assets=140.0)
    base = dict(g_L=[0.10, 0.10], g_D=[0.10, 0.10], NIM=[0.05, 0.05], phi=[0.01, 0.01], CIR=[0.5, 0.5],
                CoR=[0.01, 0.01], rho=[0.5, 0.5], rate=[0.1, 0.1], rate0=0.1, omega=0.0, theta=0.0, l_min=0.2,
                s_W=0.02, tax=0.3, nu=0.0, delta=0.0, k_star=0.10, pi_max=0.5, roe_star=0.15, g=0.05,
                payout_star=1 - 0.05 / 0.15, lam=0.5, fade_years=0, fade_shape="linear")
    base.update(kw)
    return op, Drivers(**base)


def _hand():
    e1 = (20 + 0.5 * 2.121) / (1 - 0.5 * 0.0105)
    np1 = 2.121 + 0.0105 * e1
    e2 = (e1 * (1 + 0.00525) + 0.00525 * 277.2 - 0.40425) / (1 - 0.00525)
    np2 = 0.0105 * (277.2 + e1 + e2) - 0.8085
    rim = 20 + (np1 - R * 20) / 1.12 + (np2 - R * e1) / 1.12 ** 2 + (0.15 - R) * e2 / (0.07 * 1.12 ** 2)
    ddm = 0.5 * np1 / 1.12 + 0.5 * np2 / 1.12 ** 2 + (0.15 - 0.05) * e2 / (0.07 * 1.12 ** 2)
    return e1, e2, np1, np2, rim, ddm


def test_golden_forecast_matches_hand_values():
    op, dr = _case()
    fc = project(op, dr)
    d = fc.rows.set_index("t")
    e1, e2, np1, np2, _, _ = _hand()
    assert d.loc[1, "E"] == pytest.approx(e1, rel=1e-9) and d.loc[2, "E"] == pytest.approx(e2, rel=1e-9)
    assert d.loc[1, "NP_attr"] == pytest.approx(np1, rel=1e-9) and d.loc[2, "NP_attr"] == pytest.approx(np2, rel=1e-9)
    assert d.loc[1, "Div"] == pytest.approx(0.5 * np1) and d.loc[1, "W"] == 0
    assert d.loc[1, "TA"] == pytest.approx(132 + e1) and d.loc[1, "LA"] == pytest.approx(22 + e1)
    require(fc.checks)  # every engine check passes


def test_golden_rim_ddm_and_identity():
    op, dr = _case()
    fc = project(op, dr)
    *_, rim, ddm = _hand()
    assert models.v_ex(fc, R, 0, "rim") == pytest.approx(rim, rel=1e-9)
    assert models.v_ex(fc, R, 0, "ddm") == pytest.approx(ddm, rel=1e-9)
    assert rim == pytest.approx(ddm, rel=1e-9)  # clean surplus + consistent terminal: the two models agree
    v0 = models.value_at(fc, R, pd.Timestamp("2024-12-31"))
    v1 = models.value_at(fc, R, pd.Timestamp("2025-12-31"))
    assert v0 == pytest.approx(rim) and v1 == pytest.approx(rim * 1.12, rel=1e-9)  # cum Div_1 at 31 Dec 2025
    ti = models.target_identity(fc, R, pd.Timestamp("2024-12-31"))
    assert abs(ti["residual"]) < 1e-9
    # mid-year valuation date: V1 = V0 (1 + r) - Div_1 carried from its year-end to the target date
    ti = models.target_identity(fc, R, pd.Timestamp("2025-10-31"))
    assert abs(ti["residual"]) < 1e-9


def test_dividend_pay_lag_both_models():
    """Owner issue 17, D62 amended: dividends paid L months after year-end in RIM and DDM; the held dividend earns the
    liquid-asset yield (golden rate path flat at 10%) until paid. By hand on the golden bank."""
    op, dr = _case()
    fc = project(op, dr)
    *_, rim, ddm = _hand()
    lag = models._years(pd.Timestamp("2024-12-31"), pd.Timestamp("2025-06-30"))  # 6 months
    phi = (1.10 / 1.12) ** lag
    # L = 0 reproduces the year-end convention exactly
    assert models.v_ex(fc, R, 0, "ddm", 0) == pytest.approx(ddm, rel=1e-9)
    assert models.v_ex(fc, R, 0, "rim", 0) == pytest.approx(rim, rel=1e-9)
    # with a flat yield every dividend (terminal stream included) is scaled by phi; RIM = DDM (gap back to zero)
    assert models.v_ex(fc, R, 0, "ddm", 6) == pytest.approx(ddm * phi, rel=1e-9)
    assert models.v_ex(fc, R, 0, "rim", 6) == pytest.approx(ddm * phi, rel=1e-9)
    for k in (0, 1):
        assert models.v_ex(fc, R, k, "rim", 6) == pytest.approx(models.v_ex(fc, R, k, "ddm", 6), rel=1e-12)
    # 31 Mar 2026: Div_1 (FY2025) declared, paid 30 Jun 2026: in the value with its held yield, discounted to the date
    d = pd.Timestamp("2026-03-31")
    f = models._years(pd.Timestamp("2025-12-31"), d)
    div1 = float(fc.rows.set_index("t").loc[1, "Div"])
    held = div1 * 1.10 ** lag
    for m in ("rim", "ddm"):
        hand = models.v_ex(fc, R, 1, m, 6) * 1.12 ** f + held / 1.12 ** models._years(d, pd.Timestamp("2026-06-30"))
        assert models.value_at(fc, R, d, m, lag_months=6) == pytest.approx(hand, rel=1e-12)
        # identity with each dividend (and its held yield) carried from its payment date
        for vd in ("2024-12-31", "2025-03-31", "2025-06-30", "2025-10-31"):
            ti = models.target_identity(fc, R, pd.Timestamp(vd), m, lag_months=6)
            assert abs(ti["residual"]) < 1e-9, (m, vd)
        a = models.value_at(fc, R, pd.Timestamp("2026-06-30"), m, lag_months=6)
        b = models.value_at(fc, R, pd.Timestamp("2026-07-01"), m, lag_months=6)
        assert b == pytest.approx((a - held) * 1.12 ** (1 / 365.25), rel=1e-9)  # leaves the day after payment
    bl = models.blend(fc, R, pd.Timestamp("2024-12-31"), {"rim": 0.6, "ddm": 0.4}, lag_months=6)
    assert bl["gap"] == pytest.approx(0.0, abs=1e-9)


def test_golden_wholesale_branch_balances():
    op, dr = _case(g_D=[0.0, 0.0], g_L=[0.30, 0.30])  # loans outrun deposits: wholesale funding fills the gap
    fc = project(op, dr)
    d = fc.rows.set_index("t")
    assert d.loc[2, "W"] > 0 and d.loc[2, "LA"] == pytest.approx(0.2 * 120)
    assert d.loc[2, "TA"] == pytest.approx(d.loc[2, "D"] + d.loc[2, "E"] + d.loc[2, "W"])
    assert all(c["status"] == "PASS" for c in fc.checks if c["check"].startswith(("balance", "asset", "clean")))


def test_capital_breach_stops_with_named_error():
    op, dr = _case(k_star=0.30)  # impossible target: the rule cannot hold Tier 1 >= k* x RWA
    fc = project(op, dr)
    with pytest.raises(EngineCheckError, match="Tier 1"):
        require(fc.checks)


def test_fade_starts_at_g_fade_start():
    """D64 amended: with g_fade_start set, fade growth runs linearly from it to terminal g."""
    op, dr = _case(fade_years=10, fade_shape="lambda", lam=0.6, g_fade_start=0.09)
    f = project(op, dr).rows.query("phase == 'fade'")
    assert f["g_E"].iloc[0] == pytest.approx(0.09 + (0.05 - 0.09) / 10)
    assert f["g_E"].iloc[-1] == pytest.approx(0.05)


def test_lambda_fade_and_terminal_consistency():
    op, dr = _case(fade_years=10, fade_shape="lambda", lam=0.6)
    fc = project(op, dr)
    f = fc.rows[fc.rows["phase"] == "fade"]
    roe_T = fc.rows.set_index("t").loc[2, "ROE"]
    assert f["ROE"].iloc[0] == pytest.approx(0.15 + 0.6 * (roe_T - 0.15))
    assert f["ROE"].iloc[-1] == pytest.approx(0.15 + 0.6 ** 10 * (roe_T - 0.15))
    assert models.v_ex(fc, R, 0, "rim") == pytest.approx(models.v_ex(fc, R, 0, "ddm"), rel=1e-9)
    bad = [c for c in fc.checks if c["check"].startswith("terminal")]
    assert bad[0]["status"] == "PASS"


def test_coe_approaches_and_reconciliation():
    a = coe.approach_a(0.12, 0.04, 0.6, 0.04, 0.06, 0.015)
    assert a == pytest.approx(0.08 + 0.06 + 0.015)
    b = coe.approach_b(0.04, 0.6, 0.04, 0.06, 0.015, 0.05, 0.02)
    assert b == pytest.approx((1 + 0.04 + 0.06 + 0.015) * 1.05 / 1.02 - 1)
    assert coe.reconcile(0.155, 0.150).central == pytest.approx(0.1525)  # < 150 bp: midpoint
    assert coe.reconcile(0.155, 0.130).central == pytest.approx(0.155)  # >= 150 bp: A pending the register
    assert math.isnan(coe.approach_b(0.04, 0.6, 0.04, 0.06, 0.015, None, 0.02))
    assert coe.reconcile(0.155, float("nan")).central == pytest.approx(0.155)


def test_justified_multiples_and_implied_coe():
    j = models.justified(0.15, 0.05, 0.12, 2 / 3)
    assert j["pb"] == pytest.approx(0.10 / 0.07) and j["pe"] == pytest.approx((2 / 3) * 1.05 / 0.07)
    op, dr = _case()
    fc = project(op, dr)
    price = models.value_at(fc, R, pd.Timestamp("2024-12-31"))
    assert models.implied_coe(op, dr, price, pd.Timestamp("2024-12-31")) == pytest.approx(R, abs=1e-6)
    assert models.implied_roe_star(op, dr, R, price, pd.Timestamp("2024-12-31")) == pytest.approx(0.15, abs=1e-6)
