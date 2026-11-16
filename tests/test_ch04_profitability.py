"""Chapter 4 starter formulas against hand-checked values (CLAUDE.md section 4: every financial formula)."""

import math

import pandas as pd
import pytest

from tzbank.chapters.ch04.profitability import dupont_panel


@pytest.fixture
def wide() -> pd.DataFrame:
    return pd.DataFrame([
        # bank with NCI split and gross loans
        dict(bank_id="B01", fiscal_year=2023, profit_attributable=90.0, net_profit=100.0, equity_attributable=400.0,
             total_equity=420.0, total_assets=4000.0, net_interest_income=300.0, non_interest_income=100.0,
             operating_expenses=200.0, impairment_charges=40.0, gross_loans=2000.0, net_loans=1900.0),
        dict(bank_id="B01", fiscal_year=2024, profit_attributable=110.0, net_profit=120.0, equity_attributable=500.0,
             total_equity=520.0, total_assets=6000.0, net_interest_income=360.0, non_interest_income=140.0,
             operating_expenses=250.0, impairment_charges=30.0, gross_loans=3000.0, net_loans=2850.0),
        # bank with no NCI split / gross loans; 2022 missing -> 2023 has no averages
        dict(bank_id="B03", fiscal_year=2021, net_profit=5.0, total_equity=50.0, total_assets=500.0,
             net_interest_income=30.0, non_interest_income=10.0, operating_expenses=30.0, impairment_charges=4.0,
             net_loans=300.0),
        dict(bank_id="B03", fiscal_year=2023, net_profit=6.0, total_equity=60.0, total_assets=600.0,
             net_interest_income=32.0, non_interest_income=8.0, operating_expenses=36.0, impairment_charges=5.0,
             net_loans=320.0),
        dict(bank_id="B03", fiscal_year=2024, net_profit=8.0, total_equity=70.0, total_assets=700.0,
             net_interest_income=40.0, non_interest_income=10.0, operating_expenses=40.0, impairment_charges=6.0,
             net_loans=360.0),
    ])


def _row(p, b, y):
    return p[(p.bank_id == b) & (p.fiscal_year == y)].iloc[0]


def test_hand_checked_b01_2024(wide):
    r = _row(dupont_panel(wide), "B01", 2024)
    assert math.isclose(r.roe, 110 / 450)                 # avg attributable equity (400+500)/2
    assert math.isclose(r.roa, 120 / 5000)                # avg assets (4000+6000)/2
    assert math.isclose(r.nii_to_avg_assets, 360 / 5000)
    assert math.isclose(r.non_interest_income_share, 140 / 500)
    assert math.isclose(r.cost_to_income, 250 / 500)
    assert math.isclose(r.cost_of_risk, 30 / 2500)        # avg gross loans (2000+3000)/2
    assert math.isclose(r.leverage, 5000 / 450)
    assert r.roe_basis.startswith("attributable")
    assert r.coverage_flag == "complete"
    assert r.evidence_class == "Derived"


def test_first_year_has_no_average_ratios(wide):
    r = _row(dupont_panel(wide), "B01", 2023)
    assert pd.isna(r.roe) and pd.isna(r.roa) and pd.isna(r.cost_of_risk)
    assert math.isclose(r.cost_to_income, 200 / 400)      # needs no average
    assert r.coverage_flag.startswith("partial")


def test_gap_year_breaks_average(wide):
    """2023 follows 2021 (2022 missing): no averages; 2024 averages 2023-2024."""
    p = dupont_panel(wide)
    assert pd.isna(_row(p, "B03", 2023).roe)
    r = _row(p, "B03", 2024)
    assert math.isclose(r.roe, 8 / 65)                    # fallback: net profit / avg total equity
    assert r.roe_basis.startswith("net profit")
    assert pd.isna(r.cost_of_risk)                        # no gross loans: never substituted
    assert math.isclose(r.cost_of_risk_net_loans_proxy, 6 / 340)
    assert math.isclose(r.leverage, 650 / 65)


def test_dupont_identity_holds(wide):
    """ROE = ROA x leverage x (attributable / net profit) on the fixture."""
    r = _row(dupont_panel(wide), "B01", 2024)
    assert math.isclose(r.roe, r.roa * r.leverage * (110 / 120))
