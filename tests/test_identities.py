"""Accounting identities on a hand-built fixture (CLAUDE.md section 4; prompt 01 section 12)."""

import pandas as pd

from tzbank.validate.identities import run_identities

TOL = {"nii": 0.005, "pbt_tax": 0.005, "profit_split": 0.005, "equity_split": 0.005, "eps": 0.02, "loans": 0.02,
       "tier1_ratio_pp": 0.002}


def _status(checks: pd.DataFrame, year: int) -> dict[str, str]:
    c = checks[(checks["vintage"] == "original") & (checks["period_end"].dt.year == year)]
    return dict(zip(c["identity"], c["status"], strict=True))


def test_all_identities_pass_on_consistent_year(identity_fixture):
    s = _status(run_identities(identity_fixture, TOL), 2023)
    assert len(s) == 7
    assert set(s.values()) == {"PASS"}


def test_each_identity_fails_on_inconsistent_year(identity_fixture):
    s = _status(run_identities(identity_fixture, TOL), 2024)
    assert s["interest income - interest expense = NII"] == "FAIL"
    assert s["PBT - tax = net profit"] == "FAIL"
    assert s["attributable profit + NCI profit = net profit"] == "FAIL"
    assert s["attributable equity + NCI = total equity"] == "FAIL"
    assert s["EPS x weighted shares = attributable profit"] == "FAIL"
    assert s["gross loans - allowance = net loans"] == "FAIL"
    assert s["Tier 1 / RWA = Tier 1 ratio"] == "FAIL"


def test_values_are_hand_checked(identity_fixture):
    c = run_identities(identity_fixture, TOL)
    r = c[(c["identity"] == "interest income - interest expense = NII") & (c["period_end"].dt.year == 2024)
          & (c["vintage"] == "original")].iloc[0]
    assert r["lhs"] == 700.0 and r["rhs"] == 690.0
    assert abs(r["relative_difference"] - 10 / 690) < 1e-12


def test_tolerance_boundary():
    """0.4% off passes at 0.5% tolerance; 0.6% off fails."""
    from tests.conftest import canon_row

    rows = []
    for year, nii in [(2020, 700 / 1.004), (2021, 700 / 1.006)]:
        for k, v in {"interest_income": 1000.0, "interest_expense": 300.0, "net_interest_income": nii}.items():
            rows.append(canon_row(canonical_name=k, value=v, period_end=pd.Timestamp(f"{year}-12-31"),
                                  source_row_id=f"{year}{k}"))
    c = run_identities(pd.DataFrame(rows), TOL, vintages=("original",))
    assert c.set_index(c["period_end"].dt.year)["status"].to_dict() == {2020: "PASS", 2021: "FAIL"}


def test_bank_level_gross_loans_flagged_not_failed():
    from tests.conftest import canon_row

    rows = [canon_row(canonical_name="gross_loans", value=1000.0, notes="Group basis. Bank-level note (...)"),
            canon_row(canonical_name="loan_loss_allowance", value=50.0, source_row_id="b"),
            canon_row(canonical_name="net_loans", value=900.0, source_row_id="c")]
    c = run_identities(pd.DataFrame(rows), TOL, vintages=("original",))
    assert c.iloc[0]["status"] == "FLAG"


def test_restated_rows_do_not_mix_with_originals():
    """A restated NII in the original vintage would break the identity; it must be excluded."""
    from tests.conftest import canon_row

    rows = [canon_row(canonical_name="interest_income", value=1000.0, source_row_id="a"),
            canon_row(canonical_name="interest_expense", value=300.0, source_row_id="b"),
            canon_row(canonical_name="net_interest_income", value=700.0, source_row_id="c",
                      forecast_base_current=False),
            canon_row(canonical_name="net_interest_income", value=650.0, is_restated=True, source_row_id="d",
                      publication_date=pd.Timestamp("2026-03-01"))]
    c = run_identities(pd.DataFrame(rows), TOL, vintages=("original",))
    assert c.iloc[0]["status"] == "PASS"
