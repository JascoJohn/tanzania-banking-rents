"""Point-in-time filter (both valuation dates) and split adjustment (CLAUDE.md section 4)."""

import warnings

import pandas as pd
import pytest

from tzbank.clean.normalise import forecast_base
from tzbank.clean.splits import adjust_long_values, apply_split_adjustment, load_corporate_actions, split_factor
from tzbank.config import valuation_dates
from tzbank.pit import MissingPublicationDateWarning, eligible

HIST, CURR = pd.Timestamp("2025-12-31"), pd.Timestamp("2026-10-31")


@pytest.fixture
def pit_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["fy2024", "fy2025_pub2026", "h1_2026", "9m_2026_late", "no_pub", "fy2025_restated_2026"],
        "period_end": pd.to_datetime(["2024-12-31", "2025-12-31", "2026-06-30", "2026-09-30", "2024-12-31",
                                      "2025-12-31"]),
        "publication_date": pd.to_datetime(["2025-03-27", "2026-05-08", "2026-07-29", "2026-11-02", None,
                                            "2026-10-30"]),
    })


def test_valuation_dates_from_config():
    vd = valuation_dates()
    assert vd["historical"] == HIST and vd["current"] == CURR


def test_eligible_historical(pit_frame):
    with pytest.warns(MissingPublicationDateWarning, match="1 of 6"):
        out = eligible(pit_frame, HIST)
    assert out["id"].tolist() == ["fy2024"]


def test_eligible_current_includes_fy2025_published_2026(pit_frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingPublicationDateWarning)
        out = eligible(pit_frame, CURR)
    assert out["id"].tolist() == ["fy2024", "fy2025_pub2026", "h1_2026", "fy2025_restated_2026"]


def test_period_after_date_excluded_even_if_published():
    df = pd.DataFrame({"period_end": pd.to_datetime(["2026-03-31"]), "publication_date": pd.to_datetime(["2025-01-01"])})
    assert eligible(df, HIST, warn=False).empty


def test_forecast_base_is_latest_eligible_restatement():
    base = dict(bank_id="B01", entity="Group", basis="GROUP", period_type="FY", canonical_name="net_profit",
                category=None, value=1.0)
    df = pd.DataFrame([
        dict(base, period_end=pd.Timestamp("2024-12-31"), publication_date=pd.Timestamp("2025-03-27"),
             source_row_id="orig"),
        dict(base, period_end=pd.Timestamp("2024-12-31"), publication_date=pd.Timestamp("2026-05-08"),
             source_row_id="restated", value=2.0),
    ])
    assert forecast_base(df, HIST).tolist() == [True, False]
    assert forecast_base(df, CURR).tolist() == [False, True]


def test_corporate_actions_has_nmb_split():
    a = load_corporate_actions()
    row = a[a["bank_id"] == "B02"].iloc[0]
    assert row["ratio"] == 10 and row["effective_date"] == pd.Timestamp("2026-08-24")


def test_split_factor_before_on_after_effective_date():
    a = load_corporate_actions()
    d = pd.Series(pd.to_datetime(["2026-08-21", "2026-08-24", "2026-08-25", "2026-08-21"]))
    b = pd.Series(["B02", "B02", "B02", "B01"])
    assert split_factor(d, b, a).tolist() == [10.0, 1.0, 1.0, 1.0]


def test_apply_split_adjustment_prices_and_volumes():
    a = load_corporate_actions()
    df = pd.DataFrame({"date": pd.to_datetime(["2026-08-21", "2026-08-25"]), "bank": ["B02", "B02"],
                       "close": [18000.0, 1800.0], "vol": [1000.0, 10000.0]})
    out = apply_split_adjustment(df, a, "date", "bank", {"close": "adj_close"}, {"vol": "adj_vol"})
    assert out["adj_close"].tolist() == [1800.0, 1800.0]
    assert out["adj_vol"].tolist() == [10000.0, 10000.0]
    assert out["close"].tolist() == [18000.0, 1800.0]  # raw kept


def test_adjust_long_values_per_share_only():
    a = load_corporate_actions()
    df = pd.DataFrame({"bank_id": ["B02", "B02", "B02", "B01"],
                       "canonical_name": ["eps_basic", "shares_outstanding", "net_profit", "eps_basic"],
                       "period_end": pd.to_datetime(["2025-12-31"] * 4), "value": [1290.0, 5e8, 7e11, 250.0]})
    adj = adjust_long_values(df, a)
    assert adj.iloc[0] == 129.0 and adj.iloc[1] == 5e9
    assert pd.isna(adj.iloc[2])
    assert adj.iloc[3] == 250.0


def test_daily_market_split_matches_workbook(root):
    """Python-recomputed split factor equals the factor typed in the workbook on every row."""
    from tzbank.ingest.inputs_workbook import load_interim

    d = load_interim("Daily Market Data", root)
    assert (d["split_factor"] == d["split_factor_workbook"]).all()
    nmb_pre = d[(d["Bank ID"] == "B02") & (d["Date"] < "2026-08-24")]
    assert (nmb_pre["adj_close"] * 10 == nmb_pre["Close (TZS)"]).all()
