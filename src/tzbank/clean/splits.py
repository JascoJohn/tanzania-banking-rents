"""Split adjustment from config/corporate_actions.csv (CLAUDE.md section 3.4; prompt 01 section 4.4).

A ratio-r split effective on date E means each share became r shares. For an observation dated before E,
adjusted price = price / r and adjusted share count = count x r. Raw columns are always kept.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path

PER_SHARE_PRICE_VARS = {"share_price_close", "eps_basic", "eps_cumulative_interim", "dps", "dps_special", "dps_total", "mkt_52_week_high",
                        "mkt_52_week_low", "mkt_dividend_per_share_dps"}
SHARE_COUNT_VARS = {"shares_outstanding", "weighted_avg_shares", "treasury_shares", "mkt_12_month_traded_volume"}


def load_corporate_actions(root: Path | None = None) -> pd.DataFrame:
    """Read config/corporate_actions.csv."""
    df = pd.read_csv(path("corporate_actions", root or find_root()), dtype={"bank_id": str})
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df["ratio"] = df["ratio"].astype(float)
    return df[df["action"].str.lower() == "split"]


def split_factor(dates: pd.Series, bank_ids: pd.Series, actions: pd.DataFrame) -> pd.Series:
    """Cumulative split factor for each (date, bank): product of ratios of splits effective after the date."""
    dates = pd.to_datetime(dates)
    factor = pd.Series(1.0, index=dates.index)
    for _, a in actions.iterrows():
        hit = (bank_ids.astype(str) == a["bank_id"]) & (dates < a["effective_date"])
        factor = factor.where(~hit, factor * a["ratio"])
    return factor


def apply_split_adjustment(df: pd.DataFrame, actions: pd.DataFrame, date_col: str, bank_col: str,
                           price_cols: dict[str, str], volume_cols: dict[str, str]) -> pd.DataFrame:
    """Add ``split_factor`` and adjusted columns; prices divided, volumes multiplied by the factor."""
    out = df.copy()
    out["split_factor"] = split_factor(out[date_col], out[bank_col], actions)
    for src, dst in price_cols.items():
        out[dst] = pd.to_numeric(out[src], errors="coerce") / out["split_factor"]
    for src, dst in volume_cols.items():
        out[dst] = pd.to_numeric(out[src], errors="coerce") * out["split_factor"]
    return out


def adjust_long_values(df: pd.DataFrame, actions: pd.DataFrame, var_col: str = "canonical_name",
                       date_col: str = "period_end", bank_col: str = "bank_id") -> pd.Series:
    """Split-adjusted value for per-share and share-count variables in a long table; NaN for other variables."""
    f = split_factor(df[date_col], df[bank_col], actions)
    v = pd.to_numeric(df["value"], errors="coerce")
    adj = pd.Series(float("nan"), index=df.index)
    price = df[var_col].isin(PER_SHARE_PRICE_VARS)
    count = df[var_col].isin(SHARE_COUNT_VARS)
    adj[price] = v[price] / f[price]
    adj[count] = v[count] * f[count]
    return adj
