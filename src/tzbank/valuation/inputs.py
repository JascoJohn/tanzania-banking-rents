"""Point-in-time valuation inputs (VALUATION_SPEC §2 cut-off policy, §9; CLAUDE.md §3.3; decision D1, D5).

Every input that enters a valuation passes :func:`tzbank.pit.eligible` for that valuation date: published on or before
the date and for a period ending on or before it. Within the eligible rows the latest publication wins (D5: the
forecasting base uses the latest restatement published by the valuation date). Capital variables follow D2/D17
(NMB Bank-solo; CRDB Group from FY2015); everything else is Group.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.pit import MissingPublicationDateWarning, eligible

CAPITAL_VARS = {"rwa_total", "tier1_capital", "tier1_ratio", "total_capital_ratio", "tier2_capital"}
MACRO = {"y_long": "macro_tzs_long_bond_yield_d34", "ds": "macro_tanzania_sovereign_default_spread",
         "crp": "macro_tanzania_country_risk_premium", "erp": "macro_mature_market_implied_erp",
         "beta": "macro_bank_beta_emerging_markets_levered", "rf_us": "macro_us_10_year_treasury_yield",
         "cpi_tz": "macro_tanzania_headline_cpi_inflation_latest_y_y", "cbr": "macro_bot_central_bank_rate",
         "infl_tz_fwd": "macro_tanzania_inflation_forecast_medium_term",
         "infl_us_fwd": "macro_us_long_run_inflation_expectation", "tzs_usd": "macro_tzs_per_usd",
         "ngdp_tz": "macro_tanzania_nominal_gdp_growth_forecast", "infl_bdi": "macro_burundi_inflation_forecast",
         "infl_cod": "macro_drc_inflation_forecast"}


def basis_for(bank: str, var: str, fy: int) -> str:
    """D2/D17 basis for capital variables; Group otherwise."""
    if var in CAPITAL_VARS:
        return "BANK" if bank == "B02" or (bank == "B01" and fy <= 2014) else "GROUP"
    return "GROUP"


def eligible_long(root: Path, vd: str) -> pd.DataFrame:
    """bank_period_long restricted to rows eligible at ``vd`` (no publication date -> excluded, D1)."""
    lg = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False, dtype={"bank_id": str})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingPublicationDateWarning)
        e = eligible(lg, vd, warn=False)
    return e[e["value"].notna()].copy()


def pick(e: pd.DataFrame, bank: str, var: str, ptype: str, fy: int | None = None,
         period_end: str | None = None, basis: str | None = None) -> tuple[float, str]:
    """Latest-published eligible value for one key; returns (value, publication_date) or (nan, '')."""
    x = e[(e["bank_id"] == bank) & (e["canonical_name"] == var) & (e["period_type"] == ptype)]
    if fy is not None:
        x = x[x["fiscal_year"] == fy]
    if period_end is not None:
        x = x[x["period_end"].astype(str).str[:10] == period_end]
    b = basis or (basis_for(bank, var, fy) if fy is not None else "GROUP")
    xb = x[x["basis"] == b]
    x = xb if len(xb) else x[x["basis"] == "GROUP"] if b == "GROUP" else xb
    if x.empty:
        return float("nan"), ""
    r = x.sort_values("publication_date").iloc[-1]
    return float(r["value"]), str(r["publication_date"])[:10]


@dataclass
class BaseInputs:
    """Opening position and market inputs for one bank at one valuation date (all eligible at that date)."""

    bank: str
    vd: pd.Timestamp
    anchor_fy: int                       # last fiscal year with eligible audited statements
    fy: dict[str, float]                 # anchor-year balances and flows
    fy_prev: dict[str, float]            # anchor-1 balances (averages)
    interim: dict[str, float]            # latest eligible interim in the year after the anchor (YTD)
    interim_months: int
    shares: float                        # ordinary shares outstanding at vd (ex-treasury, split-adjusted as reported)
    price: float
    final_dps_anchor: float              # final dividend per share declared for the anchor year (eligible at vd)
    macro: dict[str, float]
    sources: dict[str, str] = field(default_factory=dict)  # variable -> publication date used (audit trail)


FY_VARS = ["total_assets", "net_loans", "gross_loans", "customer_deposits", "borrowings", "total_equity",
           "equity_attributable", "nci_equity", "net_interest_income", "non_interest_income", "operating_expenses",
           "impairment_charges", "profit_before_tax", "income_tax_expense", "net_profit", "profit_attributable",
           "interest_earning_assets", "cash_and_central_bank", "placements_with_banks", "government_securities",
           "rwa_total", "tier1_capital", "tier1_ratio", "total_capital_ratio", "shares_outstanding", "dps",
           "dps_special", "loan_loss_allowance"]
INTERIM_VARS = ["total_assets", "net_loans", "gross_loans", "customer_deposits", "borrowings", "total_equity",
                "equity_attributable", "net_interest_income", "non_interest_income", "operating_expenses",
                "impairment_charges", "profit_before_tax", "net_profit", "profit_attributable"]


def base_inputs(bank: str, vd: str, root: Path | None = None) -> BaseInputs:
    root = root or find_root()
    d = pd.Timestamp(vd)
    e = eligible_long(root, vd)
    fys = e[(e["bank_id"] == bank) & (e["period_type"] == "FY") & (e["canonical_name"] == "net_profit")]
    anchor = int(fys["fiscal_year"].max())
    src: dict[str, str] = {}

    def year(fy: int) -> dict[str, float]:
        out = {}
        for v in FY_VARS:
            val, pub = pick(e, bank, v, "FY", fy)
            out[v] = val
            if fy == anchor:
                src[f"FY{fy}.{v}"] = pub
        return out

    fy, prev = year(anchor), year(anchor - 1)
    # latest interim (9M preferred, then H1) for the year after the anchor
    interim, months = {}, 0
    for ptype, m, pe in (("9M", 9, f"{anchor + 1}-09-30"), ("H1", 6, f"{anchor + 1}-06-30")):
        val, pub = pick(e, bank, "net_profit", ptype, period_end=pe)
        if np.isfinite(val):
            months = m
            for v in INTERIM_VARS:
                interim[v], p2 = pick(e, bank, v, ptype, period_end=pe)
                src[f"{ptype}{anchor + 1}.{v}"] = p2
            break
    sh, p_sh = pick(e, bank, "shares_outstanding", "AS AT", period_end=d.date().isoformat())
    if not np.isfinite(sh):  # AS AT row not yet public at vd: latest eligible annual report (D1)
        sh, p_sh = fy["shares_outstanding"], src.get(f"FY{anchor}.shares_outstanding", "")
        src["shares_note"] = "AS AT shares not eligible at vd; FY shares used"
    src["shares_outstanding"] = p_sh
    price, p_pr = pick(e, bank, "share_price_close", "AS AT", period_end=d.date().isoformat())
    src["share_price_close"] = p_pr
    fin = e[(e["bank_id"] == bank) & (e["period_type"] == "DIV")
            & e["canonical_name"].str.startswith(f"mkt_dividend_per_share_fy{anchor}_final")]
    final_dps = float(fin.sort_values("publication_date")["value"].iloc[-1]) if len(fin) else float("nan")
    src["final_dps_anchor"] = str(fin["publication_date"].max())[:10] if len(fin) else "not eligible"
    # macro (eligible at vd, the value 'as at' vd)
    m = pd.read_csv(path("processed", root) / "macro.csv")
    m = m[m["As At (Valuation Date)"].astype(str).str[:10] == d.date().isoformat()].copy()
    m["period_end"] = m["As At (Valuation Date)"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingPublicationDateWarning)
        m = eligible(m, vd, warn=False)
    macro = {}
    for k, cn in MACRO.items():
        x = m[m["canonical_name"] == cn].dropna(subset=["value_numeric"])
        macro[k] = float(x["value_numeric"].iloc[-1]) if len(x) else float("nan")
        src[f"macro.{k}"] = str(x["publication_date"].iloc[-1])[:10] if len(x) else "not eligible / not held"
    return BaseInputs(bank, d, anchor, fy, prev, interim, months, sh, price, final_dps, macro, src)


def pit_violations(b: BaseInputs) -> list[str]:
    """Every recorded source date must be on or before the valuation date (engine check)."""
    bad = []
    for k, p in b.sources.items():
        try:
            if p and p[:4].isdigit() and pd.Timestamp(p) > b.vd:
                bad.append(f"{k} published {p}")
        except ValueError:
            continue
    return bad
