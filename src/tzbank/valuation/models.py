"""Valuation models on one forecast (VALUATION_SPEC §6; blueprint D12 for implied persistence).

Year-end convention: the dividend declared for fiscal year t belongs to holders at the end of year t (cum at that date)
and leaves the value afterwards. Anchor k = end of forecast year k (k = 0 is the audited anchor year).

    V_ex(k) RIM = B_k + sum_{t=k+1..N} (NP_attr,t - r B_t-1) / (1 + r)^(t-k) + (ROE* - r) B_N / ((r - g)(1 + r)^(N-k))
    V_ex(k) DDM = sum_{t=k+1..N} Div_t / (1 + r)^(t-k) + (ROE* - g) B_N / ((r - g)(1 + r)^(N-k))
    value at date d (anchor k = last year-end <= d, f = years from anchor k to d):
        V(d) = V_ex(k) (1 + r)^f, plus Div_k when d falls exactly on anchor k (cum-dividend)
    target price V1 = V(vd + 12 months), computed independently from its own anchor; identity (§6, §10):
        V1 = V0 (1 + r) - sum of dividends leaving the value in (vd, vd + 12m], each carried to vd + 12m at r.

Dividend payment lag (owner issue 17, `valuation.dividend_pay_lag_months`; D62 amended: both models). The FY t
dividend is paid L months after year-end t (pay date P_t, l = years(Dec 31, P)). Until paid it stays in the bank and
earns the liquid-asset yield y (the T-bill path of the forecast in the payment year; the last year's rate for the
terminal stream); the dividend and the yield earned on it leave the value at the payment date (the yield is not in
the forecast's profit). Per dividend, phi_t = ((1 + y_t) / (1 + r))^l:
    DDM: each dividend, the terminal stream included, is valued at Div_t phi_t.
    RIM: dividends leave book value at the payment date; between year-end and payment the held dividend earns y and
         bears the capital charge r, a residual income of Div_t ((1 + y)^l - (1 + r)^l) at P_t, i.e. Div_t (phi_t - 1)
         at year-end t; adding it to the year-end RIM gives RIM = DDM exactly (clean surplus), as at L = 0.
    A declared dividend stays in the value until paid: V(d) = V_ex(k) (1 + r)^f + Div_k (1 + y_k)^l (1 + r)^-(years(d,
    P_k)) for d <= P_k. L = 0 reproduces the year-end convention exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tzbank.banks.engine import Drivers, Forecast, Opening, project


def _years(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Year fraction: whole anniversaries plus days / 365.25 (Dec 31 -> Dec 31 = 1 even across a leap year; Dec 31 ->
    Oct 31 = 1 - 61 / 365.25); additive over a fixed month-day grid."""
    ann = pd.Timestamp(b.year, a.month, min(a.day, 28) if a.month == 2 and a.day == 29 else a.day)
    return (b.year - a.year) + (b - ann).days / 365.25


def pay_date(anchor: pd.Timestamp, lag_months: int) -> pd.Timestamp:
    """Payment date of the dividend declared for the fiscal year ending at ``anchor``."""
    return anchor + pd.DateOffset(months=int(lag_months)) if lag_months else anchor


def _lag(lag_months: int) -> float:
    a = pd.Timestamp(2000, 12, 31)
    return _years(a, pay_date(a, lag_months))


def _y(fc: Forecast, t: int) -> float:
    """Liquid-asset yield in the payment year of the FY t dividend (year t + 1): the forecast's T-bill path."""
    rates = list(fc.drivers.rate)
    return float(rates[min(t, len(rates) - 1)]) if rates else 0.0


def _phi(fc: Forecast, r: float, t: int, lag_months: int) -> float:
    return ((1 + _y(fc, t)) / (1 + r)) ** _lag(lag_months) if lag_months else 1.0


def v_ex(fc: Forecast, r: float, k: int = 0, model: str = "rim", lag_months: int = 0) -> float:
    dr = fc.drivers
    N = fc.N
    if r <= dr.g:
        return float("nan")
    E = fc.rows["E"].to_numpy()  # rows are indexed t = 0..N in order
    disc = (1 + r) ** np.arange(1, N - k + 1)
    divs = fc.rows["Div"].to_numpy()[k + 1:]
    tvd = (dr.roe_star - dr.g) * E[N] / ((r - dr.g) * (1 + r) ** (N - k))  # PV of the terminal dividend stream
    phi = np.array([_phi(fc, r, t, lag_months) for t in range(k + 1, N + 1)])
    phi_T = _phi(fc, r, N + 1, lag_months)
    if model == "rim":
        ri = fc.rows["NP_attr"].to_numpy()[k + 1:] - r * E[k:N]
        tv = (dr.roe_star - r) * E[N] / ((r - dr.g) * (1 + r) ** (N - k))
        held = (divs * (phi - 1) / disc).sum() + tvd * (phi_T - 1)  # residual income on dividends held until paid
        return float(E[k] + (ri / disc).sum() + tv + held)
    return float((divs * phi / disc).sum() + tvd * phi_T)


def _held(fc: Forecast, r: float, k: int, div: float, lag_months: int) -> float:
    """Dividend k plus the yield earned on it until its payment date (the amount that leaves at P_k)."""
    return div * (1 + _y(fc, k)) ** _lag(lag_months) if lag_months else div


def value_at(fc: Forecast, r: float, date: pd.Timestamp, model: str = "rim", anchor_div0: float = 0.0,
             lag_months: int = 0) -> float:
    """Equity value at ``date`` (VALUATION_SPEC §6 tau convention, year-end anchors; payment lag per module doc)."""
    y0 = int(fc.rows.loc[fc.rows["t"] == 0, "year"].iloc[0])
    k = max(0, min(fc.N - 1, date.year - y0 - (0 if (date.month, date.day) == (12, 31) else 1)))
    anchor = pd.Timestamp(y0 + k, 12, 31)
    f = _years(anchor, date)
    v = v_ex(fc, r, k, model, lag_months) * (1 + r) ** f
    pay = pay_date(anchor, lag_months)
    if date <= pay:  # declared, not yet paid: cum (L = 0: only on the anchor date itself)
        div = anchor_div0 if k == 0 else float(fc.rows.loc[fc.rows["t"] == k, "Div"].iloc[0])
        v += _held(fc, r, k, div, lag_months) / (1 + r) ** _years(date, pay) if lag_months else div
    return v


def blend(fc: Forecast, r: float, date: pd.Timestamp, w: dict[str, float], anchor_div0: float = 0.0,
          lag_months: int = 0) -> dict:
    rim = value_at(fc, r, date, "rim", anchor_div0, lag_months)
    ddm = value_at(fc, r, date, "ddm", anchor_div0, lag_months)
    return {"rim": rim, "ddm": ddm, "blend": w["rim"] * rim + w["ddm"] * ddm,
            "gap": abs(rim - ddm) / abs(rim) if rim else float("nan")}


def target_identity(fc: Forecast, r: float, vd: pd.Timestamp, model: str = "rim", anchor_div0: float = 0.0,
                    lag_months: int = 0) -> dict:
    """V1 computed independently at vd + 12 months against V0 (1 + r) - amounts leaving the value in between (each
    dividend with the yield earned until paid, carried from its payment date; payment lag per module doc)."""
    v1d = vd + pd.DateOffset(years=1)
    v0 = value_at(fc, r, vd, model, anchor_div0, lag_months)
    v1 = value_at(fc, r, v1d, model, anchor_div0, lag_months)
    out_div = 0.0
    for _, row in fc.rows[fc.rows["t"] >= 0].iterrows():
        t = int(row["t"])
        pay = pay_date(pd.Timestamp(int(row["year"]), 12, 31), lag_months)
        leaves = pay + pd.Timedelta(days=1)  # cum on the payment date, ex the day after
        if vd < leaves <= v1d:
            div = anchor_div0 if t == 0 else float(row["Div"])
            out_div += _held(fc, r, t, div, lag_months) * (1 + r) ** _years(pay, v1d)
    expected = v0 * (1 + r) ** _years(vd, v1d) - out_div
    return {"v0": v0, "v1": v1, "v1_identity": expected, "residual": (v1 - expected) / v0}


def justified(roe_star: float, g: float, r: float, payout_star: float) -> dict:
    """VALUATION_SPEC §6 justified P/B and P/E."""
    return {"pb": (roe_star - g) / (r - g), "pe": payout_star * (1 + g) / (r - g)}


def implied_coe(op: Opening, dr: Drivers, price_equity: float, date: pd.Timestamp, anchor_div0: float = 0.0,
                lo: float | None = None, hi: float = 0.60, lag_months: int = 0) -> float:
    """r at which the RIM value equals the market capitalisation, forecast fixed (bracketed root)."""
    fc = project(op, dr)
    lo = lo if lo is not None else dr.g + 1e-4

    def f(r: float) -> float:
        return value_at(fc, r, date, "rim", anchor_div0, lag_months) - price_equity

    try:
        return float(brentq(f, lo, hi)) if f(lo) * f(hi) < 0 else float("nan")
    except ValueError:
        return float("nan")


def implied_roe_star(op: Opening, dr: Drivers, r: float, price_equity: float, date: pd.Timestamp,
                     anchor_div0: float = 0.0, lo: float = -0.10, hi: float = 0.80, lag_months: int = 0) -> float:
    """ROE* at which the value equals the price, r fixed; payout* = 1 - g / ROE* keeps g internally funded."""

    def f(x: float) -> float:
        d = replace(dr, roe_star=x, payout_star=1 - dr.g / x if x else 0.0)
        return value_at(project(op, d), r, date, "rim", anchor_div0, lag_months) - price_equity

    try:
        return float(brentq(f, max(lo, dr.g + 1e-4), hi)) if f(max(lo, dr.g + 1e-4)) * f(hi) < 0 else float("nan")
    except ValueError:
        return float("nan")


def iso_surface(op: Opening, dr: Drivers, r: float, price_equity: float, date: pd.Timestamp,
                lams: np.ndarray, ers: np.ndarray, anchor_div0: float = 0.0, lag_months: int = 0) -> pd.DataFrame:
    """Value / price over (lambda, ER*) and the ER* that matches the price for each lambda (blueprint D12)."""
    rows = []
    for lam in lams:
        vals = []
        for er in ers:
            x = r + er
            d = replace(dr, lam=float(lam), roe_star=x, payout_star=1 - dr.g / x if x else 0.0, fade_shape="lambda")
            vals.append(value_at(project(op, d), r, date, "rim", anchor_div0, lag_months) / price_equity)
        v = np.array(vals)
        cross = float("nan")
        for i in range(len(ers) - 1):
            if (v[i] - 1) * (v[i + 1] - 1) <= 0 and v[i + 1] != v[i]:
                cross = float(ers[i] + (1 - v[i]) * (ers[i + 1] - ers[i]) / (v[i + 1] - v[i]))
                break
        for er, vv in zip(ers, v, strict=True):
            rows.append({"lambda": float(lam), "er_star": float(er), "value_to_price": float(vv),
                         "er_star_at_price": cross})
    return pd.DataFrame(rows)


def per_share(v: float, shares: float) -> float:
    return v / shares if shares and math.isfinite(shares) else float("nan")
