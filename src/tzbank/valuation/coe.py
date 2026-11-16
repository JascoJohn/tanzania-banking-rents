"""Cost of equity, approaches A and B, and the reconciliation rule (VALUATION_SPEC §5).

A (TZS build-up):  COE_A = (y_long - DS) + beta (ERP + CRP) + LP
B (USD CAPM):      COE_USD = rf_US + beta (ERP + CRP) + LP;  COE_B = (1 + COE_USD)(1 + pi_TZ)/(1 + pi_US) - 1
Reconciliation:    |A - B| < 150 bp -> midpoint; otherwise the register records the choice (here: A, flagged).
B needs forward inflation for Tanzania and the US; while those inputs are TO COLLECT, B is not computed and the
central COE is A (flagged), never a trailing-CPI substitute (the spec requires forward-looking inflation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class COE:
    a: float
    b: float
    central: float
    low: float
    high: float
    rule: str


def approach_a(y_long: float, ds: float, beta: float, erp: float, crp: float, lp: float) -> float:
    """VALUATION_SPEC §5 approach A."""
    return (y_long - ds) + beta * (erp + crp) + lp


def usd_capm(rf_us: float, beta: float, erp: float, crp: float, lp: float) -> float:
    """VALUATION_SPEC §5, the USD leg of approach B: rf_US + beta (ERP + CRP) + LP (also the chapter 12 hurdle)."""
    return rf_us + beta * (erp + crp) + lp


def approach_b(rf_us: float, beta: float, erp: float, crp: float, lp: float, pi_tz: float | None,
               pi_us: float | None) -> float:
    """VALUATION_SPEC §5 approach B; NaN when either forward inflation input is missing."""
    if pi_tz is None or pi_us is None or not (math.isfinite(pi_tz) and math.isfinite(pi_us)):
        return float("nan")
    usd = usd_capm(rf_us, beta, erp, crp, lp)
    return (1 + usd) * (1 + pi_tz) / (1 + pi_us) - 1


def reconcile(a: float, b: float, threshold_bp: float = 150) -> COE:
    """Midpoint when |A - B| < threshold; A (flagged for the register) otherwise or when B is missing."""
    if not math.isfinite(b):
        return COE(a, b, a, a, a, "B not computable (forward inflation TO COLLECT): central = A, flagged")
    lo, hi = min(a, b), max(a, b)
    if abs(a - b) * 1e4 < threshold_bp:
        return COE(a, b, (a + b) / 2, lo, hi, f"|A - B| < {threshold_bp:.0f} bp: midpoint")
    return COE(a, b, a, lo, hi, f"|A - B| >= {threshold_bp:.0f} bp: investigate; central = A pending the register")
