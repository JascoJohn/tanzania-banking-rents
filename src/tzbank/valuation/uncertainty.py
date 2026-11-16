"""Uncertainty layer (VALUATION_SPEC §7): sensitivity grids, scenarios, seeded Monte Carlo; and the five-step valuation
bridge (VALUATION_SPEC §2). Every function takes a ``Case`` (opening, drivers, COE, date, shares) and re-runs the engine;
nothing here is a separate model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from tzbank.banks.engine import Drivers, Opening, project
from tzbank.valuation.models import _years, blend


@dataclass
class Case:
    bank: str
    val: str
    vd: pd.Timestamp
    op: Opening
    dr: Drivers
    r: float
    shares: float
    price: float
    w: dict
    anchor_div0: float = 0.0
    lag_months: int = 0  # dividend payment lag (valuation.dividend_pay_lag_months; DDM only)


def value(c: Case, dr: Drivers | None = None, r: float | None = None) -> float:
    """Blended equity value at the valuation date (TZS)."""
    fc = project(c.op, dr or c.dr)
    return blend(fc, r if r is not None else c.r, c.vd, c.w, c.anchor_div0, c.lag_months)["blend"]


def _with_g(dr: Drivers, g: float) -> Drivers:
    return replace(dr, g=g, payout_star=1 - g / dr.roe_star)


def _with_roe(dr: Drivers, roe: float) -> Drivers:
    return replace(dr, roe_star=roe, payout_star=1 - dr.g / roe)


def grids(c: Case, step_bp: int = 50, half_bp: int = 200) -> dict[str, pd.DataFrame]:
    """COE x terminal g, COE x ROE*, NIM x cost of risk: value per share, +/- 200 bp in 50 bp steps."""
    s = np.arange(-half_bp, half_bp + 1, step_bp) / 1e4
    out = {}
    rows = []
    for dr_ in s:
        for dg in s:
            g = c.dr.g + dg
            v = value(c, _with_g(c.dr, g), c.r + dr_) if c.r + dr_ > g else np.nan
            rows.append({"coe": c.r + dr_, "g": g, "value_per_share": v / c.shares})
    out["coe_x_g"] = pd.DataFrame(rows)
    rows = []
    for dr_ in s:
        for de in s:
            roe = c.dr.roe_star + de
            v = value(c, _with_roe(c.dr, roe), c.r + dr_) if roe > c.dr.g and c.r + dr_ > c.dr.g else np.nan
            rows.append({"coe": c.r + dr_, "roe_star": roe, "value_per_share": v / c.shares})
    out["coe_x_roe_star"] = pd.DataFrame(rows)
    rows = []
    for dn in s:
        for dc in s:
            d = c.dr.shifted(NIM=dn, CoR=dc)
            d = replace(d, CoR=[max(0.0, x) for x in d.CoR])
            rows.append({"nim_shift": dn, "cor_shift": dc, "value_per_share": value(c, d) / c.shares})
    out["nim_x_cor"] = pd.DataFrame(rows)
    return out


MC_VARS = ["NIM", "CoR", "CIR", "g_A", "COE", "LP", "rate"]


def mc_bounds(base: Drivers, bear: Drivers, bull: Drivers, coe_shift: tuple[float, float],
              lp_shift: tuple[float, float]) -> dict[str, tuple[float, float]]:
    """Triangular bounds (low, high) of each shift, mode 0, from the register's bear and bull driver sets."""
    def d(name: str) -> tuple[float, float]:
        a = float(np.mean(np.array(getattr(bear, name)) - np.array(getattr(base, name))))
        b = float(np.mean(np.array(getattr(bull, name)) - np.array(getattr(base, name))))
        return min(a, b, 0.0), max(a, b, 0.0)
    ga = (min(d("g_L")[0], d("g_D")[0]), max(d("g_L")[1], d("g_D")[1]))
    return {"NIM": d("NIM"), "CoR": d("CoR"), "CIR": d("CIR"), "g_A": ga, "COE": coe_shift, "LP": lp_shift,
            "rate": d("rate")}


def _tri_ppf(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-12:
        return np.zeros_like(u)
    c = (0 - lo) / (hi - lo)
    return stats.triang.ppf(u, c, loc=lo, scale=hi - lo)


def monte_carlo(c: Case, bounds: dict[str, tuple[float, float]], corr: dict[str, float], draws: int,
                seed: list[int], persist: pd.DataFrame | None = None, coe_base: float | None = None) -> pd.DataFrame:
    """Gaussian-copula draws with triangular marginals (mode = base); correlations NIM-rate and CoR-g_A. With
    ``persist`` (owner, ch09-11 review B), each draw also takes one (lambda, ER*) pair jointly from the chapter 5
    moving-block bootstrap replicates; ROE* = base COE + ER*, payout* = 1 - g / ROE* (draws with ROE* <= g + 0.5 pp
    are left empty and counted)."""
    k = len(MC_VARS)
    R = np.eye(k)
    ix = {v: i for i, v in enumerate(MC_VARS)}
    for (a, b), rho in (((ix["NIM"], ix["rate"]), corr.get("NIM_rate", 0.0)),
                        ((ix["CoR"], ix["g_A"]), corr.get("CoR_growth", 0.0))):
        R[a, b] = R[b, a] = rho
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(k), R, size=draws)
    u = stats.norm.cdf(z)
    x = {v: _tri_ppf(u[:, ix[v]], *bounds[v]) for v in MC_VARS}
    vals = np.empty(draws)
    lam = np.full(draws, c.dr.lam)
    ers = np.full(draws, np.nan)
    if persist is not None and len(persist):
        pick = rng.integers(0, len(persist), size=draws)
        lam = persist["lambda"].to_numpy()[pick]
        ers = persist["er_star"].to_numpy()[pick]
    for i in range(draws):
        d = c.dr.shifted(NIM=x["NIM"][i], CoR=x["CoR"][i], CIR=x["CIR"][i], g_L=x["g_A"][i], g_D=x["g_A"][i],
                         rate=x["rate"][i])
        d = replace(d, CoR=[max(0.0, v) for v in d.CoR])
        if persist is not None and len(persist):
            roe = (coe_base if coe_base is not None else c.r) + ers[i]
            if roe <= d.g + 0.005:
                vals[i] = np.nan
                continue
            d = replace(d, lam=float(lam[i]), roe_star=roe, payout_star=1 - d.g / roe, fade_shape="lambda")
        r = c.r + x["COE"][i] + x["LP"][i]
        vals[i] = value(c, d, r) / c.shares if r > d.g else np.nan
    return pd.DataFrame({**{v: x[v] for v in MC_VARS}, "lambda": lam, "er_star": ers, "value_per_share": vals})


def scenarios(cases: dict[str, Case]) -> pd.DataFrame:
    """Value per share and the first three forecast years' EPS and ROE for bear, base and bull."""
    rows = []
    for name, c in cases.items():
        fc = project(c.op, c.dr)
        ex = fc.rows[fc.rows["phase"] == "explicit"].head(3)
        rows.append({"scenario": name, "value_per_share": value(c) / c.shares,
                     **{f"eps_{int(r.year)}": r.NP_attr / c.shares for r in ex.itertuples()},
                     **{f"roe_{int(r.year)}": r.ROE for r in ex.itertuples()}})
    return pd.DataFrame(rows)


def bridge(hist: Case, curr: Case, hist_drivers_on_curr: Drivers, div_paid_between: float,
           div_pay_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Five-step sequential bridge, historical -> current equity value (VALUATION_SPEC §2; fixed order). Dividends
    paid between the dates are carried from their payment date to the current date at the historical COE (owner
    issue 17); without a payment date they are deducted at face value."""
    v0 = value(hist)
    v1 = v0 * (1 + hist.r) ** _years(hist.vd, curr.vd)                       # 1 passage of time
    carry = (1 + hist.r) ** _years(div_pay_date, curr.vd) if div_pay_date is not None else 1.0
    v2 = v1 - div_paid_between * carry                                       # 2 dividends paid between the dates
    c3 = replace(curr, dr=hist_drivers_on_curr, r=hist.r)
    v3 = value(c3)                                                           # 3 new information (actuals)
    c4 = replace(curr, r=hist.r)
    v4 = value(c4)                                                           # 4 forecast revisions
    v5 = value(curr)                                                         # 5 change in COE
    steps = [("historical value", v0), ("1 passage of time", v1 - v0), ("2 dividends paid", v2 - v1),
             ("3 new information", v3 - v2), ("4 forecast revisions", v4 - v3), ("5 change in COE", v5 - v4)]
    df = pd.DataFrame(steps, columns=["step", "tzs"])
    df.loc[len(df)] = ["current value", v5]
    df.loc[len(df)] = ["residual (reported, not absorbed)", v5 - df["tzs"].iloc[:6].sum()]
    return df
