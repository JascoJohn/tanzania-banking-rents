"""Driver-based forecast engine (VALUATION_SPEC §4; blueprint §2 and D8; prompt 09a).

Explicit years t = 1..T (VALUATION_SPEC §4 projection equations):

    L_t = L_t-1 (1 + g_L),  D_t = D_t-1 (1 + g_D),  OL_t = omega TA_t,  LA_min = l_min D_t
    Gap_t = L_t + theta TA_t + LA_min - (D_t + OL_t + E_tot,t);  W_t = max(0, Gap);  LA_t = LA_min + max(0, -Gap)
    TA_t = L_t + LA_t + theta TA_t                                   (solved algebraically each year)
    NIM_t = NIM path_t + (asset_beta - deposit_beta x avg D / avg IEA) x (rate_t - rate_0)   (open issue 11)
    NII = NIM avg(L + LA) - s_W avg W;  NonII = phi avg TA;  OpEx = CIR (NII + NonII);  Imp = CoR avg L
    NP = (NII + NonII - OpEx - Imp)(1 - t);  NP_attr = NP (1 - nu)      [x FX factor on the subsidiary share, CRDB]
    E_t = E_t-1 + NP_attr - Div_t;  T1_t = E_t (1 - delta);  RWA_t = rho TA_t
    Div_t = min(pi_max NP_attr, max(0, ((E_t-1 + NP_attr)(1 - delta) - k* RWA_t) / (1 - delta)))
            [CRDB: also <= parent profit + u x subsidiary dividends]

E_tot = E_attr / (1 - kappa), kappa = NCI share of total equity at the anchor (constant). The year is solved by fixed-point
iteration because E_t enters TA_t. NMB: delta and rho are calibrated on Bank-solo Tier 1 and RWA against Group equity and
assets (D2), so the rule binds on the Bank-solo ratio.

Fade years T+1..T+F: ROE moves to ROE* (``lambda``: ROE_t - ROE* = lambda (ROE_t-1 - ROE*), owner pre-flight D / D8;
``linear``: VALUATION_SPEC §4); equity growth moves linearly to terminal g; Div = NP_attr - g_t B_t-1.
Terminal: residual income (ROE* - r) B_N growing at g; dividends (ROE* - g) B_N, which requires g = ROE* (1 - payout*).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


class EngineCheckError(RuntimeError):
    """A VALUATION_SPEC §4 / §10 engine check failed: the run stops with this named error."""


@dataclass
class Opening:
    """Anchor-year balances (Group; equity attributable ex the declared final dividend)."""

    year: int
    loans: float
    deposits: float
    liquid: float
    wholesale: float
    equity_attr: float
    nci_equity: float
    total_assets: float

    @property
    def other_liab(self) -> float:
        return self.total_assets - self.deposits - self.wholesale - self.equity_attr - self.nci_equity


@dataclass
class Drivers:
    """One scenario's driver set (assumption register values)."""

    g_L: list[float]
    g_D: list[float]
    NIM: list[float]
    phi: list[float]
    CIR: list[float]
    CoR: list[float]
    rho: list[float]
    rate: list[float]
    rate0: float
    omega: float
    theta: float
    l_min: float
    s_W: float
    tax: float
    nu: float
    delta: float
    k_star: float
    pi_max: float
    roe_star: float
    g: float
    payout_star: float
    lam: float
    fade_years: int = 10
    fade_shape: str = "lambda"
    asset_beta: float = 0.0
    deposit_beta: float = 0.0
    nowcast: dict | None = None            # {"np_attr_ytd": x, "months": m}: first year = YTD actual + remainder
    sub: dict | None = None                # {"s_sub", "u", "payout_sub", "fx": [annual TZS depreciation vs sub ccy]}
    ldr_bounds: tuple[float, float] = (0.0, 10.0)
    wholesale_max: float = 1.0
    k_req: float | None = None             # requirement only: the capital CHECK in the nowcast year (D56)
    div1_override: float | None = None     # sensitivity: year-1 dividend set by hand (e.g. the actual declared one)
    phase_in: bool = True                  # D56: k* phased in from the year-1 ratio when the gap is <= phase_in_max_gap
    phase_in_max_gap: float = 0.01
    icg_cap: bool = False                  # D56 (base case): from year 2, growth capped by internal capital generation
    g_fade_start: float | None = None      # D64 amended: fade-start growth (IMF nominal GDP); None = last explicit g_E
    payout_floor: float = 0.20

    @property
    def T(self) -> int:
        return len(self.g_L)

    def shifted(self, **kw) -> Drivers:
        """Copy with additive shifts on path drivers (``NIM=+0.005``) or replacements of scalars."""
        d = replace(self)
        for k, v in kw.items():
            cur = getattr(d, k)
            setattr(d, k, [x + v for x in cur] if isinstance(cur, list) else v)
        return d


@dataclass
class Forecast:
    rows: pd.DataFrame                       # one row per year (explicit then fade), index 0 = anchor
    drivers: Drivers
    checks: list[dict] = field(default_factory=list)

    @property
    def N(self) -> int:
        return int(self.rows["t"].max())


def _year(prev: dict, dr: Drivers, i: int, kappa: float, tol: float = 1e-10, k_t: float | None = None) -> dict:
    k_t = dr.k_star if k_t is None else k_t
    L = prev["L"] * (1 + dr.g_L[i])
    D = prev["D"] * (1 + dr.g_D[i])
    la_min = dr.l_min * D
    E = prev["E"]
    out: dict = {}
    for _ in range(200):
        e_tot = E / (1 - kappa)
        ta_w = (L + la_min) / (1 - dr.theta)
        gap = L + dr.theta * ta_w + la_min - (D + dr.omega * ta_w + e_tot)
        if gap > 0:
            W, TA, LA = gap, ta_w, la_min
        else:
            W, TA = 0.0, (D + e_tot) / (1 - dr.omega)
            LA = TA * (1 - dr.theta) - L
        avg_iea = ((prev["L"] + prev["LA"]) + (L + LA)) / 2
        avg_d = (prev["D"] + D) / 2
        nim = dr.NIM[i] + (dr.asset_beta - dr.deposit_beta * avg_d / avg_iea) * (dr.rate[i] - dr.rate0)
        nii = nim * avg_iea - dr.s_W * (prev["W"] + W) / 2
        nonii = dr.phi[i] * (prev["TA"] + TA) / 2
        opex = dr.CIR[i] * (nii + nonii)
        imp = dr.CoR[i] * (prev["L"] + L) / 2
        pbt = nii + nonii - opex - imp
        np_ = pbt * (1 - dr.tax)
        npa = np_ * (1 - dr.nu)
        fx_f, distributable = 1.0, np.inf
        if dr.sub:
            fx_f = float(np.prod([1 - x for x in dr.sub["fx"][: i + 1]]))
            s = dr.sub["s_sub"]
            npa = npa * (1 - s + s * fx_f)
        if i == 0 and dr.nowcast:
            m = dr.nowcast["months"]
            npa = dr.nowcast["np_attr_ytd"] + (12 - m) / 12 * npa
        if dr.sub:
            s_eff = dr.sub["s_sub"] * fx_f / (1 - dr.sub["s_sub"] + dr.sub["s_sub"] * fx_f)
            distributable = npa * (1 - s_eff) + dr.sub["u"] * dr.sub["payout_sub"] * npa * s_eff
        rwa = dr.rho[i] * TA
        cap_div = max(0.0, ((prev["E"] + npa) * (1 - dr.delta) - k_t * rwa) / (1 - dr.delta))
        div = max(0.0, min(dr.pi_max * npa, cap_div, distributable))
        if i == 0 and dr.div1_override is not None:
            div = dr.div1_override
        e_new = prev["E"] + npa - div
        out = {"L": L, "D": D, "LA": LA, "LA_min": la_min, "W": W, "TA": TA, "OL": dr.omega * TA,
               "NEA": dr.theta * TA, "E": e_new, "E_tot": e_new / (1 - kappa), "NIM_eff": nim, "NII": nii,
               "NonII": nonii, "OpEx": opex, "Imp": imp, "PBT": pbt, "NP": np_, "NP_attr": npa, "Div": div,
               "RWA": rwa, "T1": e_new * (1 - dr.delta), "cap_div": cap_div, "distributable": distributable,
               "fx_factor": fx_f, "phase": "explicit", "k_target": k_t}
        if abs(e_new - E) <= tol * max(1.0, abs(E)):
            break
        E = e_new
    return out


def _icg_slack(prev: dict, dr: Drivers, i: int, kappa: float, k_t: float, g: float) -> float:
    """Capital room above the payout floor when year i's loan growth is g (deposits keep their spread to loans)."""
    d2 = replace(dr, g_L=[*dr.g_L[:i], g, *dr.g_L[i + 1:]],
                 g_D=[*dr.g_D[:i], g + (dr.g_D[i] - dr.g_L[i]), *dr.g_D[i + 1:]])
    y = _year(prev, d2, i, kappa, k_t=k_t)
    return y["cap_div"] - dr.payout_floor * y["NP_attr"]


def project(op: Opening, dr: Drivers) -> Forecast:
    """Explicit years, fade and terminal inputs (VALUATION_SPEC §4)."""
    kappa = op.nci_equity / (op.equity_attr + op.nci_equity) if op.equity_attr + op.nci_equity else 0.0
    prev = {"L": op.loans, "D": op.deposits, "LA": op.liquid, "W": op.wholesale, "TA": op.total_assets,
            "E": op.equity_attr}
    rows = [{"t": 0, "year": op.year, **prev, "E_tot": op.equity_attr + op.nci_equity, "phase": "anchor",
             "OL": op.other_liab, "NEA": op.total_assets - op.loans - op.liquid}]
    k_path: list[float] = [dr.k_star] * dr.T
    for i in range(dr.T):
        r = _year(prev, dr, i, kappa, k_t=k_path[i])
        if i == 0 and dr.phase_in and dr.T > 1:
            # D56 (owner, 09b pre-flight): if year 1 ends below k* by at most phase_in_max_gap, k* rises linearly from
            # the year-1 ratio to the full target by the last explicit year; a larger gap is left to the check (stop)
            ratio1 = r["T1"] / r["RWA"]
            gap = dr.k_star - ratio1
            if 0 < gap <= dr.phase_in_max_gap + 1e-12:
                k_path = [dr.k_star] + [ratio1 + gap * j / (dr.T - 1) for j in range(1, dr.T)]
        if i >= 1 and dr.icg_cap and r["cap_div"] < dr.payout_floor * r["NP_attr"] - 1e-6:
            # D56: the year's loan (and deposit) growth is lowered to the largest value at which the capital rule
            # still allows the payout floor, i.e. RWA growth <= ROE (1 - floor) adjusted for Tier 1 / equity and the
            # phased k*; found by bisection on the growth rate (loans and deposits scale together)
            g0 = dr.g_L[i]

            lo, hi = -0.5, g0
            if _icg_slack(prev, dr, i, kappa, k_path[i], lo) >= 0:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if _icg_slack(prev, dr, i, kappa, k_path[i], mid) >= 0 else (lo, mid)
                g_new = lo
                dg = g_new - g0
                dr = replace(dr, g_L=[*dr.g_L[:i], g_new, *dr.g_L[i + 1:]],
                             g_D=[*dr.g_D[:i], dr.g_D[i] + dg, *dr.g_D[i + 1:]])
                r = _year(prev, dr, i, kappa, k_t=k_path[i])
                r["icg_capped_from"] = g0
        rows.append({"t": i + 1, "year": op.year + i + 1, **r})
        prev = r
    last, before = rows[-1], rows[-2]
    roe_T = last["NP_attr"] / before["E"]
    g_T = last["E"] / before["E"] - 1 if dr.g_fade_start is None else dr.g_fade_start  # D64 amended
    E = last["E"]
    for f in range(1, dr.fade_years + 1):
        if dr.fade_shape == "lambda":
            roe = dr.roe_star + dr.lam ** f * (roe_T - dr.roe_star)
        else:
            roe = roe_T + (dr.roe_star - roe_T) * f / dr.fade_years
        gf = g_T + (dr.g - g_T) * f / dr.fade_years
        npa = roe * E
        div = npa - gf * E
        rows.append({"t": dr.T + f, "year": op.year + dr.T + f, "E": E * (1 + gf), "NP_attr": npa, "Div": div,
                     "ROE": roe, "g_E": gf, "phase": "fade"})
        E = E * (1 + gf)
    df = pd.DataFrame(rows)
    df["E_prev"] = df["E"].shift(1)
    df["ROE"] = df["NP_attr"] / df["E_prev"]
    fc = Forecast(df, dr)
    fc.checks = engine_checks(fc)
    return fc


def engine_checks(fc: Forecast, tol: float = 1e-6) -> list[dict]:
    """VALUATION_SPEC §4 and §10 checks on one forecast; each row: check, status, worst value."""
    d, dr = fc.rows, fc.drivers
    ex = d[d["phase"] == "explicit"]
    out = []

    def add(name: str, ok: bool, worst: float, detail: str = "") -> None:
        out.append({"check": name, "status": "PASS" if ok else "FAIL", "worst": float(worst), "detail": detail})

    bal = (ex["TA"] - (ex["D"] + ex["OL"] + ex["E_tot"] + ex["W"])).abs() / ex["TA"]
    add("balance sheet closes (A = L + E)", bool((bal < tol).all()), bal.max())
    assets = (ex["TA"] - (ex["L"] + ex["LA"] + ex["NEA"])).abs() / ex["TA"]
    add("asset side adds up", bool((assets < tol).all()), assets.max())
    # D56 (owner, 09b pre-flight): in the nowcast year (year 1, interim-based) the check uses the requirement only;
    # the buffered k* applies from year 2. Dividends follow the rule with k* in every year.
    k = ex["k_target"].astype(float).copy()
    if dr.nowcast and dr.k_req is not None:
        k.iloc[0] = dr.k_req
    cap = (ex["T1"] - k * ex["RWA"]) / ex["RWA"]
    phased = bool((ex["k_target"].iloc[1:] < dr.k_star - 1e-12).any())
    add("Tier 1 >= k* x RWA", bool((cap >= -tol).all()), cap.min(),
        f"k* = {dr.k_star:.4f}" + (f"; year 1 (nowcast) vs requirement {dr.k_req:.4f}" if dr.nowcast and
                                   dr.k_req is not None else "") + ("; k* phased in from the year-1 ratio (D56)"
                                                                   if phased else ""))
    gap1 = dr.k_star - float(ex["T1"].iloc[0] / ex["RWA"].iloc[0])
    add("year-1 gap to k* <= 1 pp (phase-in allowed; else STOP)", gap1 <= dr.phase_in_max_gap + 1e-12, gap1)
    liq = (ex["LA"] - ex["LA_min"]) / ex["TA"]
    add("liquid assets >= floor", bool((liq >= -tol).all()), liq.min())
    ldr = ex["L"] / ex["D"]
    add("LDR within bounds", bool(ldr.between(*dr.ldr_bounds).all()), ldr.max(),
        f"bounds {dr.ldr_bounds[0]:.2f}-{dr.ldr_bounds[1]:.2f}")
    ws = ex["W"] / (ex["D"] + ex["W"])
    add("wholesale share within bound", bool((ws <= dr.wholesale_max + tol).all()), ws.max(),
        f"max {dr.wholesale_max:.2f}")
    cs = (d["E"] - (d["E_prev"] + d["NP_attr"] - d["Div"])).iloc[1:].abs() / d["E"].iloc[1:].abs()
    add("clean surplus (E = E_-1 + NP_attr - Div)", bool((cs < tol).all()), cs.max())
    add("terminal: g = ROE* (1 - payout*)", abs(dr.g - dr.roe_star * (1 - dr.payout_star)) < 1e-9,
        dr.g - dr.roe_star * (1 - dr.payout_star))
    neg = d.loc[d["phase"] == "fade", "Div"]
    add("fade dividends non-negative", bool((neg >= -tol).all()), neg.min() if len(neg) else 0.0)
    return out


def require(checks: list[dict]) -> None:
    """Stop the run with a named error when any check failed (VALUATION_SPEC §4: 'any breach stops the run')."""
    bad = [c for c in checks if c["status"] == "FAIL"]
    if bad:
        raise EngineCheckError("; ".join(f"{c['check']} (worst {c['worst']:.6g})" for c in bad))
