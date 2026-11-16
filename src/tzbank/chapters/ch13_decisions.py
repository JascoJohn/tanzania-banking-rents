"""Owner decisions on contested analytical choices (prompt 18; D83-D86): pipeline tables for the report text.

Nothing here changes the registered rating rule (D4), the base-case inputs or the ratings. Every analysis shows how
much a conclusion depends on a contested choice, by re-valuing the register's base case with one choice changed
(``ch13_report_support.quote``: value, 12-month target, expected return and the D4 rating). Tables
``outputs/tables/ch13_dec_*.csv``, charts ``outputs/figures/ch13_dec_*.png``, manifest ``outputs/manifests/
ch13_dec.json`` (dashboard chapter 13 page), findings pack v1.10 (CH13.DEC_*).

- C3/R1 ``interp_weights``: weight w on interpretation B (2015-18 exceptional: break-adjusted ER*) against A (normal
  cycle: full-cycle ER*), same fade speed; value, value/price, rating and the price-crossing weight.
- R6 ``mc_conditional``: P(value > price) given A (base-centred Monte Carlo) and given B (bull-centred), and the
  w-weighted mixture; ``mc_inputs``: the distributions of the drawn inputs.
- R3 ``configurations`` and ``joint_grid``: full-cycle ER* with full-cycle lambda, break-adjusted with break-adjusted,
  and the base (full-cycle ER*, break-adjusted lambda); ER* x lambda grid with the rating at each point.
- R2 ``bank_coe``: bank betas (monthly OLS; daily Dimson) with standard errors, Amihud illiquidity, a re-levered beta;
  COE, value and rating under each. Part 3 item 4: value at the local brokers' COEs.
- R5 ``hist_coe``: chapter 5 persistence with a constant COE, the time-varying series and +/- 2 points.
- R4/C10 ``capital_policies`` and ``surplus_drag_roles``; R9 ``terminal_growth``; R10 ``price_requires`` and
  ``implied_growth_table``; C2 ``rate_roles``; C4 ``rc_curr``; C5 ``event_robust``; C11 ``outliers``;
  C13 ``monitoring``; C14 ``income_vs_return``; R7 ``peer_structure``; Part 3 ``analyst_*``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import write_csv, write_json, write_text

FOCUS = ("B02", "B01")
T = {"B01": "CRDB", "B02": "NMB"}
BID = {v: k for k, v in T.items()}
DATES = ("HIST", "CURR")
W = (0.0, 0.25, 0.5, 0.75, 1.0)
PEER_G = 0.06  # R7: one illustrative long-run nominal growth for the African peers' justified P/B (Scenario)
LOCAL_COE = {"Tanzania Securities, NMB 2023 (used)": 0.1847, "Tanzania Securities, NMB 2023 (CAPM)": 0.205,
             "Tanzania Securities, CRDB 2024": 0.2065, "Solomon, both banks, Mar 2026": 0.1497}


def _t(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, low_memory=False, dtype={"bank_id": str})


def _foot(src: str, ev: str) -> str:
    return f"Source: {src}. Evidence class: {ev}."


def _reg(ctx, val: str, bank: str, symbol: str, col: str = "base") -> float:
    r = ctx.reg
    x = r[(r["valuation"] == val) & (r["bank"] == T[bank]) & (r["symbol"] == symbol)]
    return float(x[col].iloc[0]) if len(x) else np.nan


def _quote(ctx, val, b, dr=None, r=None) -> dict:
    from tzbank.chapters.ch13_report_support import quote

    return quote(ctx, val, b, dr, r)


def _with_er(c, er: float, lam: float | None = None):
    from tzbank.valuation import uncertainty as un

    d = un._with_roe(c.dr, c.r + er)  # noqa: SLF001
    return replace(d, lam=float(np.clip(lam, 0.0, 0.99))) if lam is not None else d


# --- C3 / R1 ------------------------------------------------------------------------------------------------------------
def interp_weights(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cross = [], []
    for val in DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            er_a, er_b = c.dr.roe_star - c.r, _reg(ctx, val, b, "ER_star", "bull")
            for w in W:
                er = (1 - w) * er_a + w * er_b
                q = _quote(ctx, val, b, _with_er(c, er))
                rows.append({"valuation": val, "bank": T[b], "weight_on_B": w, "er_star_A": er_a, "er_star_B": er_b,
                             "er_star": er, "lambda": c.dr.lam, "coe": c.r, "value_per_share": q["value_per_share"],
                             "price": c.price, "value_to_price": q["value_per_share"] / c.price,
                             "expected_return": q["expected_return"], "rating": q["rating"],
                             "evidence_class": "Scenario"})
            f = lambda w, c=c, b=b, val=val, ea=er_a, eb=er_b: _quote(ctx, val, b, _with_er(c, (1 - w) * ea + w * eb)  # noqa: E731
                                                    )["value_per_share"] - c.price
            lo, hi = 0.0, 1.0
            flo, fhi = f(lo), f(hi)
            if flo * fhi > 0:
                w_star, status = np.nan, ("value above price at every weight (no crossing within 0-100%)" if flo > 0
                                          else "value below price at every weight (no crossing within 0-100%)")
                # how far beyond 100% the crossing lies, for scale
                ext = next((x for x in np.arange(1.05, 5.01, 0.05) if f(x) >= 0), np.nan) if fhi < 0 else np.nan
            else:
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if f(mid) * flo > 0:
                        lo, flo = mid, f(mid)
                    else:
                        hi = mid
                w_star, status, ext = (lo + hi) / 2, "crossing within 0-100%", np.nan
            cross.append({"valuation": val, "bank": T[b], "price": c.price, "er_star_A": er_a, "er_star_B": er_b,
                          "crossing_weight": w_star, "status": status,
                          "weight_needed_if_beyond_100pct": ext,
                          "reading": "the weight on the exceptional-episode reading (B) at which value equals price; "
                                     "a weight on an interpretation of the past, not the probability of a crisis",
                          "evidence_class": "Scenario"})
    return pd.DataFrame(rows), pd.DataFrame(cross)


def weights_figure(t: pd.DataFrame, figd: Path) -> Path:
    fig, axs = plt_subplots(1, 2, figsize=(10, 4.0))
    for ax, b in zip(axs, ("NMB", "CRDB"), strict=True):
        for val, ls in (("CURR", "-"), ("HIST", ":")):
            g = t[(t["bank"] == b) & (t["valuation"] == val)]
            ratio = 10 if (b == "NMB" and val == "HIST") else 1  # NMB 31 Dec 2025 on the post-split basis
            date = "31 Oct 2026" if val == "CURR" else "31 Dec 2025"
            ax.plot(g["weight_on_B"] * 100, g["value_per_share"] / ratio, color=COLORS[BID[b]], ls=ls, marker="o",
                    ms=3, label=f"value, {date}")
            ax.axhline(g["price"].iloc[0] / ratio, color="#1f1f1d", ls=ls, lw=0.8, label=f"price, {date}")
        style(ax, f"{b}", "TZS per share (post-split)")
        ax.set_xlabel("weight on 2015-18 being exceptional, percent", fontsize=8)
    # one legend for both panels, below them (prompt 21: no legend over the data); line styles carry the dates
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend([h[i] for i in (0, 2, 1, 3)], ["value, 31 Oct 2026 (solid)", "value, 31 Dec 2025 (dotted)",
                                              "price, 31 Oct 2026", "price, 31 Dec 2025"],
               loc="lower center", ncol=4, fontsize=7, frameon=False)
    return save(fig, figd / "ch13_dec_interp_weights.png", _foot(
        "register v1.3 base case with ER* = (1 - w) x full-cycle + w x break-adjusted; w is a weight on an "
        "interpretation of the past, not the probability of a crisis in a given year", "Scenario"))


# --- R6 -----------------------------------------------------------------------------------------------------------------
def mc_conditional(ctx) -> pd.DataFrame:
    m = _t(ctx.root, "ch11_monte_carlo.csv")
    rows = []
    for val in DATES:
        for b in FOCUS:
            g = m[(m["valuation"] == val) & (m["bank"] == T[b])].set_index("variant")
            pa = float(g.loc[g.index.str.startswith("base")].iloc[0]["prob_value_above_price"])
            pb = float(g.loc[g.index.str.startswith("bull")].iloc[0]["prob_value_above_price"])
            for w in W:
                rows.append({"valuation": val, "bank": T[b], "weight_on_B": w,
                             "prob_value_above_price_given_A": pa, "prob_value_above_price_given_B": pb,
                             "prob_value_above_price_mixture": (1 - w) * pa + w * pb,
                             "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def mc_inputs(ctx) -> pd.DataFrame:
    rows = []
    for b in FOCUS:
        mc = ctx.monte_carlo("CURR", b)
        c = ctx.cases[("CURR", b, "base")]
        for var, series in (("ER*", mc["er_star"]), ("lambda", mc["lambda"]), ("COE", c.r + mc["COE"] + mc["LP"]),
                            ("loan growth shift", mc["g_A"])):
            s = pd.Series(series).dropna()
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                rows.append({"valuation": "CURR", "bank": T[b], "input": var, "quantile": q,
                             "value": float(s.quantile(q)), "draws": len(s), "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def mc_inputs_figure(ctx, figd: Path) -> tuple[Path, pd.DataFrame]:
    fig, axs = plt_subplots(1, 4, figsize=(11, 3.0))
    data = []
    for b in FOCUS:
        mc = ctx.monte_carlo("CURR", b)
        c = ctx.cases[("CURR", b, "base")]
        cols = {"ER* (break-adjusted draws, centred)": mc["er_star"] * 100, "persistence lambda": mc["lambda"],
                "COE, percent": (c.r + mc["COE"] + mc["LP"]) * 100, "growth shift, points": mc["g_A"] * 100}
        for ax, (lab, s) in zip(axs, cols.items(), strict=True):
            ax.hist(pd.Series(s).dropna(), bins=40, color=COLORS[BID[T[b]]], alpha=0.55, label=T[b])
            style(ax, lab, "draws")
            data += [{"bank": T[b], "input": lab, "value": float(v), "evidence_class": "Estimated"}
                     for v in pd.Series(s).dropna().sample(min(1000, len(s.dropna())), random_state=1)]
    axs[0].legend(fontsize=7, frameon=False)
    return save(fig, figd / "ch13_dec_mc_inputs.png", _foot("chapter 11 Monte Carlo, 31 Oct 2026, 10,000 draws per "
                                                           "bank (Kilian bootstrap for ER* and lambda)", "Estimated")), \
        pd.DataFrame(data)


# --- C6 / C7 ------------------------------------------------------------------------------------------------------------
def rating_application(ctx) -> pd.DataFrame:
    """Annex A9: the one rating rule (D4) applied per bank, with the smallest COE change (0.25-point steps, other drivers
    held) that changes the rating. D57 (how a call is judged afterwards) and D54 (evidence verdicts) are not ratings."""
    from tzbank.chapters.ch09_valuation import BUY, DOWN

    rows = []
    for val in DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            q = _quote(ctx, val, b)
            from tzbank.valuation import uncertainty as un

            bear = un.value(ctx.cases[(val, b, "bear")]) / c.shares
            sw = {}
            for sign in (-1, 1):
                for k in range(1, 41):
                    dx = sign * 0.0025 * k
                    if _quote(ctx, val, b, r=c.r + dx)["rating"] != q["rating"]:
                        sw[sign] = (dx, _quote(ctx, val, b, r=c.r + dx)["rating"])
                        break
            rows.append({"valuation": val, "bank": T[b], "price": c.price, "value_per_share": q["value_per_share"],
                         "target_12m": q["target_12m"], "dps_12m": q["dps_12m"],
                         "expected_return": q["expected_return"], "coe": c.r, "buy_threshold": c.r + BUY,
                         "sell_threshold": c.r - BUY, "bear_downside": bear / c.price - 1, "bear_limit": DOWN,
                         "rating": q["rating"],
                         "coe_change_to_switch_down": sw.get(-1, (np.nan, ""))[0],
                         "rating_after_lower_coe": sw.get(-1, (np.nan, ""))[1],
                         "coe_change_to_switch_up": sw.get(1, (np.nan, ""))[0],
                         "rating_after_higher_coe": sw.get(1, (np.nan, ""))[1],
                         "rule": "D4: Buy if E[r] > COE + 5 pp and bear downside >= -25%; Sell if E[r] < COE - 5 pp; "
                                 "else Hold", "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


# --- R3 -----------------------------------------------------------------------------------------------------------------
def configurations(ctx) -> pd.DataFrame:
    p = _t(ctx.root, "ch05_calibration_candidates.csv")
    rows = []
    for val in DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            bench = p[(p["bank_id"] == b) & (p["specification"] == "ar1_benchmark")].iloc[0]
            lam_full = float(bench["lambda"])
            er_a, er_b = c.dr.roe_star - c.r, _reg(ctx, val, b, "ER_star", "bull")
            cfgs = [("1. full-cycle ER*, full-cycle lambda", er_a, lam_full,
                     f"full-cycle lambda = {lam_full:.2f}: no identified mean reversion; the engine caps lambda at "
                     "0.99, so excess returns fade over the ten-year fade window from today's level, not from a "
                     "long-run mean (ER* is undefined at lambda = 1)"),
                    ("2. break-adjusted ER*, break-adjusted lambda", er_b, c.dr.lam, "internally consistent "
                     "exceptional-episode reading (2015-18 dummied out)"),
                    ("3. conservative hybrid (base case): full-cycle ER*, break-adjusted lambda", er_a, c.dr.lam,
                     "judgemental conservative specification: the full-cycle level with the break-adjusted speed, "
                     "because the full-cycle lambda identifies no reversion")]
            for name, er, lam, note in cfgs:
                q = _quote(ctx, val, b, _with_er(c, er, lam))
                case = {"1": "Full-cycle", "2": "Break-adjusted", "3": "Conservative hybrid (base case)"}[name[0]]
                rows.append({"valuation": val, "bank": T[b], "configuration": name, "case": case, "er_star": er,
                             "lambda": min(lam, 0.99), "lambda_estimated": lam, "value_per_share": q["value_per_share"],
                             "price": c.price, "value_to_price": q["value_per_share"] / c.price,
                             "expected_return": q["expected_return"], "rating": q["rating"], "note": note,
                             "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def joint_grid(ctx) -> pd.DataFrame:
    rows = []
    for b in FOCUS:
        c = ctx.cases[("CURR", b, "base")]
        er_a, er_b = c.dr.roe_star - c.r, _reg(ctx, "CURR", b, "ER_star", "bull")
        base_rating = _quote(ctx, "CURR", b)["rating"]
        for er in np.arange(round(min(er_a, 0.0) - 0.02, 3), er_b + 0.04, 0.005):
            for lam in (0.3, 0.45, 0.6, 0.75, 0.9, 0.99):
                q = _quote(ctx, "CURR", b, _with_er(c, float(er), lam))
                rows.append({"valuation": "CURR", "bank": T[b], "er_star": float(er), "lambda": lam,
                             "value_per_share": q["value_per_share"], "price": c.price,
                             "value_to_price": q["value_per_share"] / c.price, "rating": q["rating"],
                             "differs_from_base_rating": q["rating"] != base_rating, "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


# --- R2 and Part 3 item 4 -----------------------------------------------------------------------------------------------
def _coe_parts(ctx, val: str, b: str) -> tuple[float, float]:
    """(y_long - DS, ERP + CRP) from the register's COE_A evidence line (VALUATION_SPEC §5: COE = (y_long - DS) +
    beta x (ERP + CRP) + LP)."""
    import re

    r = ctx.reg
    ev = str(r[(r["valuation"] == val) & (r["bank"] == T[b]) & (r["symbol"] == "COE_A")]["evidence"].iloc[0])
    x = {k: float(re.search(rf"{k} ([0-9.]+)", ev).group(1)) for k in ("y_long", "DS", "ERP", "CRP")}
    return x["y_long"] - x["DS"], x["ERP"] + x["CRP"]


def bank_betas(root: Path) -> pd.DataFrame:
    from tzbank.chapters import ch14_market_data as md

    p = md.panel(root)
    mkt = md.market(root)
    rows = []
    for tick in ("NMB", "CRDB"):
        g = p[p["ticker"] == tick].set_index("trading_date").sort_index()
        r = g["lr_nonstale"]
        d = md.dimson(r, mkt.reindex(r.index), k=1)
        ra = g["lr_all"]
        mon = pd.concat([ra.resample("ME").sum().rename("y"), mkt.reindex(ra.index).resample("ME").sum().rename("x")],
                        axis=1).dropna()
        import statsmodels.api as sm

        f = sm.OLS(mon["y"], sm.add_constant(mon["x"])).fit()
        tv = pd.to_numeric(g["turnover"], errors="coerce")
        amihud = float((ra.abs() / (tv / 1e9)).replace([np.inf, -np.inf], np.nan).dropna().mean())
        rows += [{"bank": tick, "estimate": "monthly OLS beta vs market proxy ex banks", "beta": float(f.params["x"]),
                  "se": float(f.bse["x"]), "n": int(f.nobs), "evidence_class": "Estimated"},
                 {"bank": tick, "estimate": "daily Dimson beta (lags -1, 0, +1; non-stale days)",
                  "beta": d["beta_sum"], "se": d["beta_sum_se"], "n": d["n"], "evidence_class": "Estimated"},
                 {"bank": tick, "estimate": "Amihud illiquidity (|r| per TZS bn traded, mean)", "beta": np.nan,
                  "se": np.nan, "n": int(ra.notna().sum()), "amihud": amihud, "evidence_class": "Estimated"}]
    return pd.DataFrame(rows)


def bank_coe(ctx, betas: pd.DataFrame) -> pd.DataFrame:
    fin = _t(ctx.root, "ch13_financial_table.csv")
    ea = {b: float((x["equity"] / x["total_assets"]).iloc[0]) for b, x in
          fin[fin["fiscal_year"] == "FY2025A"].groupby("bank")}
    rows = []
    for val in DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            _, m = _coe_parts(ctx, val, b)
            beta0, lp0 = _reg(ctx, val, b, "beta"), _reg(ctx, val, b, "LP")
            rf = c.r - lp0 - beta0 * m  # exact: the register row reproduces the register COE
            asset_beta = beta0 * np.mean(list(ea.values()))
            ami = betas[betas["estimate"].str.startswith("Amihud")].set_index("bank")["amihud"]
            lp_scaled = float(np.clip(lp0 * ami[T[b]] / ami.mean(), 0.0, 0.03)) if ami.notna().all() else lp0
            cases = [("register (common)", beta0, lp0)]
            for est in ("monthly OLS", "daily Dimson"):
                x = betas[(betas["bank"] == T[b]) & betas["estimate"].str.startswith(est)].iloc[0]
                cases.append((f"{est} beta", float(x["beta"]), lp0))
            cases += [("re-levered common asset beta (E/A)", asset_beta / ea[T[b]], lp0),
                      ("register beta, Amihud-scaled LP (illustrative)", beta0, lp_scaled)]
            for name, beta, lp in cases:
                coe = rf + beta * m + lp
                q = _quote(ctx, val, b, r=coe) if coe > c.dr.g + 0.01 else {"value_per_share": np.nan,
                                                                           "rating": "n/a (COE - g < 1 pt)",
                                                                           "expected_return": np.nan}
                rows.append({"valuation": val, "bank": T[b], "basis": name, "beta": beta, "lp": lp, "coe": coe,
                             "coe_register": c.r, "value_per_share": q["value_per_share"], "price": c.price,
                             "value_to_price": q["value_per_share"] / c.price if q["value_per_share"] == q[
                                 "value_per_share"] else np.nan, "expected_return": q["expected_return"],
                             "rating": q["rating"], "evidence_class": "Estimated"})
            for name, coe in LOCAL_COE.items():
                q = _quote(ctx, val, b, r=coe)
                rows.append({"valuation": val, "bank": T[b], "basis": f"local broker COE: {name}", "beta": np.nan,
                             "lp": np.nan, "coe": coe, "coe_register": c.r, "value_per_share": q["value_per_share"],
                             "price": c.price, "value_to_price": q["value_per_share"] / c.price,
                             "expected_return": q["expected_return"], "rating": q["rating"],
                             "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def coe_interval(ctx, bank_coe_tab: pd.DataFrame) -> pd.DataFrame:
    """Cost of equity as an interval around the registered central value (prompt 24, 3c.6; VALUATION_SPEC §5), at
    31 Oct 2026, with the value and rating at each bound. Four sources of uncertainty, each a low / high COE:
    market estimation = the larger of half the approach A - B gap and beta x the change in the mature-market ERP
    between the two valuation dates; bank-specific systematic risk = COE on the bank's own daily Dimson and monthly
    OLS betas (``ch13_dec_bank_coe``); liquidity = the register's LP range (0 to 3%); country risk = beta x the change
    in the country risk premium between the two dates. Combined: the downward and upward deviations added in
    quadrature (independent sources), which is narrower than the sum."""
    import re

    rows = []
    val = "CURR"
    for b in FOCUS:
        c = ctx.cases[(val, b, "base")]
        mid = c.r
        beta = _reg(ctx, val, b, "beta")
        lp0 = _reg(ctx, val, b, "LP")
        ev = {d: str(ctx.reg[(ctx.reg["valuation"] == d) & (ctx.reg["bank"] == T[b]) & (ctx.reg["symbol"] == "COE_A")]
                     ["evidence"].iloc[0]) for d in DATES}
        par = {d: {k: float(re.search(rf"{k} ([0-9.]+)", e).group(1)) for k in ("ERP", "CRP")} for d, e in ev.items()}
        coe_a = _reg(ctx, val, b, "COE_A")
        cab = _t(ctx.root, "ch09_coe_ab.csv")
        coe_b = float(cab[(cab["valuation"] == val) & (cab["bank"] == T[b])]["coe_b"].iloc[0])
        h_mkt = max(abs(coe_a - coe_b) / 2, beta * abs(par["CURR"]["ERP"] - par["HIST"]["ERP"]))
        bc = bank_coe_tab[(bank_coe_tab["valuation"] == val) & (bank_coe_tab["bank"] == T[b])
                          & bank_coe_tab["basis"].isin(["daily Dimson beta", "monthly OLS beta"])]["coe"]
        h_cty = beta * abs(par["CURR"]["CRP"] - par["HIST"]["CRP"])
        comps = [("market estimation (approaches A and B; ERP between the two dates)", mid - h_mkt, mid + h_mkt),
                 ("bank-specific systematic risk (own Dimson and OLS betas)", min(float(bc.min()), mid),
                  max(float(bc.max()), mid)),
                 ("liquidity premium (register range 0-3%)", mid - lp0, mid + (0.03 - lp0)),
                 ("country risk premium (change between the two dates)", mid - h_cty, mid + h_cty)]
        lo = mid - float(np.sqrt(sum((mid - a) ** 2 for _, a, _ in comps)))
        hi = mid + float(np.sqrt(sum((z - mid) ** 2 for _, _, z in comps)))
        comps.append(("combined (deviations added in quadrature)", lo, hi))
        q0 = _quote(ctx, val, b, r=mid)
        for name, a, z in comps:
            ql = _quote(ctx, val, b, r=a) if a > c.dr.g + 0.01 else {"value_per_share": np.nan, "rating": "n/a"}
            qh = _quote(ctx, val, b, r=z)
            rows.append({"valuation": val, "bank": T[b], "source": name, "coe_central": mid, "coe_low": a,
                         "coe_high": z, "value_central": q0["value_per_share"], "value_at_low_coe": ql["value_per_share"],
                         "value_at_high_coe": qh["value_per_share"], "rating_at_low_coe": ql["rating"],
                         "rating_at_high_coe": qh["rating"], "price": c.price, "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- R5 -----------------------------------------------------------------------------------------------------------------
def hist_coe(root: Path) -> pd.DataFrame:
    from tzbank.chapters.ch05 import models as m5

    e = _t(root, "ch05_excess_returns.csv")
    e = e[e["in_estimation"].astype(bool)]
    const = float(_t(root, "ch09_coe_ab.csv").query("valuation == 'CURR'")["coe_used"].iloc[0])
    rows = []
    for b in FOCUS:
        g = e[e["bank_id"] == b].set_index("fiscal_year").sort_index()
        variants = {"time-varying COE (chapter 5 series)": g["roe"] - g["coe"],
                    f"constant COE {const:.1%} (today's)": g["roe"] - const,
                    "time-varying COE + 2 points": g["roe"] - g["coe"] - 0.02,
                    "time-varying COE - 2 points": g["roe"] - g["coe"] + 0.02}
        for name, er in variants.items():
            er = er.dropna()
            d = m5.pairs(er)
            dm = pd.DataFrame({"d_episode": d["year"].between(2015, 2018).astype(int).to_numpy()}, index=d.index)
            dd = pd.concat([d, dm], axis=1)
            full = m5.ar1(d)
            brk = m5.ar1(dd, ["d_episode"], [0.0])
            ch = m5.chow(d, d["year"].between(2015, 2018).to_numpy())
            ch["p"] = ch.get("p_value", np.nan)
            rows.append({"bank": T[b], "coe_series": name, "n": full["n"], "lambda_ols_full_cycle": full["lambda"],
                         "er_star_ols_full_cycle": full["er_star"], "mean_er": float(er.mean()),
                         "lambda_ols_break_adjusted": brk["lambda"], "er_star_ols_break_adjusted": brk["er_star"],
                         "chow_f_2015_18": ch.get("F", np.nan), "chow_p_2015_18": ch.get("p", np.nan),
                         "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- R4 / C10 -----------------------------------------------------------------------------------------------------------
def capital_policies(ctx) -> pd.DataFrame:
    from tzbank.banks.engine import project

    rows = []
    for b in FOCUS:
        c = ctx.cases[("CURR", b, "base")]
        req = c.dr.k_req if c.dr.k_req else 0.145
        pols = [("register base (management buffer, phased)", c.dr),
                ("1. regulatory minimum + 2.5-point conservation buffer", replace(c.dr, k_star=req, phase_in=False)),
                ("2. + management buffer (register rule), fully phased from year 1", replace(c.dr, phase_in=False)),
                ("3a. + illustrative D-SIB surcharge 1.0 point", replace(c.dr, k_star=c.dr.k_star + 0.01,
                                                                    phase_in=False)),
                ("3b. + illustrative D-SIB surcharge 2.0 points", replace(c.dr, k_star=c.dr.k_star + 0.02,
                                                                     phase_in=False))]
        for name, dr in pols:
            fc = project(c.op, dr).rows
            ex = fc[(fc["t"] >= 1) & fc["RWA"].notna()]
            q = _quote(ctx, "CURR", b, dr)
            n_cap = int((ex["Div"] >= dr.pi_max * ex["NP_attr"] * (1 - 1e-9)).sum())
            n_capital = int((ex["Div"] >= ex["cap_div"] * (1 - 1e-9)).sum())
            binding = (f"payout cap (pi_max = FY2025 payout) binds in {n_cap} of {len(ex)} explicit years; the capital "
                       f"target binds in {n_capital}")
            rows.append({"valuation": "CURR", "bank": T[b], "policy": name, "target_tier1": dr.k_star,
                         "payout_cap_pi_max": dr.pi_max, "binding": binding,
                         "requirement_incl_buffer": req, "loan_growth_avg": float((ex["L"] / ex["L"].shift(1)
                                                                                    ).dropna().mean() - 1)
                         if len(ex) > 1 else np.nan,
                         "payout_avg": float((ex["Div"] / ex["NP_attr"]).mean()),
                         "roe_year1": float(ex["ROE"].iloc[0]), "roe_last_explicit": float(ex["ROE"].iloc[-1]),
                         "value_per_share": q["value_per_share"], "price": c.price, "rating": q["rating"],
                         "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def surplus_drag_roles(root: Path) -> pd.DataFrame:
    s = _t(root, "ch13_surplus_drag.csv")
    role = {"requirement incl. conservation buffer": "headline (with the conservation buffer)",
            "regulatory minimum": "upper bound (counts all capital above the minimum as idle; no bank runs there)",
            "model": "lower bound (the model's own target; partly circular)"}
    s["role"] = [next((v for k, v in role.items() if k in str(h)), "other") for h in s["hurdle"]]
    return s[["bank", "fiscal_year", "hurdle", "role", "hurdle_ratio", "tier1_ratio", "headroom", "drag_tzs",
              "drag_share_of_profit"]].assign(evidence_class="Estimated")


# --- R9 / R10 ----------------------------------------------------------------------------------------------------------
def terminal_growth(ctx) -> pd.DataFrame:
    fin = _t(ctx.root, "ch13_financial_table.csv")
    rows = []
    for val in DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            b_ret = 1 - c.dr.payout_star
            pay = fin[(fin["bank"] == T[b]) & (fin["fiscal_year"] == "FY2025A")]["payout"]
            rows.append({"valuation": val, "bank": T[b], "roe_star": c.dr.roe_star, "payout_star": c.dr.payout_star,
                         "retention_b": b_ret, "g_implied_b_x_roe": b_ret * c.dr.roe_star, "g_register": c.dr.g,
                         "difference": b_ret * c.dr.roe_star - c.dr.g,
                         "fy2025_payout": float(pay.iloc[0]) if len(pay) else np.nan,
                         "reading": "g = b x ROE* holds by construction (payout* = 1 - g / ROE*); the check is "
                                    "whether that payout is plausible against FY2025 and the capital rule",
                         "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def price_requires(ctx) -> pd.DataFrame:
    br = _t(ctx.root, "ch13_roe_sustainable_bridge.csv")
    imp = _t(ctx.root, "ch10_implied.csv")
    rows = []
    for b in ("NMB", "CRDB"):
        g = br[br["bank"] == b]
        for r in g.itertuples():
            rows.append({"bank": b, "step_no": int(r.step_no), "step": r.step, "roe_pts": r.roe_pts,
                         "source": "ch13_roe_sustainable_bridge", "evidence_class": "Scenario"})
        x = imp[(imp["bank"] == b) & (imp["valuation"] == "CURR")]
        if len(x):
            req = float(x["implied_roe_star"].iloc[0])
            rows.append({"bank": b, "step_no": 99, "step": "ROE* the 31 Oct 2026 price requires (chapter 10)",
                         "roe_pts": req, "source": "ch10_implied", "evidence_class": "Estimated"})
        for s in ("fee-income advantage narrowing", "deposit competition (repricing test)"):
            rows.append({"bank": b, "step_no": 98, "step": f"{s}: not a separate engine driver; see ch13_rate_"
                         "sensitivity_repricing / ch04 fee tables", "roe_pts": np.nan, "source": "not quantified",
                         "evidence_class": "Interpretation"})
    return pd.DataFrame(rows)


def implied_growth_table(root: Path) -> pd.DataFrame:
    g = _t(root, "ch13_text_implied_growth.csv")
    g["admissible_solution"] = np.where(g["status"].str.startswith("solved"), "yes", "no")
    g["value_at_guard_vs_price"] = g["value_at_guard_g"] / g["price"]
    return g[["valuation", "date", "bank", "price", "coe", "guard_g", "value_at_guard_g", "price_post_split",
              "value_at_guard_g_post_split", "value_at_guard_vs_price", "admissible_solution", "status"]].assign(
        evidence_class="Scenario")


# --- C2 / C4 ------------------------------------------------------------------------------------------------------------
def rate_roles(root: Path) -> pd.DataFrame:
    rt = _t(root, "ch06_rates_used.csv").set_index("year")
    fb = _t(root, "ch13_franchise_breakeven.csv") if (path("tables", root) / "ch13_franchise_breakeven.csv").exists() \
        else pd.DataFrame()
    rows = [{"rate": "BoT overall T-bill rate, 2025 annual mean", "value": float(rt.loc[2025, "tbill"]),
             "date": "2025 (calendar-year mean of quarter-ends)", "source": "ch06_rates_used (BoT)",
             "role": "historical funding comparisons and the FY2025 yield-minus-bill point (ch06, ch14 / B5)",
             "chapters": "6, 14"},
            {"rate": "364-day T-bill, latest auction", "value": 0.062852, "date": "23 Sep 2026 auction",
             "source": "register v1.3 'rate' (D64)", "role": "forward rate path at 31 Oct 2026; break-even "
             "comparisons (thesis monitor r0, r5)", "chapters": "6, 9-11, 13"}]
    fb = fb.drop_duplicates(["bank", "fiscal_year", "breakeven_tbill"]) if len(fb) else fb
    for r in fb.itertuples() if len(fb) else []:
        rows.append({"rate": f"net deposit franchise break-even bill rate, {r.bank} FY{r.fiscal_year}",
                     "value": float(r.breakeven_tbill), "date": f"FY{r.fiscal_year}",
                     "source": "ch13_franchise_breakeven", "role": "the bill rate at which the net deposit "
                     "franchise contribution turns zero (thesis monitor r0/r5)", "chapters": "6, 13"})
    rows.append({"rate": "6.65%", "value": np.nan, "date": "", "source": "not found in reports/source, the built "
                 "reports, the owner's drafts, the register or any pipeline table", "role": "no role: not printed",
                 "chapters": ""})
    return pd.DataFrame(rows).assign(evidence_class="Observed")


def rc_curr(root: Path) -> pd.DataFrame:
    r = _t(root, "ch13_rate_consistent.csv")
    r = r[r["valuation"] == "CURR"].copy()
    r["what_changes"] = r["mode"].map({
        "rate path only": "the T-bill path; NIM moves only through the engine's rate pass-through "
                          "(asset beta - deposit beta x D/IEA, open issue 11)",
        "COE only": "the cost of equity only (discount rate); the T-bill path is unchanged",
        "rate path and COE together": "the T-bill path and the COE by the same amount"})
    r["held_fixed"] = "all other register v1.3 drivers (growth, CIR, cost of risk, COE, ROE*, lambda, g)"
    return r[["scenario_id", "scenario_label", "bank", "mode", "shift", "what_changes", "held_fixed",
              "value_per_share", "base_value", "value_change", "rating"]].assign(evidence_class="Scenario")


# --- C5 / C11 -----------------------------------------------------------------------------------------------------------
def event_robust(root: Path, reps: int = 2000) -> pd.DataFrame:
    from scipy import stats

    e = _t(root, "ch14_results_events.csv")
    e = e[e["include"].astype(bool) & (e["role"] == "event bank")]
    rng = np.random.default_rng(int(settings(root)["random_seed"]) + 18)
    rows = []
    for (b, wdw), g in e.groupby(["event_bank", "window"]):
        x = g["car_market"].dropna().to_numpy()
        if len(x) == 0:
            continue
        pos = int((x > 0).sum())
        sign_p = float(stats.binomtest(pos, len(x), 0.5).pvalue) if len(x) else np.nan
        boot = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
        rows.append({"bank": b, "window": wdw, "events": len(x), "mean_car": float(x.mean()),
                     "median_car": float(np.median(x)), "positive": pos, "sign_test_p": sign_p,
                     "bootstrap_ci_lo": float(np.quantile(boot, 0.05)), "bootstrap_ci_hi": float(np.quantile(boot, 0.95)),
                     "reading": "exploratory: few events and thin trading; a CI spanning zero means no reliable "
                                "average effect", "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def outliers(root: Path) -> pd.DataFrame:
    from tzbank.chapters.ch05.persistence import EXCLUDE_SHOWN

    d = _t(root, "ch04_dupont_panel.csv")
    name = {"B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB", "B07": "MUCOBA", "B01": "CRDB", "B02": "NMB"}
    rows = []
    for (b, y), why in EXCLUDE_SHOWN.items():
        y = int(y)
        x = d[(d["bank_id"] == b) & (d["fiscal_year"] == y)]
        roe = float(x["roe"].iloc[0]) if len(x) else np.nan
        yr = d[d["fiscal_year"] == y]["roe"]
        rows.append({"bank": name.get(b, b), "fiscal_year": y, "roe": roe, "reason": why,
                     "panel_mean_roe_with": float(yr.mean()), "panel_mean_roe_without": float(
                         d[(d["fiscal_year"] == y) & ~(d["bank_id"] == b)]["roe"].mean()),
                     "affects_nmb_crdb_findings": "no (NMB and CRDB rows unchanged)", "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- C13 / C14 ----------------------------------------------------------------------------------------------------------
# review thresholds of the monitoring table (prompt 23, D92): a level-based trigger sits one standard deviation of the
# FY2010-25 history beyond the mean (so a bank already below its mean does not trigger on publication day), and NIM
# sits 0.5 point below the FY2026 forecast path. Others are fixed levels or the thesis-break thresholds of chapter 13.
MON_SD = 1.0
MON_NIM_MARGIN = 0.005


def _fmt_mon(x: float, kind: str, signed: bool = False) -> str:
    """Display text of a monitoring value: ratios in percent, the deposit beta as a coefficient (0.07, prompt 23)."""
    if pd.isna(x):
        return "–"
    if kind == "coef":
        return (f"{x:+.2f}" if signed else f"{x:.2f}").replace("-", "−")
    v = x * 100
    return (f"{v:+.1f}%" if signed else f"{v:.1f}%").replace("-", "−")


def monitoring(root: Path) -> pd.DataFrame:
    """C13 monitoring table (Blueprint ch. 13): what would change the view, by area, with the current value, the
    review threshold and the distance; ``status`` is 'triggered' when the threshold is already crossed."""
    tb = _t(root, "ch13_text_thesis_breaks.csv")
    fin = _t(root, "ch13_financial_table.csv")
    d4 = _t(root, "ch04_dupont_panel.csv")
    wide = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    tot = {T[b]: float(wide[(wide["bank_id"] == b) & (wide["fiscal_year"] == 2025)]["total_capital_ratio"].iloc[0])
           for b in FOCUS if len(wide[(wide["bank_id"] == b) & (wide["fiscal_year"] == 2025)])}
    k = MON_SD
    sd_txt = f"{k:g} sd" if k != 1 else "one sd"
    rows = []
    for b in ("NMB", "CRDB"):
        f = fin[(fin["bank"] == b) & (fin["fiscal_year"] == "FY2025A")].iloc[0]
        f26 = fin[(fin["bank"] == b) & (fin["fiscal_year"] == "FY2026E")]
        nim_path = float(f26["NIM"].iloc[0]) if len(f26) else float(f["NIM"])
        h = d4[(d4["bank_id"] == BID[b]) & d4["fiscal_year"].between(2010, 2025)]
        tt = tb[tb["bank"] == b]

        def br(key, tt=tt):
            x = tt[tt["indicator"].str.startswith(key)]
            return x.iloc[0] if len(x) else None

        def lo(col, h=h):
            return float(h[col].mean() - k * h[col].std())

        def hi(col, h=h):
            return float(h[col].mean() + k * h[col].std())

        cor, cap, bill, beta, pv = (br("cost of risk"), br("binding capital"), br("T-bill"), br("deposit beta"),
                                    br("price vs"))
        # (area, indicator, current, threshold, direction (+1: triggers above, -1: below, 0: none), rule, why, kind)
        items = [
            ("operating performance", "ROE", f["ROE"], lo("roe"), -1,
             f"below the FY2010-25 mean less {sd_txt}", "the rent the price pays for", "pct"),
            ("operating performance", "ROA", float(h["roa"].iloc[-1]), lo("roa"), -1,
             f"below the FY2010-25 mean less {sd_txt}", "earning power per asset", "pct"),
            ("operating performance", "NIM", f["NIM"], nim_path - MON_NIM_MARGIN, -1,
             f"more than {MON_NIM_MARGIN * 100:.1f} point below the FY2026 forecast", "the margin the bill rate supports",
             "pct"),
            ("operating performance", "non-interest income share (fees)", float(h["non_interest_income_share"].iloc[-1]),
             lo("non_interest_income_share"), -1, f"below the FY2010-25 mean less {sd_txt}", "fee franchise", "pct"),
            ("operating performance", "cost-to-income", f["CIR"], hi("cost_to_income"), 1,
             f"above the FY2010-25 mean plus {sd_txt}", "cost efficiency", "pct"),
            ("credit quality", "NPL ratio", f["NPL"], 0.05, 1, "above 5% (BoT tolerance)", "asset quality", "pct"),
            ("credit quality", "cost of risk", cor["current"] if cor is not None else f["CoR"],
             cor["threshold"] if cor is not None else np.nan, 1, "above the through-cycle mean",
             "credit costs normalising", "pct"),
            ("credit quality", "NPL coverage", f["NPL_coverage"], 1.0 if b == "NMB" else 0.6, -1,
             "below 100%" if b == "NMB" else "below 60%", "provision cushion", "pct"),
            ("capital", CAP_LABEL[b][0], f["Tier1"], np.nan, 0, "see headroom", "loss-absorbing capital", "pct"),
            ("capital", CAP_LABEL[b][1], tot.get(b, np.nan), np.nan, 0, "see headroom", "total regulatory capital",
             "pct"),
            ("capital", "payout ratio FY2025", f["payout"], np.nan, 0, "cut below the FY2025 ratio",
             "payout capacity the capital rule allows", "pct"),
            ("capital", "binding capital headroom", cap["current"] if cap is not None else np.nan, 0.02, -1,
             "below 2 points", "room to grow and pay", "pct"),
            ("funding", "T-bill vs franchise break-even", bill["current"] if bill is not None else np.nan,
             bill["threshold"] if bill is not None else np.nan, -1, "bill below break-even", "deposit franchise value",
             "pct"),
            ("funding", "deposit beta (upper 90% bound)", beta["current"] if beta is not None else np.nan, 0.10, 1,
             "above 0.10: depositors start to reprice", "deposit inertia", "coef"),
            ("valuation", "price / updated base value", pv["current"] / pv["threshold"] if pv is not None else np.nan,
             1.0, -1, "price below base value", "the Sell call itself", "pct")]
        thesis = {"T-bill vs franchise break-even": 0, "cost of risk": 1, "deposit beta (upper 90% bound)": 2,
                  "binding capital headroom": 3, "price / updated base value": 4}
        for area, ind, cur, thr, side, rule, why, kind in items:
            lab = thesis.get(ind)
            dist = cur - thr if pd.notna(cur) and pd.notna(thr) else np.nan
            if side == 0 or pd.isna(dist):
                status = ""
            else:
                status = "triggered" if (dist > 0 if side > 0 else dist < 0) else "clear"
            rows.append({"bank": b, "monitor_row": f"r{lab + (0 if b == 'NMB' else 5)}" if lab is not None else "",
                         "area": area, "indicator": ind, "current": cur, "threshold": thr, "distance": dist,
                         "current_display": _fmt_mon(cur, kind), "threshold_display": _fmt_mon(thr, kind),
                         "distance_display": _fmt_mon(dist, kind, signed=True), "trigger": rule,
                         "why_it_matters": why, "status": status,
                         "threshold_basis": "author's review thresholds, not regulatory limits (prompt 23: one sd "
                                            "beyond the FY2010-25 mean; NIM 0.5 point below the FY2026 forecast)",
                         "caption": "triggers are reasons to review the thesis, not automatic rating changes",
                         "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def income_vs_return(root: Path) -> pd.DataFrame:
    ann = _t(root, "ch12_returns_annual.csv")
    rt = _t(root, "ch06_rates_used.csv").set_index("year")["tbill"]
    rows = []
    for b in ("NMB", "CRDB"):
        g = ann[(ann["bank"] == b) & ann["period"].astype(str).str.fullmatch(r"\d{4}")].copy()
        g["year"] = g["period"].astype(int)
        g = g.set_index("year").sort_index()
        last = int(g.index.max())
        for start in range(int(g.index.min()), last):
            yrs = [y for y in range(start + 1, last + 1)]
            if not all(y in g.index and y in rt.index for y in yrs):
                continue
            tsr = float(np.prod(1 + g.loc[yrs, "tzs_return"])) - 1
            bill = float(np.prod(1 + rt.loc[yrs])) - 1
            n = len(yrs)
            rows.append({"bank": b, "holding_period": f"Dec {start} - Dec {last}", "years": n, "tsr": tsr,
                         "rolled_tbill_return": bill, "tsr_annualised": (1 + tsr) ** (1 / n) - 1,
                         "tbill_annualised": (1 + bill) ** (1 / n) - 1, "excess_annualised":
                         (1 + tsr) ** (1 / n) - (1 + bill) ** (1 / n),
                         "basis": "TSR: year-end prices plus dividends (ch12; NMB split- and CRDB rights-adjusted); "
                                  "bills: annual BoT overall rate compounded (ch06; auction yields not held as a series)",
                         "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def income_figure(root: Path, iv: pd.DataFrame, figd: Path) -> Path:
    ym = _t(root, "ch13_text_yield_minus_tbill.csv")
    fig, (a1, a2) = plt_subplots(1, 2, figsize=(11, 3.8))
    for b in ("NMB", "CRDB"):
        g = ym[ym["bank"] == b].sort_values("fiscal_year")
        a1.plot(g["fiscal_year"], g["dividend_yield"] * 100, color=COLORS[BID[b]], lw=2, label=f"{b} dividend yield")
        h = iv[iv["bank"] == b].sort_values("years")
        a2.plot(h["years"], h["excess_annualised"] * 100, color=COLORS[BID[b]], lw=2, marker="o", ms=3,
                label=f"{b} TSR less rolled T-bills")
    g = ym[ym["bank"] == "NMB"].sort_values("fiscal_year")
    a1.plot(g["fiscal_year"], g["tbill"] * 100, color="#5a5a55", ls="--", lw=1, label="T-bill (annual)")
    style(a1, "A. Income: dividend yield against the T-bill yield", "percent")
    a1.legend(fontsize=6.5, frameon=False)
    a2.axhline(0, color="#1f1f1d", lw=0.6)
    style(a2, "B. Return: TSR less rolled T-bills, to Dec 2025", "percentage points a year")
    a2.set_xlabel("holding period, years", fontsize=8)
    a2.legend(fontsize=6.5, frameon=False)
    return save(fig, figd / "ch13_dec_income_vs_return.png", _foot(
        "ch12 annual returns; ch06 BoT T-bill rates; an income comparison is not a risk-adjusted verdict", "Derived"))


# --- R7 -----------------------------------------------------------------------------------------------------------------
def peer_structure(ctx) -> pd.DataFrame:
    pm = _t(ctx.root, "ch13_peer_market.csv")
    pm = pm[pm["valuation"] == "CURR"]
    pk = _t(ctx.root, "ch13_peer_kpi.csv").sort_values("fiscal_year").groupby("peer").last()
    k = _t(ctx.root, "ch13_kpi_table.csv")
    k = k[pd.to_numeric(k["fiscal_year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").notna()]
    k = k.assign(fy=k["fiscal_year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)).sort_values("fy")
    k = k.groupby("bank").last()
    rows = []
    for r in pm.itertuples():
        tz = r.country == "Tanzania"
        g = ctx.cases[("CURR", BID[r.bank], "base")].dr.g if tz else PEER_G
        coe = r.coe_approach_a
        jpb = (r.roe - g) / (coe - g) if pd.notna(coe) and coe > g else np.nan
        src = k if tz else pk
        x = src.loc[r.bank] if r.bank in src.index else pd.Series(dtype=float)
        rows.append({"bank": r.bank, "country": r.country, "pb": r.pb, "roe": r.roe, "coe": coe,
                     "roe_minus_coe": r.roe - coe if pd.notna(coe) else np.nan,
                     "equity_assets": x.get("equity_assets", np.nan) if not tz else np.nan,
                     "tier1_ratio": x.get("tier1_ratio", np.nan) if tz else np.nan,
                     "loan_growth": x.get("net_loan_growth", np.nan) if not tz else np.nan,
                     "g_assumed": g, "justified_pb": jpb, "pb_minus_justified": r.pb - jpb if pd.notna(jpb) else np.nan,
                     "pb_basis": r.pb_basis, "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


# --- Part 3: local analyst reports --------------------------------------------------------------------------------------
REGISTER = Path("data/raw/analyst_reports/register.csv")
CAP_LABEL = {"NMB": ("CET1 ratio (NMB label; equal to tier 1)", "total capital ratio"),
             "CRDB": ("core capital ratio (CRDB label)", "total capital ratio")}


def analyst_compare(ctx) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(ctx.root / REGISTER)
    reg = reg[reg["bank"].isin(["NMB", "CRDB"]) & reg["fair_value_post_split"].notna()]  # D88: views on the two banks
    reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
    reg["target_post_split"] = reg["fair_value_post_split"]
    rows = []
    for r in reg.itertuples():
        rows.append({"source": r.broker, "date": r.date.date().isoformat(), "date_label": r.date_label, "bank": r.bank, "rating": r.rating,
                     "value_or_target": r.target_post_split, "price_at_date": r.price_post_split,
                     "value_to_price": r.target_post_split / r.price_post_split if pd.notna(r.price_post_split)
                     else np.nan, "method": r.method, "coe": r.coe, "terminal_g": r.terminal_g,
                     "evidence_class": "Observed"})
    v = _t(ctx.root, "ch09_valuations.csv")
    for x in v.itertuples():
        rows.append({"source": "this study", "date": x.date, "date_label": x.date, "bank": x.bank, "rating": x.rating,
                     "value_or_target": x.value_per_share_post_split_basis, "price_at_date": x.price_post_split_basis,
                     "value_to_price": x.value_per_share_post_split_basis / x.price_post_split_basis,
                     "method": "RIM 60% + DDM 40% (register v1.3)", "coe": x.coe, "terminal_g": np.nan,
                     "evidence_class": "Estimated"})
    cmp_ = pd.DataFrame(rows)
    arows = []
    for bk in ("NMB", "CRDB"):
        x = reg[(reg["bank"] == bk) & reg["price_post_split"].notna() & reg["fair_value_post_split"].notna()]
        x = x.sort_values("date")
        ratio = x["fair_value_post_split"] / x["price_post_split"]
        arows.append({"bank": bk, "notes_with_price": len(x), "notes_total": int((reg["bank"] == bk).sum()),
                      "first_note": x["date_label"].iloc[0], "last_note": x["date_label"].iloc[-1],
                      "corr_target_price": float(x["fair_value_post_split"].corr(x["price_post_split"]))
                      if len(x) > 2 else np.nan,
                      "log_target_change": float(np.log(x["fair_value_post_split"].iloc[-1]
                                                        / x["fair_value_post_split"].iloc[0])),
                      "log_price_change": float(np.log(x["price_post_split"].iloc[-1] / x["price_post_split"].iloc[0])),
                      "target_over_price_min": float(ratio.min()), "target_over_price_max": float(ratio.max()),
                      "reading": "exploratory: too few notes for a test; targets moved with the price",
                      "evidence_class": "Derived"})
    anch = pd.DataFrame(arows)
    last = {"NMB": 1820.0, "CRDB": 2840.0}
    rec = reg.sort_values("date").groupby("bank").tail(1)
    direc = pd.DataFrame([{"bank": r.bank, "broker": r.broker, "note_date": r.date_label, "broker_value": r.fair_value_post_split,
                           "price_31oct2026": last[r.bank], "broker_below_price": r.fair_value_post_split < last[r.bank],
                           "our_value_31oct2026": float(cmp_[(cmp_["source"] == "this study") & (cmp_["bank"] == r.bank)
                                                             & (cmp_["date"] == "2026-10-31")]["value_or_target"].iloc[0]),
                           "reading": "agreement in direction only, not validation of our level",
                           "evidence_class": "Derived"} for r in rec.itertuples()])
    return cmp_, anch, direc


def analyst_figure(cmp_: pd.DataFrame, figd: Path) -> Path:
    fig, axs = plt_subplots(1, 2, figsize=(10, 3.8))
    for ax, b in zip(axs, ("NMB", "CRDB"), strict=True):
        g = cmp_[cmp_["bank"] == b].copy()
        g["d"] = pd.to_datetime(g["date"], errors="coerce")
        g = g.dropna(subset=["d"]).sort_values("d")
        br = g[g["source"] != "this study"]
        us = g[g["source"] == "this study"]
        ax.scatter(br["d"], br["value_or_target"], color="#8c8c86", s=22, label="local broker fair value / target")
        ax.scatter(us["d"], us["value_or_target"], color=COLORS[BID[b]], marker="D", s=28, label="this study")
        pr = g.dropna(subset=["price_at_date"])
        ax.plot(pr["d"], pr["price_at_date"], color="#1f1f1d", lw=0.8, ls=":", marker="x", ms=4, label="price at date")
        style(ax, f"{b}: broker views, ours and the price (post-split TZS)", "TZS per share")
        ax.legend(fontsize=6.5, frameon=False)
        ax.tick_params(axis="x", rotation=30)
    return save(fig, figd / "ch13_dec_analyst_compare.png", _foot(
        "data/raw/analyst_reports/register.csv (owner-collected broker notes; NMB pre-split values / 10); ch09",
        "Observed; ours Estimated"))


NOTE = """# Owner decisions on contested analytical choices (prompt 18)

Generated by `tzbank.chapters.ch13_decisions` (stage `market_ext`, after chapter 14). The registered rating rule (D4),
the base-case inputs and the Sell ratings are unchanged; each table shows how a conclusion depends on one choice.

## Research question
How much do the valuation conclusions (chapters 9-12) and the evidence chapters (3-8, 14) depend on contested
choices: the 2015-18 interpretation, the parameter pairing, the cost of equity, the capital policy?

## Variable definitions
`weight_on_B`: weight on the exceptional-episode reading of 2015-18 (break-adjusted ER*), not a crisis probability.
`crossing_weight`: the weight at which value equals the 31 Oct 2026 price. Other columns: module docstring and
`docs/VARIABLE_DICTIONARY.md`.

## Sources
Register v1.3 and the valuation engine (`ch13_report_support.quote`); ch05 persistence tables; ch11 Monte Carlo;
DSE library daily data (chapter 14); ch06 rates; ch12 returns; ch13 peer tables; data/raw/analyst_reports/register.csv.

## Validation test
`tests/test_decisions.py`.

## Results
{results}

## Evidence class
Scenario (re-valuations), Estimated (betas, Monte Carlo, persistence), Derived (reshaped tables), Observed (broker
register).

## Limitations
- Interpretations A and B differ only in ER*; both use the break-adjusted fade speed, because the full-cycle lambda
  (1.00) identifies no mean reversion (configuration 1 shows what it would imply).
- Betas rest on thin, stale-affected trading; the Dimson correction helps but standard errors are wide. The
  Amihud-scaled liquidity premium is illustrative, not estimated.
- Capital policies use the engine's target-ratio rule; the D-SIB surcharges are illustrative (none is disclosed).
- The broker register is small (six notes) and partly transcribed by the owner; the anchoring reading is exploratory.
"""


def write_decisions(ctx) -> list[Path]:
    root = ctx.root
    tab, figd = path("tables", root), path("figures", root)
    w, cross = interp_weights(ctx)
    betas = bank_betas(root)
    tables = [
        (w, "ch13_dec_interp_weights.csv", "C3. Value, value/price and rating by weight on the exceptional reading",
         "Scenario"),
        (cross, "ch13_dec_crossing_weights.csv", "C3. Price-crossing weights", "Scenario"),
        (mc_conditional(ctx), "ch13_dec_mc_conditional.csv", "R6. P(value > price) given A, given B and mixed",
         "Estimated"),
        (mc_inputs(ctx), "ch13_dec_mc_inputs.csv", "R6. Distributions of the drawn inputs (quantiles)", "Estimated"),
        (rating_application(ctx), "ch13_dec_rating_rule.csv", "C6/C7. The rating rule applied, with COE switch points",
         "Scenario"),
        (configurations(ctx), "ch13_dec_configurations.csv", "R3. Three internally consistent configurations",
         "Scenario"),
        (joint_grid(ctx), "ch13_dec_joint_grid.csv", "R3. ER* x lambda grid with the rating", "Scenario"),
        (betas, "ch13_dec_bank_betas.csv", "R2. Bank betas and illiquidity", "Estimated"),
        (bcoe := bank_coe(ctx, betas), "ch13_dec_bank_coe.csv", "R2 / Part 3. COE, value and rating by COE basis",
         "Estimated"),
        (coe_interval(ctx, bcoe), "ch13_dec_coe_interval.csv", "Cost of equity as an interval, value across the range "
         "(prompt 24)", "Estimated"),
        (hist_coe(root), "ch13_dec_hist_coe.csv", "R5. Persistence under three COE constructions", "Estimated"),
        (capital_policies(ctx), "ch13_dec_capital_policies.csv", "R4. Capital policies", "Scenario"),
        (surplus_drag_roles(root), "ch13_dec_surplus_drag.csv", "C10. Surplus-capital drag: headline and bounds",
         "Estimated"),
        (terminal_growth(ctx), "ch13_dec_terminal_growth.csv", "R9. Terminal growth cross-check g = b x ROE*",
         "Scenario"),
        (price_requires(ctx), "ch13_dec_price_requires.csv", "R10. Sustainable-ROE bridge and the ROE the price "
         "requires", "Scenario"),
        (implied_growth_table(root), "ch13_dec_implied_growth.csv", "R10. Implied growth under the 4-point guard",
         "Scenario"),
        (rate_roles(root), "ch13_dec_rate_roles.csv", "C2. Every rate has one job", "Observed"),
        (rc_curr(root), "ch13_dec_rc_curr.csv", "C4. RC-CURR rows traced to their inputs", "Scenario"),
        (event_robust(root), "ch13_dec_event_robust.csv", "C5. Event CARs: sign test and bootstrap", "Estimated"),
        (outliers(root), "ch13_dec_outliers.csv", "C11. Excluded observations and their effect", "Derived"),
        (monitoring(root), "ch13_dec_monitoring.csv", "C13. Monitoring table by area", "Derived"),
        (income_vs_return(root), "ch13_dec_income_vs_return.csv", "C14. TSR against rolled T-bills", "Derived"),
        (peer_structure(ctx), "ch13_dec_peer_structure.csv", "R7. Structured peer table with justified P/B",
         "Scenario")]
    if (root / REGISTER).exists():
        cmp_, anch, direc = analyst_compare(ctx)
        tables += [(cmp_, "ch13_dec_analyst_compare.csv", "Part 3. Broker views, ours and the price", "Observed"),
                   (anch, "ch13_dec_analyst_anchoring.csv", "Part 3. Do broker targets follow price?", "Derived"),
                   (direc, "ch13_dec_analyst_direction.csv", "Part 3. Direction check at 31 Oct 2026 prices",
                    "Derived")]
    nf = path("notes", root) / "ch13_dec.md"
    files, man_t, man_f = [], [], []
    for df, name, desc, ev in tables:
        f = write_csv(ctx.stamp(df), tab / name, require_evidence=True)
        files.append(f)
        man_t.append({"path": f.relative_to(root).as_posix(), "description": desc, "evidence_class": ev,
                      "note": nf.relative_to(root).as_posix()})
    t = {name: df for df, name, _, _ in tables}
    png, mcd = mc_inputs_figure(ctx, figd)
    figs = [(weights_figure(t["ch13_dec_interp_weights.csv"], figd), t["ch13_dec_interp_weights.csv"],
             "C3. Value against the weight on the exceptional reading", "Scenario"),
            (png, mcd, "R6. Distributions of the key drawn inputs", "Estimated"),
            (income_figure(root, t["ch13_dec_income_vs_return.csv"], figd), t["ch13_dec_income_vs_return.csv"],
             "C14. Income and return", "Derived")]
    if "ch13_dec_analyst_compare.csv" in t:
        figs.append((analyst_figure(t["ch13_dec_analyst_compare.csv"], figd), t["ch13_dec_analyst_compare.csv"],
                     "Part 3. Broker views, ours and the price", "Observed"))
    for p, data, desc, ev in figs:
        d = write_csv(data, p.with_name(f"{p.stem}_data.csv"), require_evidence=True)
        files += [p, d]
        man_f.append({"path": p.relative_to(root).as_posix(), "data": d.relative_to(root).as_posix(),
                      "description": desc, "evidence_class": ev, "note": nf.relative_to(root).as_posix()})
    cw = t["ch13_dec_crossing_weights.csv"]
    res = ["- C3 crossing weights: " + "; ".join(
        f"{r.bank} {r.valuation}: {r.crossing_weight:.0%}" if pd.notna(r.crossing_weight) else
        f"{r.bank} {r.valuation}: {r.status}" for r in cw.itertuples())]
    bc = t["ch13_dec_bank_coe.csv"]
    cur = bc[(bc["valuation"] == "CURR") & ~bc["basis"].str.startswith("local")]
    res.append("- R2 ratings under bank-specific COE (31 Oct 2026): " + "; ".join(
        f"{r.bank} {r.basis}: COE {r.coe:.1%}, {r.rating}" for r in cur.itertuples()))
    files.append(write_text(NOTE.format(results="\n".join(res)), nf))
    m = {"chapter": "13_dec", "title": "Owner decisions on contested analytical choices (prompt 18)",
         "status": "complete", "hypotheses": [], "tables": man_t, "figures": man_f,
         "notes": [nf.relative_to(root).as_posix()], "inputs": ["config/assumption_register.csv",
                                                                "outputs/tables/ch05_calibration_candidates.csv",
                                                                "outputs/tables/ch11_monte_carlo.csv",
                                                                str(REGISTER)],
         "module": "tzbank.chapters.ch13_decisions", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_dec.json"))
    return files
