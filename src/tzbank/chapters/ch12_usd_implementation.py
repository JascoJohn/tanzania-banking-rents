"""Chapter 12: USD returns and implementation (blueprint §4 chapter 12; H12.1, H12.2; prompt 11; D60 proposed).

1. USD returns. Historical: annual TZS total return t = (P_t + D_t) / P_{t-1} - 1 with year-end prices and D_t the
   dividend paid in year t (the FY t-1 dividend, special included); USD = (1 + r_TZS) x FX_{t-1} / FX_t - 1 with the
   BoT end-December TZS/USD (``data/interim/bot_fx_eop.csv``) and the workbook rates at the valuation dates. Years
   without both FX end points are not translated; spans chain the TZS returns between held FX end points. Expected
   (prompt 24 rebuild): E[r_USD] = (1 + price return to the 12-month target) / (1 + d) + sum_i (DPS_i / P0)(1 - wht)
   / (1 + d)^t_i - 1, each dividend translated on its payment date t_i, for depreciation scenarios d (none; relative
   PPP from the inflation differential; the BoT history CAGR to the date; stress = the worst held calendar year); the
   round-trip trading cost is a separate line. Break-even depreciation d*: E[r_USD](d*) = h, the USD hurdle = USD
   CAPM with the same
   components as the TZS cost of equity: rf_US + beta (ERP + CRP) + LP (beta and LP from the assumption register,
   ERP and CRP from the inputs workbook at the date; owner, ch12 review A, issue 33 resolved). Dividends are net of
   withholding tax for a foreign holder (`ch12.div_wht_foreign`; owner, ch12 review B).
2. Capacity: days = position / (participation x ADV), ADV = mean daily traded value over every exchange day in the
   daily data up to the valuation date (zero-trade days count as zero; flagged block days excluded; with blocks as a
   sensitivity). Zero-trade days and block-trade evidence (data flags; days above `block_multiple` x the median).
3. Trading cost per side = DSE fee schedule on the daily trade value (tiered brokerage + VAT + market fees, capped)
   + Amihud impact ILLIQ x daily trade value, ILLIQ = mean |r_d| / traded value_d over non-block traded days.
4. Market access: desk-research table (`config/market_access.csv`), every row with a source and access date, marked
   for counsel review; no legal conclusions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, pct, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import rel, write_csv, write_json, write_text

BANKS = {"B01": "CRDB", "B02": "NMB"}
DATES = {"HIST": "valuation_dates.historical", "CURR": "valuation_dates.current"}


def _cfg(root: Path) -> dict:
    return settings(root)["ch12"]


def vdates(root: Path) -> dict[str, pd.Timestamp]:
    v = settings(root)["valuation_dates"]
    return {"HIST": pd.Timestamp(v["historical"]), "CURR": pd.Timestamp(v["current"])}


def macro_at(root: Path, vd: pd.Timestamp, name: str) -> tuple[float, str]:
    """Value of a macro variable as at the valuation date (inputs workbook; point in time by construction)."""
    m = pd.read_csv(path("processed", root) / "macro.csv", low_memory=False)
    r = m[(m["canonical_name"] == name) & (pd.to_datetime(m["As At (Valuation Date)"]) == vd)]
    if r.empty or pd.isna(r["value_numeric"].iloc[0]):
        return float("nan"), ""
    return float(r["value_numeric"].iloc[0]), str(r["Observation / Auction Date"].iloc[0])


# --- FX and returns ---------------------------------------------------------------------------------------------------
def fx_table(root: Path, vd: pd.Timestamp | None = None) -> pd.DataFrame:
    """End-December TZS/USD (BoT reports, published by `vd`) plus the workbook rate at each valuation date <= vd."""
    f = pd.read_csv(path("interim", root) / "bot_fx_eop.csv")
    f["date"] = pd.to_datetime(f["period_end"])
    if vd is not None:
        f = f[pd.to_datetime(f["publication_date"]) <= vd]
    rows = [{"date": r.date, "tzs_per_usd": r.tzs_per_usd, "source": f"BoT AR {r.file} p.{r.page}"}
            for r in f.itertuples()]
    for d in vdates(root).values():
        if vd is not None and d > vd:
            continue
        v, obs = macro_at(root, d, "macro_tzs_per_usd")
        if math.isfinite(v):
            rows.append({"date": pd.Timestamp(obs) if obs else d, "tzs_per_usd": v,
                         "source": f"inputs workbook macro (BoT indicative rate, as at {d.date()})"})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fx_year_end(fx: pd.DataFrame) -> dict[int, float]:
    """Year -> end-December rate (a rate dated 30 or 31 December)."""
    d = fx[(fx["date"].dt.month == 12) & (fx["date"].dt.day >= 30)]
    return {int(r.date.year): float(r.tzs_per_usd) for r in d.itertuples()}


def annual_returns(root: Path) -> pd.DataFrame:
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    fx = fx_year_end(fx_table(root))
    rows = []
    for b, name in BANKS.items():
        g = w[w["bank_id"] == b].set_index("fiscal_year").sort_index()
        div = g["dps_total"].fillna(g["dps"])
        for y in g.index:
            if y - 1 not in g.index:
                continue
            p0, p1, d = g.loc[y - 1, "share_price_close"], g.loc[y, "share_price_close"], div.get(y - 1)
            if pd.isna(p0) or pd.isna(p1) or pd.isna(d):
                continue
            r = (p1 + d) / p0 - 1
            usd = (1 + r) * fx[y - 1] / fx[y] - 1 if y - 1 in fx and y in fx else np.nan
            rows.append({"bank": name, "period": str(y), "start": f"{y - 1}-12-31", "end": f"{y}-12-31", "years": 1.0,
                         "price_start": p0, "price_end": p1, "dividends": d, "tzs_return": r,
                         "fx_start": fx.get(y - 1), "fx_end": fx.get(y), "usd_return": usd,
                         "fx_change": fx[y] / fx[y - 1] - 1 if y - 1 in fx and y in fx else np.nan,
                         "note": "" if math.isfinite(usd) else "USD not translated: end-December FX not held",
                         "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def span_returns(root: Path, ann: pd.DataFrame) -> pd.DataFrame:
    """Chained TZS returns between held FX end points: the full span and the gap span 2022-2025; plus the
    31 Dec 2025 - 31 Oct 2026 window from the chapter 11 tested call."""
    fx = fx_year_end(fx_table(root))
    rows = []
    for name, g in ann.groupby("bank"):
        g = g.set_index(g["period"].astype(int)).sort_index()
        first = min((y for y in fx if y >= int(g.index.min()) - 1), default=None)
        spans = [(first, int(g.index.max()))] if first is not None else []
        held = sorted(y for y in fx if y < 2025)
        if 2025 in fx and held:
            spans.append((max(held), 2025))
        for a, z in spans:
            yrs = list(range(a + 1, z + 1))
            if not all(y in g.index for y in yrs) or a not in fx or z not in fx:
                continue
            tzs = float(np.prod(1 + g.loc[yrs, "tzs_return"])) - 1
            usd = (1 + tzs) * fx[a] / fx[z] - 1
            n = z - a
            rows.append({"bank": name, "period": f"Dec {a} - Dec {z}", "years": n, "tzs_return": tzs,
                         "usd_return": usd, "tzs_cagr": (1 + tzs) ** (1 / n) - 1, "usd_cagr": (1 + usd) ** (1 / n) - 1,
                         "fx_start": fx[a], "fx_end": fx[z], "fx_cagr": (fx[z] / fx[a]) ** (1 / n) - 1,
                         "evidence_class": "Derived"})
    tc = path("tables", root) / "ch11_tested_call.csv"
    if tc.exists():
        t = pd.read_csv(tc)
        vd = vdates(root)
        f0, _ = macro_at(root, vd["HIST"], "macro_tzs_per_usd")
        f1, _ = macro_at(root, vd["CURR"], "macro_tzs_per_usd")
        for r in t.itertuples():
            n = float(r.months) / 12
            usd = (1 + r.actual_total_return) * f0 / f1 - 1
            rows.append({"bank": r.bank, "period": "31 Dec 2025 - 31 Oct 2026 (tested-call window)", "years": n,
                         "tzs_return": r.actual_total_return, "usd_return": usd,
                         "tzs_cagr": (1 + r.actual_total_return) ** (1 / n) - 1, "usd_cagr": (1 + usd) ** (1 / n) - 1,
                         "fx_start": f0, "fx_end": f1, "fx_cagr": (f1 / f0) ** (1 / n) - 1, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def depreciation(root: Path, val: str) -> pd.DataFrame:
    """Depreciation scenarios at a valuation date, from information available then (Scenario)."""
    vd = vdates(root)[val]
    cfg = _cfg(root)
    fx = fx_table(root, vd)
    ye = fx_year_end(fx)
    end = fx.iloc[-1]
    first = min(ye)
    n = (end["date"] - pd.Timestamp(first, 12, 31)).days / 365.25
    hist = (end["tzs_per_usd"] / ye[first]) ** (1 / n) - 1
    steps = {y: ye[y] / ye[y - 1] - 1 for y in ye if y - 1 in ye}
    worst = max(steps, key=steps.get)
    pi_tz, obs = macro_at(root, vd, "macro_tanzania_inflation_forecast_medium_term")  # IMF forecast (D64)
    tz_src = f"IMF forecast as at {vd.date()}"
    if not math.isfinite(pi_tz):
        pi_tz, obs = macro_at(root, vd, "macro_tanzania_headline_cpi_inflation_latest_y_y")
        tz_src = f"headline CPI y/y, {obs}"
    pi_us = cfg.get("inflation_us")
    us_src = "config override"
    if pi_us is None:
        pi_us, _ = macro_at(root, vd, "macro_us_long_run_inflation_expectation")
        us_src = f"IMF forecast as at {vd.date()}"
    ppp = (1 + pi_tz) / (1 + pi_us) - 1 if math.isfinite(pi_us) and math.isfinite(pi_tz) else np.nan
    rows = [("none", 0.0, "no change in TZS/USD"),
            ("ppp", ppp, f"relative PPP: (1 + Tanzania inflation {pct(pi_tz)}, {tz_src}) / (1 + US inflation "
                         f"{pct(pi_us)}, {us_src}) - 1 (D64)"),
            ("history", hist, f"BoT TZS/USD CAGR Dec {first} ({ye[first]:,.1f}) to {end['date'].date()} "
                              f"({end['tzs_per_usd']:,.1f}), {n:.1f} years"),
            ("stress", steps[worst], f"worst held calendar year: {worst} ({ye[worst - 1]:,.1f} -> {ye[worst]:,.1f})")]
    last = max(ye)
    if end["date"] > pd.Timestamp(last, 12, 31) and end["date"].year == last + 1:  # a part year is held (CURR)
        months = (end["date"] - pd.Timestamp(last, 12, 31)).days / 365.25 * 12
        ytd = end["tzs_per_usd"] / ye[last] - 1
        rows += [("realised_ytd", ytd, f"realised move Dec {last} ({ye[last]:,.1f}) to {end['date'].date()} "
                                       f"({end['tzs_per_usd']:,.1f}), {months:.0f} months, as the next 12 months "
                                       "(sensitivity, prompt 17 B8)"),
                 ("realised_ytd_annualised", (1 + ytd) ** (12 / months) - 1,
                  f"the same move annualised over {months:.0f} months (sensitivity)")]
    return pd.DataFrame([{"valuation": val, "scenario": s, "depreciation": d, "basis": b,
                          "base": s == cfg["base_depreciation"], "evidence_class": "Scenario"} for s, d, b in rows])


# --- liquidity, cost, capacity ----------------------------------------------------------------------------------------
def fee_per_side(q: float, fees: dict) -> float:
    """DSE fee schedule on one trade of q TZS: tiered brokerage x (1 + VAT) + market fees, capped (fraction of q)."""
    if q <= 0:
        return float("nan")
    if fees.get("mode", "schedule") == "cap":  # owner, issue 32: the DSE FAQ cap per side, conservative
        return float(fees["max_total_per_side"])
    brok, lo = 0.0, 0.0
    for hi, rate in fees["brokerage_tiers"]:
        top = q if hi is None else min(q, hi)
        if top > lo:
            brok += (top - lo) * rate
        if hi is None or q <= hi:
            break
        lo = hi
    return min(brok * (1 + fees["brokerage_vat"]) / q + fees["market_fees"], fees["max_total_per_side"])


def window(root: Path, vd: pd.Timestamp) -> pd.DataFrame:
    """Daily data up to the valuation date only (quality gate: capacity uses no later data)."""
    m = pd.read_csv(path("processed", root) / "market_daily.csv", dtype={"bank_id": str}, low_memory=False)
    m["date"] = pd.to_datetime(m["date"])
    m = m[(m["date"] <= vd) & m["bank_id"].isin(BANKS)].sort_values(["bank_id", "date"]).copy()
    mult = _cfg(root)["block_multiple"]
    out = []
    for _, g in m.groupby("bank_id"):
        g = g.copy()
        med = g.loc[g["traded_day"] == 1, "turnover"].median()
        flag = g["data_flag"].astype(str).str.contains("block", case=False)
        g["block"] = flag | (g["turnover"] > mult * med)
        g["block_reason"] = np.where(flag, "data flag (negotiated / pre-arranged block)",
                                     np.where(g["block"], f"turnover > {mult}x window median", ""))
        tr = g[g["traded_day"] == 1]
        g["ret"] = tr["adj_close"].pct_change()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def liquidity(root: Path, val: str) -> pd.DataFrame:
    vd = vdates(root)[val]
    w = window(root, vd)
    rows = []
    for b, g in w.groupby("bank_id"):
        nb = g[~g["block"]]
        ok = nb[(nb["traded_day"] == 1) & (nb["turnover"] > 0)].dropna(subset=["ret"])
        rows.append({"valuation": val, "bank": BANKS[b], "window_start": g["date"].min().date(),
                     "window_end": g["date"].max().date(), "exchange_days": len(g),
                     "zero_trade_days": int((g["traded_day"] == 0).sum()),
                     "zero_trade_share": float((g["traded_day"] == 0).mean()),
                     "block_days": int(g["block"].sum()), "block_turnover_tzs": float(g.loc[g["block"], "turnover"].sum()),
                     "block_share_of_turnover": float(g.loc[g["block"], "turnover"].sum() / g["turnover"].sum()),
                     "adv_tzs_ex_blocks": float(nb["turnover"].mean()), "adv_tzs_incl_blocks": float(g["turnover"].mean()),
                     "median_daily_tzs": float(g["turnover"].median()),
                     "amihud_illiq_per_tzs": float((ok["ret"].abs() / ok["turnover"]).mean()),
                     "amihud_illiq_per_tzs_bn": 1e9 * float((ok["ret"].abs() / ok["turnover"]).mean()),
                     "amihud_days": len(ok), "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def capacity(root: Path, liq: pd.DataFrame) -> pd.DataFrame:
    cfg = _cfg(root)
    rows = []
    for r in liq.itertuples():
        fx, _ = macro_at(root, vdates(root)[r.valuation], "macro_tzs_per_usd")
        for usd in cfg["positions_usd"]:
            for p in cfg["participation"]:
                pos = usd * fx
                q = p * r.adv_tzs_ex_blocks
                fee = fee_per_side(q, cfg["fees"])
                imp = r.amihud_illiq_per_tzs * q
                rows.append({"valuation": r.valuation, "bank": r.bank, "position_usd": usd, "position_tzs": pos,
                             "participation": p, "daily_trade_tzs": q,
                             "days_to_build_or_exit": math.ceil(pos / q),
                             "days_incl_blocks": math.ceil(pos / (p * r.adv_tzs_incl_blocks)),
                             "fee_per_side": fee, "impact_per_side": imp, "cost_per_side": fee + imp,
                             "round_trip_cost": 2 * (fee + imp), "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- expected USD returns ---------------------------------------------------------------------------------------------
def _register(root: Path, val: str, bank: str, symbol: str) -> float:
    r = pd.read_csv(root / "config" / "assumption_register.csv")
    x = r[(r["valuation"] == val) & (r["bank"] == bank) & (r["symbol"] == symbol)]
    return float(x["base"].iloc[0])


def hurdle(root: Path, val: str, bank: str) -> tuple[float, str]:
    """USD hurdle = USD CAPM with the TZS COE's beta, ERP, CRP and LP (VALUATION_SPEC §5 approach B, USD leg)."""
    from tzbank.valuation.coe import usd_capm

    vd = vdates(root)[val]
    rf, _ = macro_at(root, vd, "macro_us_10_year_treasury_yield")
    erp, _ = macro_at(root, vd, "macro_mature_market_implied_erp")
    crp, _ = macro_at(root, vd, "macro_tanzania_country_risk_premium")
    beta, lp = _register(root, val, bank, "beta"), _register(root, val, bank, "LP")
    return usd_capm(rf, beta, erp, crp, lp), (f"US 10-year {pct(rf, 2)} + beta {beta:.3f} x (ERP {pct(erp, 2)} + "
                                              f"CRP {pct(crp, 2)}) + LP {pct(lp, 2)}")


def usd_total_return(price_return: float, divs: list[tuple[float, float]], wht: float, d: float) -> float:
    """12-month USD total return of a TZS share (prompt 24, owner's method; Blueprint ch. 12):
    [P1 / FX(1) + sum_i D_i (1 - wht) / FX(t_i)] / (P0 / FX0) - 1 with FX(t) = FX0 (1 + d)^t, i.e.
    (1 + price return) / (1 + d) + sum_i (D_i / P0)(1 - wht) / (1 + d)^t_i - 1. ``divs`` = [(D_i / P0, t_i in
    years)]. Withholding applies to dividends only; no trading cost (shown as a separate line)."""
    return (1 + price_return) / (1 + d) + sum(y * (1 - wht) / (1 + d) ** t for y, t in divs) - 1


def _breakeven(price_return: float, divs: list[tuple[float, float]], wht: float, h: float) -> float:
    """Depreciation d* at which the USD total return equals the hurdle h (negative: appreciation needed)."""
    from scipy.optimize import brentq

    f = lambda d: usd_total_return(price_return, divs, wht, d) - h  # noqa: E731
    return float(brentq(f, -0.95, 3.0))


def expected_usd(root: Path, cap: pd.DataFrame) -> pd.DataFrame:
    """Expected 12-month USD return per date, bank, depreciation scenario and position (prompt 24 rebuild): the TZS
    price return to the 12-month target and each dividend in the window on its payment date are translated at the rate
    implied by the scenario's depreciation on that date; 5% withholding on dividends only; the round-trip trading cost
    of building and selling the position is a separate line (``expected_usd_return_after_cost``)."""
    cfg = _cfg(root)
    v = pd.read_csv(path("tables", root) / "ch09_valuations.csv")
    dv = pd.read_csv(path("tables", root) / "ch09_dividends_12m.csv")
    w = float(cfg["div_wht_foreign"])
    rows = []
    for val in DATES:
        dep = depreciation(root, val)
        for bank in BANKS.values():
            x = v[(v["valuation"] == val) & (v["bank"] == bank)]
            if x.empty:
                continue
            h, hb = hurdle(root, val, bank)
            p0, p1 = float(x["price"].iloc[0]), float(x["target_price_12m"].iloc[0])
            gross = float(x["expected_total_return"].iloc[0])
            dd = dv[(dv["valuation"] == val) & (dv["bank"] == bank)]
            divs = [(float(r.dps) / p0, float(r.years_from_valuation)) for r in dd.itertuples()]
            pr = p1 / p0 - 1
            dy = sum(y for y, _ in divs)
            be = _breakeven(pr, divs, w, h)
            for usd in cfg["positions_usd"]:
                c = cap[(cap["valuation"] == val) & (cap["bank"] == bank) & (cap["position_usd"] == usd)
                        & np.isclose(cap["participation"], cfg["base_participation"])]
                rt = float(c["round_trip_cost"].iloc[0])
                for d in dep.itertuples():
                    e = usd_total_return(pr, divs, w, d.depreciation)
                    tzs_net = pr + dy * (1 - w)
                    rows.append({"valuation": val, "bank": bank, "position_usd": usd,
                                 "participation": cfg["base_participation"], "scenario": d.scenario,
                                 "base_scenario": bool(d.base), "price_tzs": p0, "target_price_12m": p1,
                                 "tzs_price_return": pr, "expected_dividend_yield": dy, "dividend_wht": w * dy,
                                 "dividend_pay_dates": "; ".join(str(r.pay_date) for r in dd.itertuples()),
                                 "expected_tzs_return_gross": gross, "expected_tzs_return_12m": tzs_net,
                                 "depreciation": d.depreciation, "translation_effect": e - tzs_net,
                                 "expected_usd_return": e, "usd_hurdle": h, "hurdle_basis": hb,
                                 "excess_over_hurdle": e - h, "breakeven_depreciation": be,
                                 "round_trip_cost": rt, "expected_usd_return_after_cost": e - rt / (1 + d.depreciation),
                                 "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def usd_steps(exp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Step-by-step table for the report (prompt 24): base position and participation, the history (base) and
    realised-pace depreciation cases, at the current date."""
    e = exp[(exp["valuation"] == "CURR") & (exp["position_usd"] == cfg["base_position_usd"])
            & exp["scenario"].isin(["history", "realised_ytd_annualised"])]
    steps = [("TZS price return to the 12-month target", "tzs_price_return"),
             ("TZS dividends received (yield on price)", "expected_dividend_yield"),
             ("Withholding tax on dividends (5%)", "dividend_wht"),
             ("TZS total return after withholding", "expected_tzs_return_12m"),
             ("Assumed depreciation of the TZS, a year", "depreciation"),
             ("Translation effect", "translation_effect"),
             ("USD total return", "expected_usd_return"),
             ("USD hurdle", "usd_hurdle"),
             ("Gap to the hurdle", "excess_over_hurdle"),
             ("Round-trip trading cost (USD 20m base position)", "round_trip_cost"),
             ("USD return after trading cost", "expected_usd_return_after_cost")]
    rows = []
    for r in e.itertuples():
        case = "history (3.8% a year)" if r.scenario == "history" else "2026 pace (annualised)"
        for i, (lab, col) in enumerate(steps):
            rows.append({"bank": r.bank, "case": case, "scenario": r.scenario, "step_no": i + 1, "step": lab,
                         "value": float(getattr(r, col)), "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def hypotheses(root: Path, exp: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    cfg = _cfg(root)
    base = exp[exp["base_scenario"] & (exp["position_usd"] == cfg["base_position_usd"])]
    per1 = {(r.valuation, r.bank): r.expected_usd_return > r.usd_hurdle for r in base.itertuples()}
    v1 = "supported" if all(per1.values()) else "rejected" if not any(per1.values()) else "inconclusive"
    ev1 = "; ".join(f"{r.bank} {r.valuation}: E[r_USD] {pct(r.expected_usd_return)} vs hurdle {pct(r.usd_hurdle)} "
                    f"(TZS {pct(r.expected_tzs_return_12m)}, cost {pct(r.round_trip_cost, 2)}, depreciation "
                    f"{pct(r.depreciation)}; break-even depreciation {pct(r.breakeven_depreciation)})"
                    for r in base.itertuples())
    lg = exp[exp["base_scenario"] & (exp["position_usd"] == cfg["large_fund_position_usd"])]
    cp = cap[(cap["position_usd"] == cfg["large_fund_position_usd"])
             & np.isclose(cap["participation"], cfg["base_participation"])].set_index(["valuation", "bank"])
    per2, ev2 = {}, []
    for r in lg.itertuples():
        days = int(cp.loc[(r.valuation, r.bank), "days_to_build_or_exit"])
        capb = days > cfg["capacity_days_binding"]
        valb = r.expected_usd_return <= r.usd_hurdle
        per2[(r.valuation, r.bank)] = ("supported" if capb and not valb else "rejected" if not capb
                                       else "inconclusive (both bind)")
        ev2.append(f"{r.bank} {r.valuation}: {days} days to build USD {cfg['large_fund_position_usd'] / 1e6:.0f}m at "
                   f"{cfg['base_participation']:.0%} of ADV ({'binds' if capb else 'does not bind'}); E[r_USD] "
                   f"{pct(r.expected_usd_return)} vs hurdle {pct(r.usd_hurdle)} (valuation "
                   f"{'binds' if valb else 'does not bind'})")
    vals = set(per2.values())
    v2 = "supported" if vals == {"supported"} else "rejected" if vals == {"rejected"} else "inconclusive"
    return pd.DataFrame([
        {"hypothesis": "H12.1", "verdict": v1,
         "verdict_by_bank_date": ", ".join(f"{b} {v} {'yes' if k else 'no'}" for (v, b), k in per1.items()),
         "evidence": ev1,
         "rule": f"base case (USD {cfg['base_position_usd'] / 1e6:.0f}m at {cfg['base_participation']:.0%} of ADV, "
                 f"'{cfg['base_depreciation']}' depreciation): supported when E[r_USD] after round-trip cost exceeds "
                 "the USD hurdle for both banks at both dates; rejected when it falls short in every case; "
                 "inconclusive otherwise (D60 proposed)"},
        {"hypothesis": "H12.2", "verdict": v2,
         "verdict_by_bank_date": ", ".join(f"{b} {v} {k}" for (v, b), k in per2.items()), "evidence": "; ".join(ev2),
         "rule": f"per bank and date, capacity binds when building USD {cfg['large_fund_position_usd'] / 1e6:.0f}m at "
                 f"{cfg['base_participation']:.0%} of ADV takes more than {cfg['capacity_days_binding']} trading days; "
                 "valuation binds when the base E[r_USD] does not exceed the hurdle. Supported when capacity binds and "
                 "valuation does not everywhere; rejected when capacity does not bind everywhere; inconclusive "
                 "otherwise (D60 proposed)"}]).assign(evidence_class="Interpretation")


NOTE = """# Chapter 12: USD returns and implementation

Generated by `tzbank.chapters.ch12_usd_implementation` (stage `usd`). Blueprint §4 chapter 12; H12.1, H12.2; D60
(proposed). Expected TZS returns are chapter 9's (register {version}); repository private; publication blocked by open
issue 19 (CMSA licensing). **No investment advice; the market-access table is desk research for counsel review and
draws no legal conclusions.**

## Research question

What did NMB and CRDB return in USD, what would a USD investor expect after depreciation and trading costs against a
USD hurdle, and can a large fund build and exit a position on the DSE?

## Variable definitions

TZS total return = (P_t + D_t) / P_t-1 - 1 (year-end prices; D_t = dividend paid in year t); USD return = (1 + TZS
return) x FX_start / FX_end - 1 (TZS per USD, end-December); E[r_USD] = (1 + E[r_TZS] - round-trip cost) / (1 + d) - 1;
break-even depreciation d* = (1 + E[r_TZS] - cost) / (1 + h) - 1; hurdle h = US 10-year + EM premium; days to build =
position / (participation x ADV); cost per side = DSE fees (tiered brokerage + VAT + market fees) + Amihud impact
(ILLIQ x daily trade value). Definitions in `docs/VARIABLE_DICTIONARY.md`.

## Sources and transformation trail

`data/interim/bot_fx_eop.csv` (BoT Annual Reports, `tzbank.extract.bot_fx`; check `reports/validation/bot_fx_check.csv`),
`data/processed/macro.csv` (FX, US 10-year, ERP, CRP, CPI at the dates), `data/processed/bank_year_wide.csv` (prices,
dividends), `data/processed/market_daily.csv` (DSE daily, {d0} to {d1}), `outputs/tables/ch09_valuations.csv`,
`outputs/tables/ch11_tested_call.csv`, `config/market_access.csv`; config `ch12`. Functions: `annual_returns`,
`span_returns`, `depreciation`, `liquidity`, `capacity`, `fee_per_side`, `expected_usd`, `hypotheses`.

## Validation test

`tests/test_ch12.py`: FX alignment against known rates; USD translation and break-even identities by hand; fee tiers by
hand; capacity uses only data up to each valuation date; every market-access row has a source and access date.

## Results

### USD returns, history (Derived)

{hist}

### Depreciation scenarios and the USD hurdle (Scenario)

{dep}

### Expected USD returns, base size (Scenario)

{exp}

### Capacity, liquidity and trading cost (Derived / Estimated)

{liq}

{cap}

### Market access (desk research, for counsel review)

{access}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

Historical returns and FX: **Derived** from Observed data; costs and impact: **Estimated**; depreciation and expected
returns: **Scenario**; market access: **Observed** statements of the cited sources (not verified); verdicts:
**Interpretation**.

## Limitations

- End-December TZS/USD for 2023 and 2024 is not held (the 2024/25 BoT report prints fiscal-year columns), so 2023-24 are
  translated only as part of the Dec 2022 - Dec 2025 span; collection task (open issue 34).
- Forward US inflation is TO COLLECT; the PPP scenario uses the FOMC 2% goal as a stated assumption (open issue 31).
- The daily data start on {d0}: the 31 Dec 2025 capacity uses about four months of trading; Amihud impact is a linear
  average and understates the cost of trades much larger than the typical day.
- The fee schedule is desk research pending the owner's copy (open issue 32). The USD hurdle uses the TZS COE's beta,
  ERP, CRP and LP (issue 33 resolved); the US 10-year replaces the TZS long-bond yield net of the default spread.
- Dividend withholding tax for a foreign holder (`ch12.div_wht_foreign`, 5% for DSE-listed companies per the desk
  research) is deducted from the expected dividend yield; treaty rates and the tax position of the holder are not
  modelled (counsel review).

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

Chapter 12 does not test the rent explanations; it asks whether a USD investor can capture the value chapters 9-11
estimate. Thin trading and high transaction costs are themselves a barrier that limits arbitrage of any mispricing.
"""


def cross_rates(root: Path) -> pd.DataFrame:
    """Shilling against regional currencies (prompt 24b A7; blueprint ch. 12, currency translation): the change in the
    TZS value of one shilling in each currency, from the BoT TZS/USD (``ch12_fx``) and the peer central banks' USD rates
    (``peer_macro_collected.csv``, Observed). Positive = the shilling strengthened against that currency. Windows:
    end-2020 (the first held year of the peer rates) and end-2025, both to 30 Oct 2026."""
    fx = fx_table(root)
    fx["date"] = pd.to_datetime(fx["date"])
    m = pd.read_csv(root / "data" / "raw" / "manual_entries" / "peers" / "peer_macro_collected.csv")
    m = m[m["field"] == "usd_fx"].assign(date=lambda d: pd.to_datetime(d["date"]))
    m["currency"] = m["unit"].str.split().str[0]

    def at(d: pd.DataFrame, col: str, when: str) -> float:
        x = d[(d["date"] - pd.Timestamp(when)).abs() <= pd.Timedelta(days=3)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    rows = []
    for cur, g in m.groupby("currency"):
        for start in ("2020-12-31", "2025-12-31"):
            end = "2026-10-31"
            t0, t1 = at(fx, "tzs_per_usd", start), at(fx, "tzs_per_usd", end)
            x0, x1 = at(g, "value", start), at(g, "value", end)
            rows.append({"currency": cur, "country": g["country"].iloc[0], "start": start, "end": "2026-10-30",
                         "tzs_per_usd_start": t0, "tzs_per_usd_end": t1, "fx_per_usd_start": x0,
                         "fx_per_usd_end": x1, "tzs_change_vs_currency": (x1 / t1) / (x0 / t0) - 1,
                         "evidence_class": "Derived"})
    out = pd.DataFrame(rows)
    usd = [{"currency": "USD", "country": "United States", "start": s0, "end": "2026-10-30",
            "tzs_per_usd_start": at(fx, "tzs_per_usd", s0), "tzs_per_usd_end": at(fx, "tzs_per_usd", "2026-10-31"),
            "fx_per_usd_start": 1.0, "fx_per_usd_end": 1.0,
            "tzs_change_vs_currency": at(fx, "tzs_per_usd", s0) / at(fx, "tzs_per_usd", "2026-10-31") - 1,
            "evidence_class": "Derived"} for s0 in ("2020-12-31", "2025-12-31")]
    return pd.concat([out, pd.DataFrame(usd)], ignore_index=True)


def write_ch12(root: Path) -> list[Path]:
    ann = annual_returns(root)
    spans = span_returns(root, ann)
    dep = pd.concat([depreciation(root, v) for v in DATES], ignore_index=True)
    liq = pd.concat([liquidity(root, v) for v in DATES], ignore_index=True)
    cap = capacity(root, liq)
    exp = expected_usd(root, cap)
    hyp = hypotheses(root, exp, cap)
    acc = pd.read_csv(root / "config" / "market_access.csv").assign(evidence_class="Observed")
    fx = fx_table(root).assign(evidence_class="Observed")
    tab, figd = path("tables", root), path("figures", root)
    tables = [(fx, "ch12_fx.csv", "TZS per USD used (end-December BoT; workbook at the dates)", "Observed"),
              (ann, "ch12_returns_annual.csv", "Annual TZS and USD total returns", "Derived"),
              (spans, "ch12_returns_spans.csv", "TZS and USD returns over held FX spans", "Derived"),
              (dep, "ch12_depreciation.csv", "Depreciation scenarios at each date", "Scenario"),
              (liq, "ch12_liquidity.csv", "Liquidity, zero-trade days, blocks, Amihud (data to each date)", "Derived"),
              (cap, "ch12_capacity.csv", "Days to build or exit and trading cost by size and participation",
               "Estimated"),
              (exp, "ch12_expected_usd.csv", "Expected USD returns, hurdle and break-even depreciation", "Scenario"),
              (usd_steps(exp, _cfg(root)), "ch12_usd_steps.csv", "USD return step by step, 31 Oct 2026 (prompt 24)",
               "Scenario"),
              (cross_rates(root), "ch12_cross_rates.csv",
               "TZS against regional currencies, end-2020 and end-2025 to 30 Oct 2026 (prompt 24b)", "Derived"),
              (acc, "ch12_market_access.csv", "Market access desk research (for counsel review)", "Observed"),
              (hyp, "ch12_hypotheses.csv", "H12.1-H12.2", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    cfg = _cfg(root)
    c = cap[(cap["valuation"] == "CURR") & np.isclose(cap["participation"], cfg["base_participation"])]
    fig, ax = plt_subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(cfg["positions_usd"]))
    for k, (b, name) in enumerate(BANKS.items()):
        g = c[c["bank"] == name].sort_values("position_usd")
        ax.bar(xs + (k - 0.5) * 0.38, g["days_to_build_or_exit"], width=0.38, color=COLORS[b], label=name)
    ax.axhline(cfg["capacity_days_binding"], color="#1f1f1d", lw=0.8, ls="--")
    ax.set_xticks(xs, [f"USD {u / 1e6:.0f}m" for u in cfg["positions_usd"]])
    style(ax, f"Trading days to build or exit at {cfg['base_participation']:.0%} of daily traded value (31 Oct 2026)",
          "Trading days")
    ax.legend(fontsize=7, frameon=False)
    png = save(fig, figd / "ch12_capacity.png", "Evidence class: Estimated. ADV over the daily data to 31 Oct 2026, "
               "block days excluded; dashed line: capacity threshold (D60).")
    fd = write_csv(c[["bank", "position_usd", "participation", "days_to_build_or_exit", "evidence_class"]],
                   figd / "ch12_capacity_data.csv", require_evidence=True)
    files += [png, fd]

    hist = []
    for r in spans.itertuples():
        hist.append(f"- {r.bank}, {r.period}: TZS {pct(r.tzs_return)} ({pct(r.tzs_cagr)} a year), USD {pct(r.usd_return)} "
                    f"({pct(r.usd_cagr)} a year); TZS/USD {r.fx_start:,.1f} -> {r.fx_end:,.1f} ({pct(r.fx_cagr)} a year).")
    worst = ann.dropna(subset=["usd_return"]).sort_values("usd_return").groupby("bank").head(1)
    hist += [f"- Worst translated year, {r.bank}: {r.period} USD {pct(r.usd_return)} (TZS {pct(r.tzs_return)}, FX "
             f"{pct(r.fx_change, sign=True)})." for r in worst.itertuples()]
    hist.append(f"- Years not translated (no end-December FX): {', '.join(sorted(set(ann.loc[ann['usd_return'].isna(), 'period'])))}.")
    dep_txt = []
    for val in DATES:
        d = dep[dep["valuation"] == val]
        hs = [(bank, *hurdle(root, val, bank)) for bank in BANKS.values()]
        dep_txt.append(f"- {val}: " + "; ".join(f"{r.scenario}{' (base)' if r.base else ''} {pct(r.depreciation)}"
                                                 for r in d.itertuples())
                       + "; USD hurdle " + "; ".join(f"{b} {pct(h)} ({hb})" for b, h, hb in hs) + ".")
    dep_txt += [f"- {r.valuation} {r.scenario}: {r.basis}." for r in dep.itertuples() if r.scenario != "none"]
    b = exp[exp["position_usd"] == cfg["base_position_usd"]]
    exp_txt = []
    for (val, bank), g in b.groupby(["valuation", "bank"], sort=False):
        g = g.set_index("scenario")
        exp_txt.append(f"- {bank} {val}: E[r_TZS] {pct(g['expected_tzs_return_gross'].iloc[0])} gross, "
                       f"{pct(g['expected_tzs_return_12m'].iloc[0])} after dividend WHT "
                       f"({pct(g['dividend_wht'].iloc[0], 2)} on a {pct(g['expected_dividend_yield'].iloc[0])} yield), round trip "
                       f"{pct(g['round_trip_cost'].iloc[0], 2)}; E[r_USD] " + ", ".join(
                           f"{s} {pct(g.loc[s, 'expected_usd_return'])}" for s in g.index)
                       + f"; hurdle {pct(g['usd_hurdle'].iloc[0])}; break-even depreciation "
                         f"{pct(g['breakeven_depreciation'].iloc[0])}.")
    liq_txt = "\n".join(
        f"- {r.bank} {r.valuation} ({r.window_start} to {r.window_end}, {r.exchange_days} days): ADV ex-blocks TZS "
        f"{r.adv_tzs_ex_blocks / 1e9:,.2f}bn (incl. blocks {r.adv_tzs_incl_blocks / 1e9:,.2f}bn; median day "
        f"{r.median_daily_tzs / 1e9:,.2f}bn); zero-trade days {r.zero_trade_days} ({pct(r.zero_trade_share)}); block days "
        f"{r.block_days} ({pct(r.block_share_of_turnover)} of turnover); Amihud ILLIQ {r.amihud_illiq_per_tzs * 1e9:.4f} "
        f"per TZS bn ({r.amihud_days} days)." for r in liq.itertuples())
    cap_txt = "\n".join(
        f"- {r.bank} {r.valuation}, USD {r.position_usd / 1e6:.0f}m at {r.participation:.0%}: {r.days_to_build_or_exit} "
        f"days ({r.days_incl_blocks} incl. blocks); cost per side {pct(r.cost_per_side, 2)} (fees {pct(r.fee_per_side, 2)}, "
        f"impact {pct(r.impact_per_side, 2)})." for r in cap[np.isclose(cap["participation"], 0.20)].itertuples())
    acc_txt = ("| Topic | Item | Source (date) | Accessed | Follow-up |\n| --- | --- | --- | --- | --- |\n" + "\n".join(
        f"| {r.topic} | {r.item} | [{r.source_title}]({r.source_url}) ({r.source_date}) | {r.access_date} | {r.follow_up} |"
        for r in acc.itertuples())
        + "\n\nFull statements in `outputs/tables/ch12_market_access.csv`; every row is marked **for counsel review**.")
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Rule: {r.rule}." for r in hyp.itertuples())
    interp = ("**Interpretation.** Read with chapters 9 and 11: the USD case rests on the TZS "
              "expected return, which the base-centred Monte Carlo puts below the price for both banks at 31 Oct 2026; "
              "depreciation and a round trip of several percent widen the gap. Capacity is a separate constraint: a "
              "large position takes weeks to months to build at a fifth of daily traded value.")
    m = pd.read_csv(path("processed", root) / "market_daily.csv", usecols=["date"])
    reg = pd.read_csv(root / "config" / "assumption_register.csv", usecols=["version"])
    nf = write_text(NOTE.format(version=str(reg["version"].iloc[0]), d0=m["date"].min(), d1=m["date"].max(),
                                hist="\n".join(hist), dep="\n".join(dep_txt), exp="\n".join(exp_txt), liq=liq_txt,
                                cap=cap_txt, access=acc_txt, hyp=hyp_txt, interp=interp), path("notes", root) / "ch12.md")
    files.append(nf)
    man = {"chapter": "12", "title": "USD returns and implementation", "hypotheses": ["H12.1", "H12.2"],
           "status": "built",
           "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                      for _, n, d, c in tables],
           "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "Days to build or exit",
                        "evidence_class": "Estimated", "note": rel(nf, root)}],
           "notes": [rel(nf, root)], "module": "tzbank.chapters.ch12_usd_implementation", "run_log_stage": "usd"}
    files.append(write_json(man, path("manifests", root) / "ch12.json"))
    return files
