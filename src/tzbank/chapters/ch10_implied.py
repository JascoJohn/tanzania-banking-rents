"""Chapter 10: market-implied expectations (blueprint §0 headline metric, §4 chapter 10; H10.1; D12 implied persistence).

For each bank and date: the full iso-value surface over (lambda, ER*) with the forecast and COE fixed; the market-
implied ER* at the chapter 5 lambda and the implied lambda at the chapter 5 ER*, each with the extra assumption stated;
implied COE and implied ROE*; the rerating decomposition 31 Dec 2025 -> 31 Oct 2026 (earnings and time, discount rate,
residual) on a total-return basis; H10.1. Chapter 5 estimates: 31 Dec 2025 = re-estimated on ER through FY2024
(eligible then); 31 Oct 2026 = `ch05_calibration_candidates.csv`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import models as vm
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.context import BANKS, Ctx, pct

T = rg.TICK
LAMS = np.round(np.arange(0.0, 0.951, 0.05), 3)
ERS = np.round(np.arange(-0.05, 0.1501, 0.005), 4)


def _value(c: un.Case, lam: float, er: float) -> float:
    roe = c.r + er
    d = replace(c.dr, lam=lam, roe_star=roe, payout_star=1 - c.dr.g / roe, fade_shape="lambda")
    return un.value(c, d)


def implied_er(c: un.Case, lam: float) -> float:
    pe = c.price * c.shares
    lo, hi = c.dr.g - c.r + 1e-3, 0.40
    f = lambda er: _value(c, lam, er) - pe  # noqa: E731
    try:
        return float(brentq(f, lo, hi)) if f(lo) * f(hi) < 0 else float("nan")
    except ValueError:
        return float("nan")


def implied_lambda(c: un.Case, er: float) -> float:
    pe = c.price * c.shares
    f = lambda lam: _value(c, lam, er) - pe  # noqa: E731
    try:
        return float(brentq(f, 0.0, 0.999)) if f(0.0) * f(0.999) < 0 else float("nan")
    except ValueError:
        return float("nan")


def estimates(ctx: Ctx, val: str, b: str) -> dict:
    ps = ctx.inputs[(val, b, "persistence")]
    out = {"lam": float(ps["lam"]), "lam_lo": float(ps["lam_lo"]), "lam_hi": float(ps["lam_hi"]),
           "er_brk": float(ps["er_brk"]), "er_ttc": float(ps["er_ttc"]), "years": ps["years"]}
    if val == "HIST":
        out["er_brk_lo"], out["er_brk_hi"] = float(ps["er_brk_lo"]), float(ps["er_brk_hi"])
    else:
        c5 = pd.read_csv(path("tables", ctx.root) / "ch05_calibration_candidates.csv")
        x = c5[(c5["bank_id"] == b) & (c5["specification"] == "ar1_break_episode")].iloc[0]
        out["er_brk_lo"], out["er_brk_hi"] = float(x["er_star_lo"]), float(x["er_star_hi"])
    return out


def surfaces(ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    surf, imp = [], []
    for val in rg.DATES:
        for b in BANKS:
            c = ctx.cases[(val, b, "base")]
            iso = vm.iso_surface(c.op, c.dr, c.r, c.price * c.shares, c.vd, LAMS, ERS,
                                  lag_months=c.lag_months)
            surf.append(iso.assign(valuation=val, bank=T[b]))
            e = estimates(ctx, val, b)
            er_at = implied_er(c, e["lam"])
            lam_at = implied_lambda(c, e["er_brk"])
            lam_note = "" if math.isfinite(lam_at) else (
                "value exceeds the price at every lambda in [0, 1) at the chapter 5 ER*"
                if _value(c, 0.0, e["er_brk"]) > c.price * c.shares else
                "value is below the price at every lambda in [0, 1) at the chapter 5 ER*")
            ic = vm.implied_coe(c.op, c.dr, c.price * c.shares, c.vd, lag_months=c.lag_months)
            ir = vm.implied_roe_star(c.op, c.dr, c.r, c.price * c.shares, c.vd,
                                     lag_months=c.lag_months)
            if not math.isfinite(er_at):
                concl = "no ER* in (g - r, 40%) matches the price at the chapter 5 lambda"
            elif er_at > e["er_brk_hi"]:
                concl = "the price implies MORE long-run rent than the chapter 5 interval supports"
            elif er_at < e["er_brk_lo"]:
                concl = "the price implies LESS long-run rent than the chapter 5 interval supports"
            else:
                concl = "the price-implied ER* lies inside the chapter 5 interval"
            imp.append({"valuation": val, "bank": T[b], "coe": c.r, "lambda_ch5": e["lam"], "lambda_ch5_lo": e["lam_lo"],
                        "lambda_ch5_hi": e["lam_hi"], "er_star_ch5": e["er_brk"], "er_star_ch5_lo": e["er_brk_lo"],
                        "er_star_ch5_hi": e["er_brk_hi"], "er_star_ttc_ch5": e["er_ttc"], "ch5_years": e["years"],
                        "implied_er_star_at_ch5_lambda": er_at, "implied_lambda_at_ch5_er_star": lam_at,
                        "implied_lambda_note": lam_note,
                        "implied_coe": ic, "implied_roe_star": ir, "conclusion": concl,
                        "additional_assumption": "the register's explicit forecast (FY+1..FY+5) and the COE are right, "
                                                 "and lambda equals the chapter 5 break-adjusted estimate (for the "
                                                 "implied ER*) or ER* equals it (for the implied lambda); the fade is "
                                                 "geometric (D56)", "evidence_class": "Estimated"})
    s = pd.concat(surf, ignore_index=True)
    s["evidence_class"] = "Estimated"
    return s, pd.DataFrame(imp)


def hist_drivers_on_curr(h: un.Case, cu: un.Case):
    n = cu.dr.T

    def ext(xs: list[float]) -> list[float]:
        return (list(xs[1:]) + [xs[-1]] * n)[:n]
    return replace(cu.dr, **{k: ext(getattr(h.dr, k)) for k in ("g_L", "g_D", "NIM", "phi", "CIR", "CoR", "rho",
                                                                "rate")}, roe_star=h.dr.roe_star, g=h.dr.g,
                   payout_star=h.dr.payout_star, lam=h.dr.lam)


def rerating(ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total-return decomposition of the market move, 31 Dec 2025 -> 31 Oct 2026 (log points)."""
    rows, br_all = [], []
    for b in BANKS:
        h, cu = ctx.cases[("HIST", b, "base")], ctx.cases[("CURR", b, "base")]
        bc = ctx.inputs[("CURR", b)]
        div = bc.final_dps_anchor * bc.fy["shares_outstanding"]
        pay = vm.pay_date(pd.Timestamp(int(cu.op.year), 12, 31), cu.lag_months) if cu.lag_months else None
        br = un.bridge(h, cu, hist_drivers_on_curr(h, cu), div, pay)
        br_all.append(br.assign(bank=T[b]))
        v = dict(zip(br["step"], br["tzs"], strict=True))
        v0 = v["historical value"]
        v4 = v0 + v["1 passage of time"] + v["2 dividends paid"] + v["3 new information"] + v["4 forecast revisions"]
        v5 = v["current value"]
        mv0, mv1 = h.price * h.shares, cu.price * cu.shares
        market = math.log((mv1 + div) / mv0)
        earn = math.log((v4 + div) / v0)
        disc = math.log(v5 / v4) if v4 > 0 and v5 > 0 else float("nan")
        rows.append({"bank": T[b], "market_cap_hist": mv0, "market_cap_curr": mv1, "dividends_paid": div,
                     "market_total_return_log": market, "earnings_and_time_log": earn, "discount_rate_log": disc,
                     "residual_log": market - earn - disc, "coe_hist": h.r, "coe_curr": cu.r,
                     "value_to_price_hist": v0 / mv0, "value_to_price_curr": v5 / mv1,
                     "definition": "market = ln((MV1 + D) / MV0); earnings and time = ln((V4 + D) / V0) with V4 the "
                                   "current forecast at the historical COE (bridge steps 1-4); discount rate = ln(V5 / "
                                   "V4) (bridge step 5); residual = market - earnings - discount rate (change in "
                                   "the price-to-value gap)", "evidence_class": "Estimated"})
    b_ = pd.concat(br_all, ignore_index=True)
    b_["evidence_class"] = "Estimated"
    return pd.DataFrame(rows), b_


def h101(imp: pd.DataFrame) -> dict:
    per = {}
    for bank, g in imp.groupby("bank"):
        g = g.set_index("valuation")
        a, c = g.loc["HIST"], g.loc["CURR"]
        rose = c["implied_er_star_at_ch5_lambda"] > a["implied_er_star_at_ch5_lambda"]
        beyond = c["implied_er_star_at_ch5_lambda"] > c["er_star_ch5_hi"]
        per[bank] = "supported" if rose and beyond else "rejected" if not rose else "inconclusive"
    v = "supported" if all(x == "supported" for x in per.values()) else "rejected" if all(
        x == "rejected" for x in per.values()) else "inconclusive"
    ev = "; ".join(f"{r.bank} {r.valuation}: implied ER* at lambda {r.lambda_ch5:.2f} = {pct(r.implied_er_star_at_ch5_lambda)} "
                   f"vs chapter 5 ER* {pct(r.er_star_ch5)} [{pct(r.er_star_ch5_lo)}, {pct(r.er_star_ch5_hi)}]"
                   for r in imp.itertuples())
    return {"hypothesis": "H10.1", "verdict": v, "verdict_by_bank": ", ".join(f"{k} {x}" for k, x in per.items()),
            "evidence": ev, "rule": "per bank: supported when the price-implied ER* at the chapter 5 lambda rose from 31 "
            "Dec 2025 to 31 Oct 2026 AND at 31 Oct 2026 exceeds the upper end of the chapter 5 ER* interval; rejected "
            "when it fell; inconclusive otherwise. Additional assumption: forecast and COE fixed, lambda at its "
            "chapter 5 estimate", "evidence_class": "Interpretation"}


NOTE = """# Chapter 10: market-implied expectations

Generated by `tzbank.chapters.ch10_implied` (stage `valuation_results`). Blueprint §0 (headline metric: estimated vs
implied persistence) and §4 chapter 10; H10.1; D12. **Register {version}, input hash `{hash}`.**

## Research question

How much persistence of excess returns do the prices imply at each date, against what history supports (chapter 5),
and what moved the prices between 31 Dec 2025 and 31 Oct 2026?

## Variable definitions

Iso-value surface: value / price over lambda (0-0.95) x ER* (-5% to +15%), ROE* = COE + ER*, payout* = 1 - g / ROE*,
geometric fade (D56), explicit forecast and COE fixed. Implied ER* at the chapter 5 lambda; implied lambda at the
chapter 5 ER*; implied COE (r at which RIM value = market capitalisation); implied ROE* (r fixed). Rerating: see the
definition column of `ch10_rerating.csv`.

## Sources and transformation trail

`config/assumption_register.csv`; chapter 5 estimates (31 Dec 2025: re-estimated on ER through FY2024 in
`tzbank.valuation.register._pit_persistence`; 31 Oct 2026: `ch05_calibration_candidates.csv`); `tzbank.valuation.models`
(`iso_surface`, `implied_coe`, `implied_roe_star`), `tzbank.valuation.uncertainty.bridge`. Functions: `surfaces`,
`implied_er`, `implied_lambda`, `rerating`, `h101`.

## Validation test

`tests/test_ch09_11.py`: the implied ER* reproduces the price; the rerating components add up; implied COE reproduces
the price.

## Results (Estimated)

### Implied persistence against chapter 5

{imp}

### Rerating, 31 Dec 2025 -> 31 Oct 2026 (log points, total return)

{rer}

### Hypothesis (Interpretation)

{hyp}

## Evidence class

Surfaces, implied values and the decomposition: **Estimated**. Conclusions and H10.1: **Interpretation**.

## Limitations

- A single implied value is not identified (D12): every statement above holds the explicit forecast and the COE fixed
  and fixes either lambda or ER* at its chapter 5 estimate; the full surface is in `ch10_iso_surface.csv`.
- COE is approach A only; the implied COE absorbs any approach-B difference.
- The decomposition's 'earnings and time' term includes accretion at the historical COE; the residual is the change in
  the price-to-value gap, not a pure sentiment measure.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

If the market prices more persistent rent than the chapter 5 interval supports, it is pricing either a structural
change in the rent (a deeper deposit franchise, scale) or a continuation of the post-2018 credit recovery that chapter 8
classes as partly cyclical.
"""


def _lam_txt(x: float, note: str) -> str:
    return f"none ({note})" if not math.isfinite(x) else f"{x:.2f}"


def write_ch10(ctx: Ctx) -> list[Path]:
    root = ctx.root
    surf, imp = surfaces(ctx)
    rer, br = rerating(ctx)
    hyp = pd.DataFrame([h101(imp)])
    tab, figd = path("tables", root), path("figures", root)
    tables = [(ctx.stamp(imp), "ch10_implied.csv", "Implied ER*, lambda, COE and ROE* against chapter 5", "Estimated"),
              (ctx.stamp(surf), "ch10_iso_surface.csv", "Iso-value surface (lambda x ER*)", "Estimated"),
              (ctx.stamp(rer), "ch10_rerating.csv", "Rerating decomposition 31 Dec 2025 -> 31 Oct 2026", "Estimated"),
              (ctx.stamp(br), "ch10_bridge.csv", "Five-step valuation bridge (TZS)", "Estimated"),
              (ctx.stamp(hyp), "ch10_hypotheses.csv", "H10.1", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    fig, axes = plt_subplots(1, 2, figsize=(9.5, 4.2))
    for ax, b in zip(axes, BANKS, strict=True):
        for val, ls in (("HIST", "--"), ("CURR", "-")):
            s = surf[(surf["bank"] == T[b]) & (surf["valuation"] == val)].drop_duplicates("lambda")
            ax.plot(s["lambda"], s["er_star_at_price"] * 100, color=COLORS[b], ls=ls,
                    label=f"price-implied ({'31 Dec 2025' if val == 'HIST' else '31 Oct 2026'})")
            e = imp[(imp["bank"] == T[b]) & (imp["valuation"] == val)].iloc[0]
            ax.errorbar(e["lambda_ch5"], e["er_star_ch5"] * 100,
                        xerr=[[e["lambda_ch5"] - e["lambda_ch5_lo"]], [e["lambda_ch5_hi"] - e["lambda_ch5"]]],
                        yerr=[[(e["er_star_ch5"] - e["er_star_ch5_lo"]) * 100], [(e["er_star_ch5_hi"] - e["er_star_ch5"]) * 100]],
                        fmt="o" if val == "CURR" else "s", color="#1f1f1d", ms=4, capsize=2,
                        label=f"chapter 5 estimate ({'FY<=2024' if val == 'HIST' else 'FY<=2025'})")
        ax.set_xlabel("lambda (persistence)", fontsize=8)
        style(ax, f"{T[b]}: (lambda, ER*) pairs that match the price", "ER*, percent")
        ax.legend(fontsize=6, frameon=False)
    png = save(fig, figd / "ch10_iso_value.png", f"Evidence class: Estimated. Register {ctx.version}, input hash "
               f"{ctx.hash}. Curves: ER* at which value = price for each lambda (forecast and COE fixed).")
    fd = write_csv(ctx.stamp(surf.drop_duplicates(["valuation", "bank", "lambda"])[
        ["valuation", "bank", "lambda", "er_star_at_price", "evidence_class"]]), figd / "ch10_iso_value_data.csv",
        require_evidence=True)
    files += [png, fd]
    imp_txt = "\n".join(
        f"- **{r.bank} {('31 Dec 2025' if r.valuation == 'HIST' else '31 Oct 2026')}** (COE {pct(r.coe)}): chapter 5 "
        f"lambda {r.lambda_ch5:.2f} [{r.lambda_ch5_lo:.2f}, {r.lambda_ch5_hi:.2f}], ER* {pct(r.er_star_ch5)} "
        f"[{pct(r.er_star_ch5_lo)}, {pct(r.er_star_ch5_hi)}] (ER {r.ch5_years}); price-implied ER* at that lambda "
        f"**{pct(r.implied_er_star_at_ch5_lambda)}**; price-implied lambda at that ER* "
        f"{_lam_txt(r.implied_lambda_at_ch5_er_star, r.implied_lambda_note)}"
        f"; implied COE {pct(r.implied_coe)}; implied ROE* {pct(r.implied_roe_star)}. **{r.conclusion}**. Additional "
        f"assumption: {r.additional_assumption}." for r in imp.itertuples())
    rer_txt = "\n".join(
        f"- {r.bank}: market total return {r.market_total_return_log:+.3f} = earnings and time "
        f"{r.earnings_and_time_log:+.3f} + discount rate {r.discount_rate_log:+.3f} (COE {pct(r.coe_hist)} -> "
        f"{pct(r.coe_curr)}) + residual {r.residual_log:+.3f}; value / price {r.value_to_price_hist:.2f} -> "
        f"{r.value_to_price_curr:.2f}." for r in rer.itertuples())
    hr = hyp.iloc[0]
    hyp_txt = f"- **H10.1: {hr['verdict'].upper()}** ({hr['verdict_by_bank']}). {hr['evidence']}. Rule: {hr['rule']}."
    interp = ("**Interpretation.** The residual term measures how much of the 2026 move the "
              "model's earnings and discount-rate changes do not explain; read with the implied ER* at the chapter 5 "
              "lambda, it says whether the rerating priced more persistence than history supports (H10.1).")
    nf = write_text(NOTE.format(version=ctx.version, hash=ctx.hash, imp=imp_txt, rer=rer_txt, hyp=hyp_txt,
                                interp=interp), path("notes", root) / "ch10.md")
    files.append(nf)
    m = {"chapter": "10", "title": "Market-implied expectations", "hypotheses": ["H10.1"],
         "status": f"built - register {ctx.version}, input hash {ctx.hash}",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c_, "note": rel(nf, root)}
                    for _, n, d, c_ in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "Iso-value curves vs chapter 5",
                      "evidence_class": "Estimated", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "register_version": ctx.version, "input_hash": ctx.hash,
         "module": "tzbank.chapters.ch10_implied", "run_log_stage": "valuation_results"}
    files.append(write_json(m, path("manifests", root) / "ch10.json"))
    return files
