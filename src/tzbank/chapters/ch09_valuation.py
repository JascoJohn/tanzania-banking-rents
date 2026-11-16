"""Chapter 9: comparative valuation (blueprint §4 chapter 9; H9.1; VALUATION_SPEC §2, §6, §7, §11; D1, D4).

Per bank and date (register v1.0): RIM and capital-constrained DDM on one forecast, the 60/40 blend, value per share
(post-split NMB shares at 31 Oct 2026), the 12-month target V1 (identity-checked), expected total return, bear-case
downside, the D4 rating with an override log, multiples (justified P/B and P/E against market), sensitivity grids
(COE +/- 200 bp), the ICG growth-cap report, the payout path and the point-in-time verification log.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.banks.engine import project
from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import models as vm
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.context import BANKS, Ctx, build_ctx, pct, shares_post_split
from tzbank.valuation.inputs import eligible_long, pick, pit_violations

T = rg.TICK
BUY, DOWN = 0.05, -0.25  # D4


def rating(exp_ret: float, coe: float, downside: float) -> tuple[str, str]:
    """D4: Buy if expected return > COE + 5 pp and bear downside no worse than -25%; Sell if < COE - 5 pp; else Hold.
    A failed downside test turns a Buy into a Hold, never a Sell."""
    if exp_ret > coe + BUY:
        return ("Buy", "expected return > COE + 5 pp; bear downside within 25%") if downside >= DOWN else \
            ("Hold", "expected return > COE + 5 pp but bear downside worse than 25% (Buy -> Hold, D4)")
    if exp_ret < coe - BUY:
        return "Sell", "expected return < COE - 5 pp"
    return "Hold", "expected return within COE +/- 5 pp"


def _divs_in_window(fc, vd: pd.Timestamp, lag_months: int = 0) -> float:
    """Dividends paid in (vd, vd + 12 months]: the FY t dividend is paid `lag_months` after year-end t (owner issue 17;
    0 = the year-end convention of D56)."""
    v1d = vd + pd.DateOffset(years=1)
    tot = 0.0
    for r in fc.rows[fc.rows["t"] >= 1].itertuples():
        leaves = vm.pay_date(pd.Timestamp(int(r.year), 12, 31), lag_months) + pd.Timedelta(days=1)
        if vd < leaves <= v1d:
            tot += float(r.Div)
    return tot


def dividend_schedule(ctx: Ctx) -> pd.DataFrame:
    """Dividends paid in (vd, vd + 12 months] per share, with their payment dates (prompt 24: the USD return translates
    each dividend at the rate on its own date). Same convention as ``_divs_in_window`` (Blueprint ch. 9 / 12)."""
    rows = []
    for val in rg.DATES:
        for b in BANKS:
            c = ctx.cases[(val, b, "base")]
            fc = project(c.op, c.dr)
            v1d = c.vd + pd.DateOffset(years=1)
            for r in fc.rows[fc.rows["t"] >= 1].itertuples():
                paid = vm.pay_date(pd.Timestamp(int(r.year), 12, 31), c.lag_months)
                if c.vd < paid + pd.Timedelta(days=1) <= v1d:
                    rows.append({"valuation": val, "bank": T[b], "fiscal_year": int(r.year),
                                 "pay_date": paid.date().isoformat(), "years_from_valuation": (paid - c.vd).days / 365.25,
                                 "dps": float(r.Div) / c.shares, "price": c.price, "evidence_class": "Estimated"})
    return pd.DataFrame(rows, columns=["valuation", "bank", "fiscal_year", "pay_date", "years_from_valuation", "dps",
                                       "price", "evidence_class"])


def valuations(ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, caps, pays = [], [], []
    for val in rg.DATES:
        for b in BANKS:
            c = ctx.cases[(val, b, "base")]
            fc = project(c.op, c.dr)
            bl = vm.blend(fc, c.r, c.vd, c.w, lag_months=c.lag_months)
            v1 = vm.blend(fc, c.r, c.vd + pd.DateOffset(years=1), c.w, lag_months=c.lag_months)
            ti = vm.target_identity(fc, c.r, c.vd, lag_months=c.lag_months)
            ti_d = vm.target_identity(fc, c.r, c.vd, "ddm", lag_months=c.lag_months)
            sh = c.shares
            divw = _divs_in_window(fc, c.vd, c.lag_months)
            bear = un.value(ctx.cases[(val, b, "bear")]) / sh
            bull = un.value(ctx.cases[(val, b, "bull")]) / sh
            vps, tp, dps = bl["blend"] / sh, v1["blend"] / sh, divw / sh
            exp_ret = (tp + dps - c.price) / c.price
            down = bear / c.price - 1
            rt, why = rating(exp_ret, c.r, down)
            b0 = ctx.inputs[(val, b)].fy["equity_attributable"]
            np0 = ctx.inputs[(val, b)].fy["profit_attributable"]
            j = vm.justified(c.dr.roe_star, c.dr.g, c.r, c.dr.payout_star)
            post = shares_post_split(ctx, val, b)
            adj = post / sh
            rows.append({"valuation": val, "date": c.vd.date().isoformat(), "bank": T[b], "anchor_fy": c.op.year,
                         "coe": c.r, "rim_equity_tzs": bl["rim"], "ddm_equity_tzs": bl["ddm"],
                         "blend_equity_tzs": bl["blend"], "rim_ddm_gap": bl["gap"],
                         "rim_ddm_flag": "FLAG > 10%" if bl["gap"] > 0.10 else "within 10%",
                         "shares": sh, "value_per_share": vps, "rim_per_share": bl["rim"] / sh,
                         "ddm_per_share": bl["ddm"] / sh, "price": c.price, "value_to_price": vps / c.price,
                         "target_price_12m": tp, "expected_dps_12m": dps, "expected_total_return": exp_ret,
                         "identity_residual": ti["residual"], "identity_residual_ddm": ti_d["residual"],
                         "dividend_pay_lag_months": c.lag_months, "bear_value_per_share": bear,
                         "bull_value_per_share": bull, "bear_downside": down, "rating": rt, "rating_reason": why,
                         "market_pb": c.price * sh / b0, "market_pe": c.price * sh / np0,
                         "model_pb": bl["blend"] / b0, "justified_pb": j["pb"], "justified_pe": j["pe"],
                         "value_per_share_post_split_basis": vps / adj, "price_post_split_basis": c.price / adj,
                         "split_note": "NMB 1:10 split (24 Aug 2026): 31 Dec 2025 per-share values / 10 on the post-split "
                                       "basis" if b == "B02" else "", "evidence_class": "Estimated"})
            ex = fc.rows[fc.rows["phase"] == "explicit"]
            prevL = [c.op.loans, *ex["L"].tolist()[:-1]]
            for r, pl in zip(ex.itertuples(), prevL, strict=True):
                pays.append({"valuation": val, "bank": T[b], "year": int(r.year), "np_attr_tzs": r.NP_attr,
                             "dividend_tzs": r.Div, "payout": r.Div / r.NP_attr if r.NP_attr else np.nan,
                             "tier1_ratio": r.T1 / r.RWA, "k_target": r.k_target, "roe": r.ROE,
                             "loan_growth": r.L / pl - 1, "evidence_class": "Estimated"})
            if "icg_capped_from" in ex:
                prevL = [c.op.loans, *ex["L"].tolist()[:-1]]
                for r, pl in zip(ex.itertuples(), prevL, strict=True):
                    if pd.notna(getattr(r, "icg_capped_from", np.nan)):
                        caps.append({"valuation": val, "bank": T[b], "year": int(r.year),
                                     "register_loan_growth": r.icg_capped_from, "capped_loan_growth": r.L / pl - 1,
                                     "payout_after_cap": r.Div / r.NP_attr, "evidence_class": "Estimated"})
    return pd.DataFrame(rows), pd.DataFrame(caps), pd.DataFrame(pays)


def overrides(root: Path) -> pd.DataFrame:
    """Analyst overrides of the rule-based rating (VALUATION_SPEC §11): config/rating_overrides.csv, empty unless the
    owner adds one."""
    f = root / "config" / "rating_overrides.csv"
    if not f.exists():
        f.write_text("valuation,bank,rule_rating,override_rating,rationale,owner,date\n", encoding="utf-8")
    o = pd.read_csv(f)
    return o.assign(evidence_class="Interpretation")


def pit_log(ctx: Ctx) -> pd.DataFrame:
    """Point-in-time verification of the historical run's input set (quality gate)."""
    rows = []
    for b in BANKS:
        bi = ctx.inputs[("HIST", b)]
        for k, p in bi.sources.items():
            ok = not (p and p[:4].isdigit() and pd.Timestamp(p) > bi.vd)
            rows.append({"bank": T[b], "input": k, "publication_date": p, "valuation_date": "2025-12-31",
                         "eligible": "yes" if ok else "NO", "evidence_class": "Observed"})
        e = eligible_long(ctx.root, "2025-12-31")
        v, _ = pick(e, b, "net_profit", "FY", 2025)
        rows.append({"bank": T[b], "input": "test: audited FY2025 net profit cannot enter the 31 Dec 2025 run",
                     "publication_date": "n/a", "valuation_date": "2025-12-31",
                     "eligible": "PASS" if pd.isna(v) else "FAIL", "evidence_class": "Observed"})
        rows.append({"bank": T[b], "input": "test: no input published after 31 Dec 2025 (pit_violations)",
                     "publication_date": "n/a", "valuation_date": "2025-12-31",
                     "eligible": "PASS" if not pit_violations(bi) else "FAIL", "evidence_class": "Observed"})
    return pd.DataFrame(rows)


def crdb_dividend_sensitivity(ctx: Ctx) -> dict:
    """Owner note (09b pre-flight): the rule gives CRDB a zero FY2025 dividend in the 31 Dec 2025 run, but CRDB declared
    TZS 90 a share (published 14 Apr 2026, after the cut-off). One-line sensitivity: the run with that dividend."""
    c = ctx.cases[("HIST", "B01", "base")]
    e = eligible_long(ctx.root, "2026-10-31")
    dps = e[(e["bank_id"] == "B01") & (e["canonical_name"] == "mkt_dividend_per_share_fy2025_final")]["value"]
    d = float(dps.iloc[-1]) * c.shares if len(dps) else np.nan
    fc0 = project(c.op, c.dr)
    fc1 = project(c.op, replace(c.dr, div1_override=d))
    out = {"rule_div_fy2025": float(fc0.rows.loc[fc0.rows["t"] == 1, "Div"].iloc[0]), "actual_div_fy2025": d}
    for m in ("rim", "ddm"):
        v0 = vm.value_at(fc0, c.r, c.vd, m, lag_months=c.lag_months)
        v1 = vm.value_at(fc1, c.r, c.vd, m, lag_months=c.lag_months)
        out[f"{m}_change"] = v1 / v0 - 1
    out["checks_pass"] = all(k["status"] == "PASS" for k in fc1.checks if k["check"].startswith("Tier 1"))
    return out


def grids(ctx: Ctx) -> pd.DataFrame:
    vc = settings(ctx.root)["valuation"]
    out = []
    for val in rg.DATES:
        for b in BANKS:
            for name, df in un.grids(ctx.cases[(val, b, "base")], vc["grid_step_bp"], vc["grid_half_width_bp"]).items():
                out.append(df.assign(valuation=val, bank=T[b], grid=name))
    g = pd.concat(out, ignore_index=True)
    g["evidence_class"] = "Scenario"
    return g


def h91(v: pd.DataFrame, ctx: Ctx) -> dict:
    """D57 (accepted, H9.1 amended by the owner): a material difference = the price lies outside the Monte Carlo
    P10-P90 value range at 31 Dec 2025."""
    per, ev, cur = {}, [], {}
    for r in v.itertuples():
        b = "B02" if r.bank == "NMB" else "B01"
        mc = ctx.monte_carlo(r.valuation, b)["value_per_share"].dropna()
        p10, p90 = float(mc.quantile(0.10)), float(mc.quantile(0.90))
        out = not p10 <= r.price <= p90
        side = "above P90" if r.price > p90 else "below P10" if r.price < p10 else "inside P10-P90"
        reading = "material difference" if out else "no material difference"
        (per if r.valuation == "HIST" else cur)[r.bank] = ("supported" if out else "rejected") if r.valuation == "HIST" \
            else reading
        ev.append(f"{r.bank} {r.date}: P(value > price) {pct(float((mc > r.price).mean()), 0)}; price {r.price:,.0f} "
                  f"{side} [{p10:,.0f}, {p90:,.0f}] (value / price {r.value_to_price:.2f}) -> {reading}")
    verdict = "supported" if all(x == "supported" for x in per.values()) else "rejected" if all(
        x == "rejected" for x in per.values()) else "inconclusive"
    return {"hypothesis": "H9.1", "verdict": verdict, "verdict_by_bank": ", ".join(f"{k} {x}" for k, x in per.items()),
            "verdict_by_bank_date": "; ".join([f"{k} 31 Dec 2025 {x}" for k, x in per.items()]
                                              + [f"{k} 31 Oct 2026 {x} (supplementary reading)" for k, x in cur.items()]),
            "evidence": "; ".join(ev), "rule": "D57 (accepted, H9.1 amended): per bank at 31 Dec 2025, supported when the "
            "price lies outside the Monte Carlo P10-P90 value range (a material difference), rejected when inside; joint "
            "verdict needs both banks (inconclusive when they differ). The same test at 31 Oct 2026 is reported per bank "
            "as a supplementary reading; it does not enter the H9.1 verdict, which D57 defines at 31 Dec 2025",
            "evidence_class": "Interpretation"}


NEAR = 0.10  # a price within 10% of a P10 / P90 bound is "near the cut-off" (owner, ch01-03 review B)


def headline(mc: pd.DataFrame) -> str:
    """Owner, ch01-03 review B: chapters 9 and 11 lead with P(value > price) from the primary (base-centred) Monte
    Carlo and state where the price sits relative to the H9.1 cut-offs (P10 / P90)."""
    prim = mc[mc["variant"].str.startswith("base")] if "variant" in mc else mc
    lines = ["**Headline: probability that value exceeds the price** (Monte Carlo, base-centred, D58 amended):", ""]
    for val, lab in (("HIST", "31 Dec 2025"), ("CURR", "31 Oct 2026")):
        parts = []
        for r in prim[prim["valuation"] == val].itertuples():
            gap = min(abs(r.price - r.p10), abs(r.price - r.p90)) / r.price
            pos = "inside" if r.p10 <= r.price <= r.p90 else "outside"
            near = " - NEAR the cut-off" if gap < NEAR else ""
            parts.append(f"{r.bank} {pct(r.prob_value_above_price, 0)} (price {r.price:,.0f} {pos} P10-P90 "
                         f"[{r.p10:,.0f}, {r.p90:,.0f}], {pct(gap, 1)} from the nearer bound{near})")
        lines.append(f"- {lab}: " + "; ".join(parts) + ".")
    lines += ["", "The H9.1 verdicts are binary readings of these distributions; where a price sits near P10 or P90, a "
              "small change in inputs or seed can flip the verdict, so the probability is the primary result."]
    return "\n".join(lines)


NOTE = """# Chapter 9: comparative valuation

Generated by `tzbank.chapters.ch09_valuation` (stage `valuation_results`). Blueprint §4 chapter 9; H9.1; VALUATION_SPEC
§2, §6, §7, §11; D1 (cut-off), D4 (rating rule), D50, D55, D56. **Register {version}, input hash `{hash}`** (every
table row carries both). Repository private; publication is blocked by open issue 19 (CMSA licensing).

## Research question

What are NMB and CRDB worth at 31 Dec 2025 (information public by then) and at 31 Oct 2026, and how do those values
compare with the prices?

## Variable definitions

Blend = 60% RIM + 40% capital-constrained DDM (VALUATION_SPEC §2); value per share = blend / ordinary shares at the
date (NMB post-split at 31 Oct 2026); target price = V1 per share at the date + 12 months (identity V1 = V0 (1 + r)
- dividends carried, checked); expected total return = (target + expected DPS - price) / price; downside = bear value
per share / price - 1; rating per D4. Definitions in `docs/VARIABLE_DICTIONARY.md` (v_rim, v_ddm, k_star, ...).

## Sources and transformation trail

`config/assumption_register.csv` ({version}); `tzbank.valuation.inputs` (point-in-time data), `tzbank.banks.engine`,
`tzbank.valuation.models`, `tzbank.valuation.uncertainty`. Functions here: `valuations`, `rating`, `overrides`,
`pit_log`, `crdb_dividend_sensitivity`, `grids`, `h91`.

## Validation test

`tests/test_valuation_golden.py` (hand-built bank), `tests/test_valuation_pit.py`, `tests/test_ch09_11.py` (rating rule
hand cases, identity residuals, stamps, PIT log all PASS).

## Results (Estimated)

{lead}

{results}

### Capital: where the ICG growth cap binds, and the payout path (D56)

{caps}

{payout}

### Multiples cross-check

{mult}

Peer regression (P/B on ROE, East African banks): **not computed**: peer data are TO COLLECT
(`data/interim/inputs_peer_banks.csv`, {peer_held} of {peer_n} values held).

### Point-in-time verification (31 Dec 2025 run)

{pit}

### Hypothesis (Interpretation)

{hyp}

## Evidence class

Values, targets, returns and multiples: **Estimated** (they rest on the register's judgements). Sensitivity grids:
**Scenario**. Rating and H9.1: **Interpretation**. PIT log: **Observed**.

## Limitations

- COE is approach A only (approach B needs forward inflation, open issue 14); terminal g is a placeholder (issue 15).
- {crdb_div}
- The ICG cap lowers CRDB's base growth where internal capital generation cannot fund it; the register's growth path
  is kept and the capped path is reported.
- Dividends are treated at fiscal year-end; interim dividends are TO COLLECT.
- Peer multiples are not available; the cross-check is against justified multiples only.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

The base value capitalises the through-the-cycle excess return (D50 base): NMB's positive ER* (a durable franchise,
chapter 6, plus scale, chapter 4.5) carries most of its premium to book; CRDB's base ER* is close to zero, so its value
sits near book and rests on growth and the cycle (chapter 8).
"""


def coe_ab(ctx: Ctx) -> pd.DataFrame:
    """COE approach A (central, owner 2026-11-12) against approach B computed from the IMF inflation forecasts (D64,
    issue 14): B = (1 + rf_US + beta (ERP + CRP) + LP) (1 + pi_TZ) / (1 + pi_US) - 1 (VALUATION_SPEC §5)."""
    from tzbank.valuation import coe as coe_mod
    from tzbank.valuation.run import forward_inflation

    vc = settings(ctx.root)["valuation"]
    rows = []
    for val in rg.DATES:
        for b in BANKS:
            c, bi = ctx.cases[(val, b, "base")], ctx.inputs[(val, b)]
            reg = ctx.reg[(ctx.reg["valuation"] == val) & (ctx.reg["bank"] == T[b])].set_index("symbol")["base"]
            pz, pu = forward_inflation(vc, bi)
            bb = coe_mod.approach_b(bi.macro["rf_us"], float(reg["beta"]), bi.macro["erp"], bi.macro["crp"],
                                    float(reg["LP"]), pz, pu)
            rec = coe_mod.reconcile(float(reg["COE_A"]), bb, vc["coe_reconcile_bp"])
            rows.append({"valuation": val, "bank": T[b], "coe_a": float(reg["COE_A"]), "coe_b": bb,
                         "a_minus_b": float(reg["COE_A"]) - bb, "pi_tz": pz, "pi_us": pu, "coe_used": c.r,
                         "central_rule": central_text(vc.get("coe_central", "A"), float(reg["COE_A"]), bb,
                                                      float(vc["coe_reconcile_bp"])),
                         "spec_rule": rec.rule, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def central_text(central: str, a: float, b: float, bp: float) -> str:
    """Report wording of the central cost-of-equity choice (prompt 23; VALUATION_SPEC cost of equity): the author's
    approach, then what the specification's reconciliation rule would do with these A and B. ``central`` other than
    "A" means the registered rule itself is applied (D93, prompt 24)."""
    if central != "A":
        if not np.isfinite(b):
            return "Registered rule: approach B cannot be computed, so approach A is used"
        if abs(a - b) * 1e4 < bp:
            return f"Registered rule, applied as registered: A and B differ by less than {bp:.0f} bp, so the midpoint"
        return f"Registered rule: A and B differ by {bp:.0f} bp or more, so approach A"
    who = f"Approach {central} (author's choice)"
    if not np.isfinite(b):
        return f"{who}; Approach B cannot be computed, so the specification's rule would also use A"
    if abs(a - b) * 1e4 < bp:
        return (f"{who}; the specification's rule would use the midpoint when A and B differ by less than "
                f"{bp:.0f} bp")
    return f"{who}; A and B differ by {bp:.0f} bp or more, where the specification's rule also keeps A"


def write_ch09(ctx: Ctx) -> list[Path]:
    root = ctx.root
    v, caps, pays = valuations(ctx)
    ov = overrides(root)
    pl = pit_log(ctx)
    sens = crdb_dividend_sensitivity(ctx)
    gr = grids(ctx)
    hyp = pd.DataFrame([h91(v, ctx)])
    cab = coe_ab(ctx)
    ka = ctx.reg[ctx.reg["symbol"].isin(["COE_A", "ROE_star", "ER_star", "g", "g_fade", "lambda", "LP", "beta", "fx",
                                         "payout_star"]) | ((ctx.reg["symbol"] == "rate")
                                                            & (ctx.reg["period"] == ctx.reg.groupby(["valuation", "bank",
                                                                                                    "symbol"])["period"]
                                                               .transform("min")))][
        ["valuation", "bank", "symbol", "parameter", "period", "base", "bear", "bull", "rationale"]].assign(
        evidence_class="Scenario")
    # D93: ROE*, the terminal payout and the cost of equity as the model uses them (central COE + ER*), not the
    # register's approach-A numbers
    ka = ka.copy()
    for (val, b, sc), c in ctx.cases.items():
        m = (ka["valuation"] == val) & (ka["bank"] == T[b])
        if sc in ("base", "bear", "bull"):
            ka.loc[m & (ka["symbol"] == "ROE_star"), sc] = c.dr.roe_star
            ka.loc[m & (ka["symbol"] == "payout_star"), sc] = c.dr.payout_star
    ka = pd.concat([ka, pd.DataFrame([{"valuation": val, "bank": T[b], "symbol": "COE", "parameter":
                                       "Cost of equity used (central, registered rule)", "period": "all",
                                       **{sc: ctx.cases[(val, b, sc)].r for sc in ("base", "bear", "bull")},
                                       "rationale": "valuation.coe_central (D93)", "evidence_class": "Scenario"}
                                      for val in rg.DATES for b in BANKS])], ignore_index=True)
    sd = pd.DataFrame([{"valuation": "HIST", "bank": "CRDB", **{k: v for k, v in sens.items()},
                        "evidence_class": "Scenario"}])
    guard = pd.DataFrame([{"valuation": v, "bank": T[b], "scenario": sc, "coe": c.r, "g": c.dr.g,
                           "g_fade_start": c.dr.g_fade_start, "coe_minus_g": c.r - c.dr.g,
                           "min_gap": float(settings(root)["valuation"].get("coe_g_min_gap", 0.04)),
                           "status": "PASS" if c.r - c.dr.g >= float(settings(root)["valuation"].get("coe_g_min_gap",
                                                                                                      0.04)) - 1e-12
                           else "STOP", "evidence_class": "Derived"}
                          for (v, b, sc), c in ctx.cases.items()])
    tab, figd = path("tables", root), path("figures", root)
    tables = [(ctx.stamp(v), "ch09_valuations.csv", "Values, target, expected return, downside, rating", "Estimated"),
              (ctx.stamp(ov), "ch09_override_log.csv", "Rating override log (empty unless the owner adds one)",
               "Interpretation"),
              (ctx.stamp(gr), "ch09_sensitivity_grids.csv", "Sensitivity grids: COE x g, COE x ROE*, NIM x CoR "
                                                             "(+/- 200 bp)", "Scenario"),
              (ctx.stamp(caps if len(caps) else pd.DataFrame(columns=["valuation", "bank", "year", "evidence_class"])),
               "ch09_icg_cap.csv", "Where the ICG growth cap binds (D56)", "Estimated"),
              (ctx.stamp(guard), "ch09_coe_g_guard.csv", "COE - terminal g in every scenario (guard >= 4 pp, D64 "
               "amended)", "Derived"),
              (ctx.stamp(ka), "ch09_key_assumptions.csv", "Key register assumptions (base / bear / bull)",
               "Scenario"),
              (ctx.stamp(sd), "ch09_crdb_dividend_sensitivity.csv", "CRDB 31 Dec 2025: rule dividend vs the declared "
                                                                    "FY2025 dividend (D1 effect)", "Scenario"),
              (ctx.stamp(cab), "ch09_coe_ab.csv", "COE approach A (central) vs approach B (IMF inflation, D64)",
               "Derived"),
              (ctx.stamp(pays), "ch09_payout_path.csv", "Explicit-period payout, Tier 1 and target path", "Estimated"),
              (ctx.stamp(dividend_schedule(ctx)), "ch09_dividends_12m.csv", "Dividends in the 12 months after each "
               "valuation date, with payment dates (prompt 24)", "Estimated"),
              (ctx.stamp(pl), "ch09_pit_verification.csv", "Point-in-time verification of the 31 Dec 2025 run",
               "Observed"),
              (ctx.stamp(hyp), "ch09_hypotheses.csv", "H9.1", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    # valuation range chart (VALUATION_SPEC §7), per bank at the current date
    fig, axes = plt_subplots(1, 2, figsize=(9, 4.2))
    figdata = []
    for ax, b in zip(axes, BANKS, strict=True):
        x = v[(v["valuation"] == "CURR") & (v["bank"] == T[b])].iloc[0]
        mc = ctx.monte_carlo("CURR", b)["value_per_share"].dropna()
        bars = [("RIM", x["rim_per_share"]), ("DDM", x["ddm_per_share"]), ("Blend", x["value_per_share"]),
                ("Bear", x["bear_value_per_share"]), ("Bull", x["bull_value_per_share"]),
                ("MC P10", mc.quantile(0.10)), ("MC P90", mc.quantile(0.90)),
                ("Justified P/B, sustainable ROE", x["justified_pb"] * ctx.inputs[("CURR", b)].fy["equity_attributable"] / x["shares"])]
        ax.barh([k for k, _ in bars], [y for _, y in bars], color=COLORS[b])
        ax.axvline(x["price"], color="#1f1f1d", lw=1.2)
        style(ax, f"{T[b]} at 31 Oct 2026 (TZS a share)", "")
        figdata += [{"bank": T[b], "item": k, "value_per_share": y} for k, y in bars] + [
            {"bank": T[b], "item": "price", "value_per_share": x["price"]}]
    png = save(fig, figd / "ch09_valuation_range.png", f"Evidence class: Estimated. Register {ctx.version}, input "
               f"hash {ctx.hash}. Vertical line: price at 31 Oct 2026. MC: Monte Carlo (chapter 11).")
    fd = write_csv(ctx.stamp(pd.DataFrame(figdata).assign(evidence_class="Estimated")),
                   figd / "ch09_valuation_range_data.csv", require_evidence=True)
    files += [png, fd]
    # note
    res = []
    for r in v.itertuples():
        res.append(f"- **{r.bank} {r.date}** (anchor FY{r.anchor_fy}, COE {pct(r.coe)}): RIM {r.rim_per_share:,.0f}, "
                   f"DDM {r.ddm_per_share:,.0f}, blend **{r.value_per_share:,.0f}** a share vs price {r.price:,.0f} "
                   f"(value / price {r.value_to_price:.2f}; RIM-DDM gap {pct(r.rim_ddm_gap)}, {r.rim_ddm_flag}); "
                   f"target {r.target_price_12m:,.0f} + DPS {r.expected_dps_12m:,.0f} -> expected total return "
                   f"{pct(r.expected_total_return)}; bear {r.bear_value_per_share:,.0f} (downside "
                   f"{pct(r.bear_downside)}), bull {r.bull_value_per_share:,.0f}; **rating {r.rating}** "
                   f"({r.rating_reason}); identity residual {r.identity_residual:.1e}."
                   + (f" Post-split basis: value {r.value_per_share_post_split_basis:,.0f}, price "
                      f"{r.price_post_split_basis:,.0f}." if r.bank == "NMB" and r.valuation == "HIST" else ""))
    res.append("\n**Cost of equity, approach A (central) vs B** (IMF inflation forecasts, D64): " + "; ".join(
        f"{r.bank} {r.valuation} A {pct(r.coe_a, 2)}, B {pct(r.coe_b, 2)} (A - B {r.a_minus_b:+.2%}; "
        f"pi_TZ {pct(r.pi_tz)}, pi_US {pct(r.pi_us)})" for r in cab.itertuples())
        + ". A stays central (owner, 2026-11-12); `config/settings.yaml` `valuation.coe_central`.")
    res.append(f"\nOverride log: {len(ov)} override(s) (`config/rating_overrides.csv`).")
    res.append("\n**Per-share basis.** NMB split 1:10 on 24 Aug 2026 (`config/corporate_actions.csv`). The 31 Dec 2025 "
               "NMB rows are per pre-split share, as traded on that date; their post-split equivalents (/ 10) are in "
               "`value_per_share_post_split_basis` and `price_post_split_basis` and are quoted above. The bridge "
               "(chapter 10) and the tested call (chapter 11) use the post-split basis throughout.")
    cap_txt = ("ICG growth cap binds: " + "; ".join(
        f"{r.bank} {r.valuation} FY{r.year}: loan growth {pct(r.register_loan_growth)} -> {pct(r.capped_loan_growth)} "
        f"(payout {pct(r.payout_after_cap)})" for r in caps.itertuples()) if len(caps) else "The ICG cap does not bind.")
    cr = [(v, ctx.cases[(v, "B01", "base")]) for v in rg.DATES]
    cap_txt = cap_txt.rstrip(".") + (
        ". No share-issuance sensitivity is run (owner, issue 20 resolved): CRDB's base sustainable ROE is close to "
        "its cost of equity (" + "; ".join(f"{v} ROE* {pct(c.dr.roe_star, 2)} vs COE {pct(c.r, 2)}"
                                                      for v, c in cr)
                + "), so growth funded by new equity would earn about its cost and add approximately no value; "
                  "issuance would change the per-share path, not the value per share, to a first approximation.")
    pp = pays[pays["bank"] == "CRDB"]
    pay_txt = "CRDB payout path (base): " + "; ".join(
        f"{r.valuation} FY{r.year} {pct(r.payout)} (Tier 1 {pct(r.tier1_ratio)} vs target {pct(r.k_target)})"
        for r in pp.itertuples())
    low = pp[(pp["year"] >= pp.groupby("valuation")["year"].transform("min") + 1) & (pp["payout"] < 0.20 - 1e-9)]
    flag = low.groupby("valuation").size()
    pay_txt += ("\n\nOwner flag rule (payout below 20% in two or more of FY+2..FY+5): " +
                ("TRIGGERED for " + ", ".join(flag[flag >= 2].index) + ": base loan growth is inconsistent with "
                 "internal capital generation (OPEN_ISSUES)" if (flag >= 2).any() else "not triggered (" + ", ".join(
                     f"{k}: {int(flag.get(k, 0))} year(s) below 20%" for k in rg.DATES) + ")."))
    mult = "\n".join(f"- {r.bank} {r.date}: market P/B {r.market_pb:.2f}, P/E {r.market_pe:.1f}; model P/B "
                     f"{r.model_pb:.2f}; justified P/B {r.justified_pb:.2f}, P/E {r.justified_pe:.1f}."
                     for r in v.itertuples())
    peers = pd.read_csv(path("interim", root) / "inputs_peer_banks.csv")
    pit_txt = (f"{(pl['eligible'].isin(['yes', 'PASS'])).sum()} of {len(pl)} entries pass "
               f"(`ch09_pit_verification.csv`); every 31 Dec 2025 input is published on or before the date and the "
               "audited FY2025 rows are excluded.")
    hrow = hyp.iloc[0]
    hyp_txt = f"- **H9.1: {hrow['verdict'].upper()}** ({hrow['verdict_by_bank_date']}). {hrow['evidence']}. Rule: {hrow['rule']}."
    crdb_div = (f"The capital rule gives CRDB a zero FY2025 dividend in the 31 Dec 2025 run (rule "
                f"TZS {sens['rule_div_fy2025'] / 1e9:,.0f}bn), whereas CRDB declared TZS 90 a share "
                f"(TZS {sens['actual_div_fy2025'] / 1e9:,.0f}bn, published after the cut-off), so the historical CRDB "
                f"DDM is conservatively biased. Sensitivity with the actual dividend: RIM {pct(sens['rim_change'], 2)}, "
                f"DDM {pct(sens['ddm_change'], 2)}" + ("" if sens["checks_pass"] else
                                                      " (Tier 1 would fall below its year-2 path)") + ".")
    cur = v[v["valuation"] == "CURR"].set_index("bank")
    interp = ("**Interpretation.** " + " ".join(
        f"At 31 Oct 2026 the blended value of {k} is {x['value_to_price']:.2f} times its price, which gives a "
        f"rule-based {x['rating']}." for k, x in cur.iterrows()) +
        " The values depend most on the cost of equity and the sustainable ROE (see the register tornado); the D4 "
        "rating is mechanical and every override would be logged.")
    from tzbank.chapters.ch11_scenarios import mc_summary

    text = NOTE.format(version=ctx.version, hash=ctx.hash, lead=headline(mc_summary(ctx)),
                       results="\n".join(res), caps=cap_txt, payout=pay_txt,
                       mult=mult, peer_held=int(peers["value_numeric"].notna().sum()), peer_n=len(peers), pit=pit_txt,
                       hyp=hyp_txt, crdb_div=crdb_div, interp=interp)
    nf = write_text(text, path("notes", root) / "ch09.md")
    files.append(nf)
    m = {"chapter": "09", "title": "Comparative valuation", "hypotheses": ["H9.1"],
         "status": f"built - register {ctx.version}, input hash {ctx.hash}",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c_, "note": rel(nf, root)}
                    for _, n, d, c_ in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "Valuation range at 31 Oct 2026",
                      "evidence_class": "Estimated", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "register_version": ctx.version, "input_hash": ctx.hash,
         "module": "tzbank.chapters.ch09_valuation", "run_log_stage": "valuation_results"}
    files.append(write_json(m, path("manifests", root) / "ch09.json"))
    return files


def write_valuation_results(root: Path | None = None) -> list[Path]:
    """Stage `valuation_results`: chapters 9, 10 and 11 on one context (one Monte Carlo per bank and date)."""
    from tzbank.chapters.ch10_implied import write_ch10
    from tzbank.chapters.ch11_scenarios import write_ch11
    from tzbank.valuation.approve import key_drivers, ordering_check, scenario_ordering

    root = root or find_root()
    files = key_drivers(root)
    ctx = build_ctx(root)
    oc = ordering_check(pd.read_csv(path("outputs", root) / "valuation" / "register_tornado.csv"))
    so = scenario_ordering(ctx)
    log = pd.concat([oc.assign(check="driver: bear alone lowers, bull alone raises the base value"), so],
                    ignore_index=True)
    files.append(write_csv(ctx.stamp(log), path("outputs", root) / "valuation" / "ordering_check.csv",
                           require_evidence=True))
    bad = log[log["status"] == "FAIL"]
    if len(bad):
        from tzbank.pipeline import StageFailed

        raise StageFailed("D56 ordering rule violated (stop for the owner): " + "; ".join(
            f"{r.valuation} {r.bank} {getattr(r, 'symbol', '') if isinstance(getattr(r, 'symbol', ''), str) else ''} "
            f"{r.check}" for r in bad.itertuples()))
    return files + write_ch09(ctx) + write_ch10(ctx) + write_ch11(ctx)
