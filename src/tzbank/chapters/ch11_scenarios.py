"""Chapter 11: strategic divergence scenarios (blueprint §4 chapter 11; H11.1; VALUATION_SPEC §7).

Bear / base / bull per bank and date from the register (each bear driver tied to its chapter 6-8 evidence); the seeded
Monte Carlo (10,000 draws) with the probability that value exceeds price; the tested historical call (the 31 Dec 2025
value and rating against what happened to 31 Oct 2026, decomposed with the bridge); H11.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.banks.engine import project
from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.chapters.ch09_valuation import headline, rating
from tzbank.config import path, settings
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import models as vm
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.context import BANKS, Ctx, pct

T = rg.TICK
EVIDENCE = {"g_L": "ch. 6 (loan growth in the 2015-18 stress, ch06_liquidity)",
            "g_D": "ch. 6 (deposits follow loans; stress-period growth)",
            "CoR": "ch. 8 (2015-18 stress mean cost of risk, ch08_credit_history)",
            "rate": "ch. 6 (BoT easing path; rate-linked NIM via betas)",
            "deposit_beta": "ch. 6 (deposit beta interval, ch06_deposit_beta)",
            "asset_beta": "ch. 4.3 / 6 (IEA yield on T-bill)",
            "NIM": "ch. 4.6 (NIM range FY2019-25)", "CIR": "ch. 4.5 (cost-to-income range)",
            "phi": "ch. 4.1 / 4.4 / 4.7 (fee income; mobile-money competition)",
            "rho": "ch. 7 (RWA density; Basel II/III)", "k_star": "ch. 7 (requirement, headroom, D-SIB)",
            "pi_max": "ch. 7 (payout history)", "s_W": "ch. 6 (cost of borrowings over deposits)",
            "fx": "subsidiary FX (VALUATION_SPEC §7 bear story)", "u": "ch. 7 / subsidiary layer (upstreaming)",
            "ROE_star": "ch. 5 (D50: bear ER* = 0)", "g": "macro placeholder (open issue 15)",
            "COE_A": "market inputs + beta / LP", "t": "tax history"}


def scenario_tables(ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, evid = [], []
    for val in rg.DATES:
        for b in BANKS:
            sc = un.scenarios({s: ctx.cases[(val, b, s)] for s in ("bear", "base", "bull")})
            rows.append(sc.assign(valuation=val, bank=T[b], price=ctx.cases[(val, b, "base")].price))
            x = ctx.reg[(ctx.reg["valuation"] == val) & (ctx.reg["bank"] == T[b])]
            for sym, g in x.groupby("symbol"):
                if (g["bear"] == g["base"]).all() and (g["bull"] == g["base"]).all():
                    continue
                g = g.sort_values("period")
                evid.append({"valuation": val, "bank": T[b], "symbol": sym, "parameter": g["parameter"].iloc[0],
                             "bear_first_last": f"{g['bear'].iloc[0]:.4f} -> {g['bear'].iloc[-1]:.4f}",
                             "base_first_last": f"{g['base'].iloc[0]:.4f} -> {g['base'].iloc[-1]:.4f}",
                             "bull_first_last": f"{g['bull'].iloc[0]:.4f} -> {g['bull'].iloc[-1]:.4f}",
                             "evidence_chapter": EVIDENCE.get(sym, "register rationale"),
                             "evidence_class": "Scenario"})
    out = pd.concat(rows, ignore_index=True)
    out["evidence_class"] = "Scenario"
    return out, pd.DataFrame(evid)


def mc_summary(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for val in rg.DATES:
        for b in BANKS:
            for centred, label in ((True, "base-centred (primary)"), (False, "bull-centred (sensitivity)")):
                v = ctx.monte_carlo(val, b, centred)["value_per_share"]
                c = ctx.cases[(val, b, "base")]
                x = v.dropna()
                rows.append({"valuation": val, "bank": T[b], "variant": label, "draws": len(v), "valid": len(x),
                             "price": c.price, "er_shift": ctx.er_shift(val, b) if centred else 0.0,
                             **{f"p{q}": float(x.quantile(q / 100)) for q in (5, 10, 50, 90, 95)},
                             "mean": float(x.mean()), "prob_value_above_price": float((x > c.price).mean()),
                             "persistence_replicates_kept": float(ctx.persistence_draws(val, b)["kept_share"].iloc[0]),
                             "seed": f"{settings(ctx.root)['random_seed']} (+ stream for date and bank)",
                             "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def tested_call(ctx: Ctx, bridge: pd.DataFrame) -> pd.DataFrame:
    """The 31 Dec 2025 value and rating against actual outcomes to 31 Oct 2026 (old-share basis for NMB)."""
    ca = pd.read_csv(ctx.root / "config" / "corporate_actions.csv")
    split = {r.bank_id: float(r.ratio) for r in ca.itertuples() if r.action == "split"}
    rows = []
    for b in BANKS:
        h, cu = ctx.cases[("HIST", b, "base")], ctx.cases[("CURR", b, "base")]
        f = split.get(b, 1.0)
        sh_post = h.shares * f  # 31 Dec 2025 shares on the post-split basis (owner review D)
        fc = project(h.op, h.dr)
        vps = un.value(h) / sh_post
        tp = vm.blend(fc, h.r, h.vd + pd.DateOffset(years=1), h.w, lag_months=h.lag_months)["blend"] / sh_post
        bear = un.value(ctx.cases[("HIST", b, "bear")]) / sh_post
        div_ps = float(fc.rows.loc[fc.rows["t"] == 1, "Div"].iloc[0]) / sh_post
        p0 = h.price / f
        exp_ret = (tp + div_ps - p0) / p0
        rt, _ = rating(exp_ret, h.r, bear / p0 - 1)
        bc = ctx.inputs[("CURR", b)]
        dps_paid = bc.final_dps_anchor / f  # FY2025 final, declared per pre-split share, split-adjusted
        p1 = cu.price  # 31 Oct 2026 price, post-split
        yrs = (cu.vd - h.vd).days / 365.25
        tsr = (p1 + dps_paid) / p0 - 1
        coe_p = (1 + h.r) ** yrs - 1
        borne = {"Buy": tsr > coe_p, "Sell": tsr < coe_p, "Hold": abs(tsr - coe_p) <= 0.05 * yrs}[rt]
        br = bridge[bridge["bank"] == T[b]].set_index("step")["tzs"] / sh_post
        value_rolled = vps * (1 + h.r) ** yrs - dps_paid
        rows.append({"bank": T[b], "basis": "post-split shares (NMB 1:10, 24 Aug 2026; 31 Dec 2025 figures / 10)"
                     if f != 1 else "ordinary shares (no split)",
                     "hist_value_per_share": vps, "hist_price": p0, "hist_rating": rt,
                     "hist_expected_return_12m": exp_ret, "hist_coe": h.r,
                     "actual_price_31oct2026_same_basis": p1, "dividend_paid_per_share": dps_paid,
                     "actual_total_return": tsr, "coe_over_period": coe_p, "months": round(yrs * 12, 1),
                     "call_borne_out": "yes" if borne else "no",
                     "hist_value_rolled_to_31oct2026": value_rolled,
                     "curr_value_same_basis": br.get("current value", np.nan),
                     "bridge_time": br.get("1 passage of time"), "bridge_dividends": br.get("2 dividends paid"),
                     "bridge_new_information": br.get("3 new information"),
                     "bridge_forecast_revisions": br.get("4 forecast revisions"),
                     "bridge_coe": br.get("5 change in COE"), "bridge_residual": br.get("residual (reported, not absorbed)"),
                     "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def h111(sc: pd.DataFrame, mc: pd.DataFrame) -> dict:
    ch = {}
    for (val, bank), g in sc.groupby(["valuation", "bank"]):
        g = g.set_index("scenario")
        ch[(val, bank)] = g.loc["bear", "value_per_share"] / g.loc["base", "value_per_share"] - 1
    diffs = {v: abs(ch[(v, "NMB")] - ch[(v, "CRDB")]) for v in rg.DATES}
    ok = all(d > 0.10 for d in diffs.values())
    none = all(d <= 0.05 for d in diffs.values())
    prim = mc[mc["variant"].str.startswith("base")] if "variant" in mc else mc  # primary (base-centred) run
    pv = prim.set_index(["valuation", "bank"])["prob_value_above_price"]
    ev = "; ".join(f"{v}: bear vs base NMB {pct(ch[(v, 'NMB')])}, CRDB {pct(ch[(v, 'CRDB')])} (difference "
                   f"{pct(diffs[v])}); P(value > price) NMB {pct(pv[(v, 'NMB')], 0)}, CRDB {pct(pv[(v, 'CRDB')], 0)}"
                   for v in rg.DATES)
    return {"hypothesis": "H11.1", "verdict": "supported" if ok else "rejected" if none else "inconclusive",
            "verdict_by_bank": "joint (NMB vs CRDB)", "evidence": ev,
            "rule": "the same bear macro story (BoT easing, credit cycle, stress growth) applied to both banks: supported "
                    "when the bear-to-base value change differs between the banks by more than 10 pp at both dates; "
                    "rejected when within 5 pp at both; inconclusive otherwise", "evidence_class": "Interpretation"}


NOTE = """# Chapter 11: strategic divergence scenarios

Generated by `tzbank.chapters.ch11_scenarios` (stage `valuation_results`). Blueprint §4 chapter 11; H11.1; VALUATION_SPEC
§7. **Register {version}, input hash `{hash}`.**

## Research question

Do NMB's and CRDB's different capital-allocation and funding paths produce different value outcomes under the same
macro scenario, how wide is the value distribution, and how did the 31 Dec 2025 call fare?

## Variable definitions

Scenarios: the register's bear, base and bull driver sets (bear: BoT easing -200 bp, credit cycle to the 2015-18 stress
cost of risk, stress-period growth, higher deposit beta, D-SIB surcharge where visible, subsidiary FX depreciation).
Monte Carlo: {draws:,} draws, Gaussian copula with triangular marginals (register bear/bull as bounds), NIM-rate and
CoR-growth correlations, plus one (lambda, ER*) pair per draw taken jointly from the chapter 5 moving-block bootstrap
replicates at that date (31 Dec 2025: ER through FY2024), each ER* draw shifted by (base ER* - median of the
draws) so the distribution centres on the base case (owner, 2026-11-09); ROE* = base COE + ER*; fixed seed. The
unshifted run is the labelled 'bull-centred' sensitivity.
Tested call: 31 Dec 2025 value and D4 rating against the actual total return to 31
Oct 2026 (post-split basis for NMB: 31 Dec 2025 figures / 10), decomposed with the five-step bridge.

## Sources and transformation trail

`config/assumption_register.csv`; `tzbank.valuation.uncertainty` (`scenarios`, `monte_carlo`, `bridge`); chapter 9
`rating`. Functions here: `scenario_tables`, `mc_summary`, `tested_call`, `h111`.

## Validation test

`tests/test_ch09_11.py`: Monte Carlo reproducible with the seed; scenario ordering; tested-call arithmetic.

## Results (Estimated)

{lead}

### Scenarios (value per share, EPS and ROE for the first three forecast years)

{sc}

### Monte Carlo

{mc}

### Exempt policy drivers: measured value direction (D56)

Growth paths, terminal g and the payout cap are exempt from the driver-level ordering check (their value effect flips
sign with ROE against COE and the capital constraint); their measured direction from the register tornado:

{exempt}

Scenario-level check (bear <= base <= bull in value, per bank and date): {scen_check}.

### Tested historical call (31 Dec 2025 -> 31 Oct 2026)

{call}

### Hypothesis (Interpretation)

{hyp}

## Evidence class

Scenario values: **Scenario**; Monte Carlo, tested call: **Estimated**; verdicts: **Interpretation**.

## Limitations

- Scenario bounds are the register's judgements; the Monte Carlo describes that judgement, not a probability of the
  world. It varies NIM, cost of risk, CIR, growth, COE, LP and the rate path (VALUATION_SPEC §7) and, since the
  ch09-11 review, (lambda, ER*) jointly from the chapter 5 moving-block bootstrap (standard method: D47 found it
  under-covers, so these draws are narrower than the Andrews interval for lambda).
- The tested call covers 10 months, one realisation; it is a check on the method, not a track record.
- {crdb}

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

The bear case loads the temporary credit-cycle explanation (chapter 8) and the rate-dependence of the deposit
franchise (chapter 6); a bank whose value falls more in the bear case has more of its rent in those cyclical sources.
"""


def write_ch11(ctx: Ctx) -> list[Path]:
    root = ctx.root
    sc, ev = scenario_tables(ctx)
    mc = mc_summary(ctx)
    br = pd.read_csv(path("tables", root) / "ch10_bridge.csv")
    call = tested_call(ctx, br)
    hyp = pd.DataFrame([h111(sc, mc)])
    tab, figd = path("tables", root), path("figures", root)
    tables = [(ctx.stamp(sc), "ch11_scenarios.csv", "Bear / base / bull: value, EPS and ROE", "Scenario"),
              (ctx.stamp(ev), "ch11_scenario_evidence.csv", "Scenario drivers and their chapter 6-8 evidence",
               "Scenario"),
              (ctx.stamp(mc), "ch11_monte_carlo.csv", "Monte Carlo percentiles and P(value > price)", "Estimated"),
              (ctx.stamp(call), "ch11_tested_call.csv", "Tested historical call with the bridge", "Estimated"),
              (ctx.stamp(hyp), "ch11_hypotheses.csv", "H11.1", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    fig, axes = plt_subplots(1, 2, figsize=(9.5, 4))
    hist = []
    for ax, b in zip(axes, BANKS, strict=True):
        v = ctx.monte_carlo("CURR", b)["value_per_share"].dropna()
        cnt, edges = np.histogram(v, bins=50)
        ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge", color=COLORS[b])
        ax.axvline(ctx.cases[("CURR", b, "base")].price, color="#1f1f1d", lw=1.2)
        style(ax, f"{T[b]}: Monte Carlo value per share, 31 Oct 2026", "draws")
        hist += [{"bank": T[b], "bin_left": float(lo), "count": int(n)} for lo, n in zip(edges[:-1], cnt, strict=True)]
    png = save(fig, figd / "ch11_monte_carlo.png", f"Evidence class: Estimated. Register {ctx.version}, input hash "
               f"{ctx.hash}. Vertical line: price at 31 Oct 2026.")
    fd = write_csv(ctx.stamp(pd.DataFrame(hist).assign(evidence_class="Estimated")),
                   figd / "ch11_monte_carlo_data.csv", require_evidence=True)
    files += [png, fd]
    sc_txt = []
    for (val, bank), g in sc.groupby(["valuation", "bank"], sort=False):
        g = g.set_index("scenario")
        eps = [c for c in g.columns if c.startswith("eps_")]
        roe = [c for c in g.columns if c.startswith("roe_")]
        sc_txt.append(f"- {bank} {val}: " + "; ".join(
            f"{s} {g.loc[s, 'value_per_share']:,.0f} (EPS " + ", ".join(f"{g.loc[s, e]:,.0f}" for e in eps if
                                                                      pd.notna(g.loc[s, e])) + "; ROE " + ", ".join(
                f"{g.loc[s, r]:.1%}" for r in roe if pd.notna(g.loc[s, r])) + ")" for s in ("bear", "base", "bull"))
            + f"; price {g['price'].iloc[0]:,.0f}.")
    tor = pd.read_csv(path("outputs", root) / "valuation" / "ordering_check.csv")
    ex = tor[tor["status"].astype(str).str.startswith("EXEMPT")]
    exempt = "\n".join(f"- {r.bank} {r.valuation} `{r.symbol}`: bear alone {r.bear_impact:+.1%}, bull alone "
                        f"{r.bull_impact:+.1%} of base value: {r.measured_direction}." for r in ex.itertuples())
    scen = tor[tor["check"].astype(str).str.startswith("scenario")]
    scen_check = f"{(scen['status'] == 'PASS').sum()} of {len(scen)} PASS"
    mc_txt = "\n".join(f"- {r.bank} {r.valuation}, {r.variant} (ER* shift {r.er_shift:+.2%}): P5 {r.p5:,.0f}, P10 "
                       f"{r.p10:,.0f}, median {r.p50:,.0f}, P90 "
                       f"{r.p90:,.0f}, P95 {r.p95:,.0f}; P(value > price {r.price:,.0f}) = {pct(r.prob_value_above_price, 0)} "
                       f"({r.valid:,} valid of {r.draws:,} draws)." for r in mc.itertuples())
    call_txt = "\n".join(
        f"- **{r.bank}** ({r.basis}): 31 Dec 2025 value {r.hist_value_per_share:,.0f} vs price {r.hist_price:,.0f}, "
        f"rating **{r.hist_rating}** (expected 12-month return {pct(r.hist_expected_return_12m)}, COE "
        f"{pct(r.hist_coe)}). Actual to 31 Oct 2026 ({r.months} months): price {r.actual_price_31oct2026_same_basis:,.0f}"
        f" + dividend {r.dividend_paid_per_share:,.0f} = total return **{pct(r.actual_total_return)}** vs COE over the "
        f"period {pct(r.coe_over_period)}: call borne out: **{r.call_borne_out}**. Bridge (per share, same basis): time "
        f"{r.bridge_time:+,.0f}, dividends {r.bridge_dividends:+,.0f}, new information {r.bridge_new_information:+,.0f}, "
        f"forecast revisions {r.bridge_forecast_revisions:+,.0f}, COE {r.bridge_coe:+,.0f}, residual "
        f"{r.bridge_residual:+,.0f} -> 31 Oct 2026 value {r.curr_value_same_basis:,.0f}." for r in call.itertuples())
    hr = hyp.iloc[0]
    hyp_txt = f"- **H11.1: {hr['verdict'].upper()}**. {hr['evidence']}. Rule: {hr['rule']}."
    interp = ("**Interpretation.** The tested call separates what the model got right (the "
              "fundamentals in bridge steps 3-4) from what it did not forecast (the price move beyond the value change, "
              "chapter 10's residual).")
    crdb = ("CRDB's 31 Dec 2025 run assumes a zero FY2025 dividend (capital rule); CRDB declared one after the cut-off "
            "(see chapter 9 for the sensitivity).")
    nf = write_text(NOTE.format(version=ctx.version, hash=ctx.hash, draws=int(settings(root)["valuation"]["mc_draws"]),
                                sc="\n".join(sc_txt), mc=mc_txt, lead=headline(mc), exempt=exempt,
                                scen_check=scen_check, call=call_txt, hyp=hyp_txt, crdb=crdb, interp=interp),
                    path("notes", root) / "ch11.md")
    files.append(nf)
    m = {"chapter": "11", "title": "Strategic divergence scenarios", "hypotheses": ["H11.1"],
         "status": f"built - register {ctx.version}, input hash {ctx.hash}",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c_, "note": rel(nf, root)}
                    for _, n, d, c_ in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "Monte Carlo value distribution",
                      "evidence_class": "Estimated", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "register_version": ctx.version, "input_hash": ctx.hash,
         "module": "tzbank.chapters.ch11_scenarios", "run_log_stage": "valuation_results"}
    files.append(write_json(m, path("manifests", root) / "ch11.json"))

    return files
