"""Prompt 09a: proposed assumption register and the engine dry run (no valuation results are published).

Writes ``config/assumption_register_proposed.csv`` (owner approves it as ``config/assumption_register.csv``),
``outputs/valuation/dryrun_engine_checks.csv`` (check name, status and the size of the worst deviation; no values,
prices or per-share numbers) and ``reports/valuation_09a_report.md``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.banks.engine import Drivers, EngineCheckError, Opening, project, require
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import coe as coe_mod
from tzbank.valuation import models as vm
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.inputs import pit_violations

SCEN = ("base", "bear", "bull")


def _vals(reg: pd.DataFrame, val: str, bank: str, sym: str, scen: str) -> list[float]:
    x = reg[(reg["valuation"] == val) & (reg["bank"] == rg.TICK[bank]) & (reg["symbol"] == sym)]
    return [float(v) for v in x.sort_values("period")[scen]]


class CoeGrowthGuardError(RuntimeError):
    """COE - terminal g below `valuation.coe_g_min_gap` in a scenario (D64 amended): the stage stops."""


def forward_inflation(vc: dict, b) -> tuple[float | None, float | None]:
    """Approach B inflation inputs: the config override if set, else the macro forecasts at the date (IMF WEO / REO,
    D64; MA-0007/08, MA-0029/30). NaN or missing -> None (B not computed)."""
    out = []
    for key, mk in (("inflation_tz_forward", "infl_tz_fwd"), ("inflation_us_forward", "infl_us_fwd")):
        v = vc.get(key)
        v = b.macro.get(mk) if v is None else v
        out.append(float(v) if v is not None and np.isfinite(v) else None)
    return out[0], out[1]


def case(reg: pd.DataFrame, inputs: dict, val: str, bank: str, scen: str, root: Path) -> un.Case:
    """Opening and drivers for one bank, date and scenario, read from the register (VALUATION_SPEC §8)."""
    st = settings(root)
    vc = st["valuation"]
    b = inputs[(val, bank)]
    o = inputs[(val, bank, "opening")]
    fy = b.fy

    def one(sym: str) -> float:
        return _vals(reg, val, bank, sym, scen)[0]

    def opt(sym: str) -> float | None:
        v = _vals(reg, val, bank, sym, scen)
        return v[0] if v and np.isfinite(v[0]) else None

    op = Opening(year=b.anchor_fy, loans=fy["net_loans"], deposits=fy["customer_deposits"],
                 liquid=fy["interest_earning_assets"] - fy["net_loans"], wholesale=fy["borrowings"],
                 equity_attr=o["e_ex"], nci_equity=fy["nci_equity"], total_assets=fy["total_assets"])
    sub = None
    if bank == "B01":
        sub = {"s_sub": one("s_sub"), "u": one("u"), "payout_sub": 1.0, "fx": [one("fx")] * o["T"]}
    roe, g = one("ROE_star"), one("g")
    dr = Drivers(g_L=_vals(reg, val, bank, "g_L", scen), g_D=_vals(reg, val, bank, "g_D", scen),
                 NIM=_vals(reg, val, bank, "NIM", scen), phi=_vals(reg, val, bank, "phi", scen),
                 CIR=_vals(reg, val, bank, "CIR", scen), CoR=_vals(reg, val, bank, "CoR", scen),
                 rho=_vals(reg, val, bank, "rho", scen), rate=_vals(reg, val, bank, "rate", scen),
                 rate0=_vals(reg, val, bank, "rate", "base")[0], omega=one("omega"), theta=one("theta"),
                 l_min=one("l_min"), s_W=one("s_W"), tax=one("t"), nu=one("nu"), delta=one("delta"),
                 k_star=one("k_star"), pi_max=one("pi_max"), roe_star=roe, g=g, payout_star=one("payout_star"),
                 lam=one("lambda"), fade_years=int(vc["fade_years"]), fade_shape=vc["fade_shape"],
                 asset_beta=one("asset_beta"), deposit_beta=one("deposit_beta"),
                 nowcast={"np_attr_ytd": b.interim["profit_attributable"], "months": b.interim_months}
                 if b.interim_months else None, sub=sub, ldr_bounds=tuple(vc["ldr_bounds"]),
                 wholesale_max=float(vc["wholesale_share_max"]), k_req=one("k_req"), icg_cap=scen == "base",
                 payout_floor=float(vc.get("payout_floor", 0.20)), g_fade_start=opt("g_fade"))
    co = coe_mod.reconcile(one("COE_A"), coe_mod.approach_b(
        b.macro["rf_us"], one("beta"), b.macro["erp"], b.macro["crp"], one("LP"), *forward_inflation(vc, b)),
        vc["coe_reconcile_bp"])
    r = co.a if vc.get("coe_central", "A") == "A" else co.central
    if vc.get("coe_central", "A") != "A":
        # D93: ROE* = base central COE + ER* (VALUATION_SPEC §4). The approved register stores ROE* built on the
        # approach-A base COE; with the registered rule the central (base) COE may differ, and ER* (the estimated
        # quantity) is held, so ROE* and the derived terminal payout 1 - g / ROE* move with the central COE.
        base_b = coe_mod.approach_b(b.macro["rf_us"], _vals(reg, val, bank, "beta", "base")[0], b.macro["erp"],
                                    b.macro["crp"], _vals(reg, val, bank, "LP", "base")[0], *forward_inflation(vc, b))
        r_base = coe_mod.reconcile(_vals(reg, val, bank, "COE_A", "base")[0], base_b, vc["coe_reconcile_bp"]).central
        roe_c = r_base + one("ER_star")
        dr = replace(dr, roe_star=roe_c, payout_star=1 - dr.g / roe_c)
    gap = float(vc.get("coe_g_min_gap", 0.04))
    if r - dr.g < gap - 1e-12:  # D64 amended guard: stop on violation (every scenario, both dates)
        raise CoeGrowthGuardError(f"COE - g guard: {val} {bank} {scen}: COE {r:.4f} - g {dr.g:.4f} = {r - dr.g:.4f} "
                                  f"< {gap:.2f} (valuation.coe_g_min_gap)")
    return un.Case(bank, val, b.vd, op, dr, r, b.shares, b.price, vc["weights"],
                   lag_months=int(vc.get("dividend_pay_lag_months", 0)))


BASE_CUT_STOP = 0.02  # 09a review C: a feasibility cut above 2 pp in the base case is a stop for the owner


def make_feasible(reg: pd.DataFrame, inputs: dict, root: Path, step: float = 0.01, max_cut: float = 0.20
                  ) -> tuple[pd.DataFrame, list[dict]]:
    """Capital-feasibility GUARD (VALUATION_SPEC §4: Tier 1 >= k* x RWA without new capital; 09a review C). Scenarios
    are set from evidence first (bear from the 2015-18 stress); only where one still breaches are loan and deposit
    growth from year 2 cut in 1 pp steps (floored at the evidence range). Every cut is returned for the check log and
    written into the rationale; a base-case cut above 2 pp is a STOP for the owner."""
    reg = reg.copy()
    notes: list[dict] = []
    for val in rg.DATES:
        for bank in ("B02", "B01"):
            for s in ("bear", "bull"):  # D56: the base case uses the ICG growth cap instead of cuts
                cut = 0.0
                while cut <= max_cut + 1e-12:
                    c = case(reg, inputs, val, bank, s, root)
                    ok = next(k for k in project(c.op, c.dr).checks if k["check"].startswith("Tier 1"))["status"]
                    if ok == "PASS":
                        break
                    cut += step
                    for sym in ("g_L", "g_D"):
                        m = (reg["valuation"] == val) & (reg["bank"] == rg.TICK[bank]) & (reg["symbol"] == sym)
                        idx = reg[m].sort_values("period").index[1:]
                        reg.loc[idx, s] = np.maximum(reg.loc[idx, s].astype(float) - step, reg.loc[idx, "range_low"])
                if cut > 0:
                    m = (reg["valuation"] == val) & (reg["bank"] == rg.TICK[bank]) & reg["symbol"].isin(["g_L", "g_D"])
                    reg.loc[m, "rationale"] += (f" Capital feasibility ({s}): growth from year 2 cut by {cut * 100:.0f} pp "
                                                "so Tier 1 >= k* x RWA without new capital.")
                notes.append({"valuation": val, "bank": rg.TICK[bank], "scenario": s, "cut": round(cut, 4),
                              "stop": s == "base" and cut > BASE_CUT_STOP + 1e-12})
    return reg, notes


def _check(rows: list, val: str, bank: str, scen: str, name: str, ok: bool, worst: float, detail: str = "") -> None:
    rows.append({"valuation": val, "bank": rg.TICK[bank], "scenario": scen, "check": name,
                 "status": "PASS" if ok else "FAIL", "worst_deviation": worst, "detail": detail,
                 "evidence_class": "Derived"})


def dry_run(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = root or find_root()
    st = settings(root)
    vc = st["valuation"]
    seed = int(st["random_seed"])
    reg, inputs = rg.build(root)
    reg, notes = make_feasible(reg, inputs, root)
    rows: list[dict] = []
    info: dict = {"feasibility": notes, "register": reg}
    for n_ in notes:
        b_ = "B01" if n_["bank"] == "CRDB" else "B02"
        _check(rows, n_["valuation"], b_, n_["scenario"], "capital-feasibility guard: growth cut from year 2 "
               "(base cut > 2 pp = STOP for the owner)", not n_["stop"], n_["cut"],
               f"cut {n_['cut'] * 100:.0f} pp" + (" - STOP: owner decision needed" if n_["stop"] else ""))
    for val in rg.DATES:
        for bank in ("B02", "B01"):
            b = inputs[(val, bank)]
            bad = pit_violations(b)
            _check(rows, val, bank, "all", "information cut-off: every input published on or before the valuation "
                   "date", not bad, float(len(bad)), "; ".join(bad) or f"{len(b.sources)} sourced inputs checked")
            bf = inputs[(val, bank, "buffer")]
            _check(rows, val, bank, "base", "bank buffer: anchor-year headroom within 1 pp of the FY2015-anchor mean "
                   "(else STOP for the owner)", not bf["stop"], bf["anchor"] - bf["mean"],
                   f"mean {bf['mean']:.4f}, anchor {bf['anchor']:.4f}, base buffer {bf['base']:.4f}")
            x = reg[(reg["valuation"] == val) & (reg["bank"] == rg.TICK[bank])]
            fin = x[["base", "bear", "bull"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
            _check(rows, val, bank, "all", "register: every driver has base, bear and bull", bool(fin.all()),
                   float((~fin).sum()))
            _check(rows, val, bank, "all", "register: values inside the evidence range", bool(
                (x["inside_range"] != "NO").all()), float((x["inside_range"] == "NO").sum()))
            cases = {s: case(reg, inputs, val, bank, s, root) for s in SCEN}
            info[(val, bank)] = cases
            for s, c in cases.items():
                fc = project(c.op, c.dr)
                for ck in fc.checks:
                    _check(rows, val, bank, s, f"engine: {ck['check']}", ck["status"] == "PASS", ck["worst"],
                           ck["detail"])
                try:
                    require(fc.checks)
                    stopped = ""
                except EngineCheckError as exc:
                    stopped = str(exc)
                _check(rows, val, bank, s, "run completes without a named engine error", not stopped, 0.0,
                       stopped[:300])
                bl = vm.blend(fc, c.r, c.vd, c.w, lag_months=c.lag_months)
                _check(rows, val, bank, s, "RIM and DDM within 10% (else flagged)", bl["gap"] <= vc["rim_ddm_gap_flag"],
                       bl["gap"])
                ti = vm.target_identity(fc, c.r, c.vd, lag_months=c.lag_months)
                _check(rows, val, bank, s, "target-price identity V1 = V0 (1 + r) - Div (carried)",
                       abs(ti["residual"]) < vc["identity_tolerance"], ti["residual"])
                _check(rows, val, bank, s, "r > g and ROE* > g", c.r > c.dr.g and c.dr.roe_star > c.dr.g,
                       c.r - c.dr.g)
            c = cases["base"]
            coe_a = _vals(reg, val, bank, "COE_A", "base")[0]  # approach A itself; c.r is the central COE (D93)
            co = coe_mod.reconcile(coe_a, coe_mod.approach_b(b.macro["rf_us"], b.macro["beta"], b.macro["erp"],
                                                           b.macro["crp"], 0.015, *forward_inflation(vc, b)))
            _check(rows, val, bank, "base", "COE: approach A computed; B and reconciliation", math.isfinite(co.a),
                   0.0 if not math.isfinite(co.b) else co.a - co.b, co.rule)
            c5 = pd.read_csv(path("tables", root) / "ch05_coe_annual.csv")
            lbl = "valuation_historical" if val == "HIST" else "valuation_current"
            ref = float(c5.loc[c5["date_label"] == lbl, "coe"].iloc[0])
            _check(rows, val, bank, "base", "COE A equals chapter 5's valuation-date COE (same inputs, LP 1.5%)",
                   abs(coe_a - ref) < 1e-9, coe_a - ref)
            g = un.grids(c, vc["grid_step_bp"], vc["grid_half_width_bp"])
            for k, df in g.items():
                _check(rows, val, bank, "base", f"sensitivity grid {k} computed (9 x 9)", len(df) == 81,
                       float(df["value_per_share"].isna().sum()), "cells with r <= g are left empty")
            coe_sh = (cases["bull"].r - cases["base"].r - (0.0 - 0.015), cases["bear"].r - cases["base"].r - 0.015)
            bounds = un.mc_bounds(cases["base"].dr, cases["bear"].dr, cases["bull"].dr,
                                  (min(coe_sh), max(coe_sh)), (-0.015, 0.015))
            mc1 = un.monte_carlo(c, bounds, vc["mc_correlation"], int(vc["mc_draws_dry_run"]), [seed, 9, 1])
            mc2 = un.monte_carlo(c, bounds, vc["mc_correlation"], int(vc["mc_draws_dry_run"]), [seed, 9, 1])
            _check(rows, val, bank, "base", f"Monte Carlo reproducible with the seed ({vc['mc_draws_dry_run']} dry-run "
                   f"draws; {vc['mc_draws']} in the published run)", mc1.equals(mc2),
                   float(mc1["value_per_share"].isna().mean()))
            sc = un.scenarios(cases)
            _check(rows, val, bank, "all", "scenarios bear / base / bull computed", bool(sc["value_per_share"].notna().all()),
                   0.0)
            pe = c.price * c.shares
            ic = vm.implied_coe(c.op, c.dr, pe, c.vd, lag_months=c.lag_months)
            ir = vm.implied_roe_star(c.op, c.dr, c.r, pe, c.vd, lag_months=c.lag_months)
            _check(rows, val, bank, "base", "implied COE solved by bracketed root", math.isfinite(ic), 0.0,
                   "root found" if math.isfinite(ic) else "no root in (g, 60%)")
            _check(rows, val, bank, "base", "implied ROE* solved by bracketed root", math.isfinite(ir), 0.0,
                   "root found" if math.isfinite(ir) else "no root in (g, 80%)")
            iso = vm.iso_surface(c.op, c.dr, c.r, pe, c.vd, np.linspace(0.0, 0.9, 10), np.linspace(-0.05, 0.15, 21),
                                 lag_months=c.lag_months)
            _check(rows, val, bank, "base", "iso-value surface (lambda x ER*) computed", len(iso) == 210,
                   float(iso.groupby("lambda")["er_star_at_price"].first().isna().sum()),
                   "lambdas with no ER* matching the price inside -5%..+15%")
            j = vm.justified(c.dr.roe_star, c.dr.g, c.r, c.dr.payout_star)
            _check(rows, val, bank, "base", "justified P/B and P/E computed", all(math.isfinite(v) for v in j.values()),
                   0.0)
    # bridge mechanics, per bank (historical -> current)
    for bank in ("B02", "B01"):
        h, cu = info[("HIST", bank)]["base"], info[("CURR", bank)]["base"]
        hd = h.dr
        n = cu.dr.T

        def ext(xs: list[float], n: int = n) -> list[float]:
            return (list(xs[1:]) + [xs[-1]] * n)[:n]
        hdc = replace(cu.dr, **{k: ext(getattr(hd, k)) for k in ("g_L", "g_D", "NIM", "phi", "CIR", "CoR", "rho",
                                                                 "rate")}, roe_star=hd.roe_star, g=hd.g,
                      payout_star=hd.payout_star, lam=hd.lam)
        bcur = inputs[("CURR", bank)]
        div = bcur.final_dps_anchor * bcur.fy["shares_outstanding"]
        br = un.bridge(h, cu, hdc, div)
        res = float(br["tzs"].iloc[-1]) / float(br["tzs"].iloc[-2])
        _check(rows, "HIST->CURR", bank, "base", "five-step bridge computed in the fixed order; residual reported",
               abs(res) < 1e-9, res, "steps: time, dividends, new information, forecast revisions, COE")
    out = pd.DataFrame(rows)
    return out, reg, info


REPORT = """# Prompt 09a: valuation engine and proposed assumption register (STOP: owner approval)

Generated by `tzbank.valuation.run` (stage `valuation`). VALUATION_SPEC §2-8 and §10; blueprint D8, D12; decisions
D50 (confirmed), D52-D55. **No valuation results are published**: the engine ran only on the golden test
(`tests/test_valuation_golden.py`) and one dry run per bank and date; this report and the check log contain check
statuses and deviations, never values, prices or per-share numbers.

## What was built

- `tzbank.valuation.inputs`: point-in-time loader. Every input passes `pit.eligible` for its valuation date; latest
  restatement published by the date wins (D5); capital on the D2/D17 basis. Anchors: **31 Dec 2025 -> FY2024 audited +
  9M 2025 interim**; **31 Oct 2026 -> FY2025 audited + 9M 2026 interim**.
- `tzbank.banks.engine`: driver engine (VALUATION_SPEC §4): loans and deposits separately; liquid assets or wholesale
  funding balance the sheet (solved each year by fixed-point iteration); the capital rule with delta (NMB on Bank-solo
  Tier 1 and RWA, D2); the CRDB subsidiary layer (subsidiary profit share with an FX path, NCI, distributable capacity
  = parent profit + upstream ratio x subsidiary profit); NIM tied to the rate path through asset and deposit betas
  (open issue 11); first year = interim actual + remaining months; fade and terminal.
- `tzbank.valuation.coe`: approaches A and B and the 150 bp reconciliation rule.
- `tzbank.valuation.models`: RIM, capital-constrained DDM, value at any date, target price V1 with the identity,
  justified P/B and P/E, implied COE and implied ROE* (bracketed roots), the (lambda, ER*) iso-value surface.
- `tzbank.valuation.uncertainty`: COE x g, COE x ROE*, NIM x CoR grids (+/- 200 bp, 50 bp steps); bear / base / bull
  scenarios; Monte Carlo over NIM, CoR, CIR, asset growth, COE, LP and the rate path (Gaussian copula, triangular
  marginals from the register, NIM-rate and CoR-growth correlations, fixed seed); the five-step bridge.
- `tzbank.valuation.register`: the proposed register `config/assumption_register_proposed.csv`.

## Fade (owner pre-flight D; open issue 3)

The fade uses the **break-adjusted lambda** (D50): ROE_t - ROE* = lambda (ROE_t-1 - ROE*) over the 10 fade years
(`valuation.fade_shape = lambda`; VALUATION_SPEC §4's linear fade is the alternative). The reason is necessity, not
preference: the through-the-cycle AR(1) lambda is 1.00 for both banks, so mean reversion is not established on the full
sample and that lambda cannot serve as a fade speed. Any gap left after 10 years (lambda^10 x the remaining excess) is
closed at the terminal year.

## Rate path and the deposit franchise (owner pre-flight C; open issue 11)

NIM_t = NIM path_t + (asset beta - deposit beta x avg deposits / avg IEA) x (T-bill_t - T-bill_0). The rate path per
scenario: **base** flat at the latest annual T-bill; **bear** BoT easing, -200 bp over two years; **bull** +100 bp.
Asset betas (IEA yield on the T-bill) and deposit betas (cost of deposits on the T-bill) are estimated on annual changes
with the chapter 6 method; for 31 Dec 2025 on FY<=2024 data only. So the deposit franchise contribution moves with rates
in every scenario.

## Information sets (D1)

{pit}

## Proposed register summary

{reg}

## Dry-run engine checks

{checks}

## Points for the owner

{points}

## Limitations

- Approach B is not computed: forward Tanzanian and US inflation are TO COLLECT (macro MA-0007/08, MA-0029/30); the
  central COE is approach A, flagged (`valuation.inflation_*_forward`, null).
- Terminal g is a placeholder (7% base, 5% bear, 9% bull) until the nominal GDP forecast is collected (MA-0010/32).
- Interim dividends and payment dates are TO COLLECT; dividends are treated at fiscal year-end (cum on the year-end date).
- Subsidiary FX paths (BIF, CDF) are TO COLLECT; the bear case assumes 10% a year depreciation against TZS.
- The CURR T-bill path starts from the 2025 annual mean (the latest year in `ch06_rates_used.csv`).
- The licensing question for publishing ratings and target prices (VALUATION_SPEC §11 open decision; D9) is still open.
"""


def write_valuation(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    checks, reg, info = dry_run(root)
    feas = info.pop("feasibility")
    vdir = path("outputs", root) / "valuation"
    files = [write_csv(reg.assign(evidence_class=np.where(reg["distribution"] == "derived", "Derived",
                                                          "Scenario")), root / "config" /
                       "assumption_register_proposed.csv", require_evidence=True),
             write_csv(checks, vdir / "dryrun_engine_checks.csv", require_evidence=True)]
    pit = "\n".join(
        f"- {rg.TICK[b]} {v} ({rg.DATES[v]}): anchor FY{info[(v, b)]['base'].op.year}; information cut-off check "
        + checks[(checks['valuation'] == v) & (checks['bank'] == rg.TICK[b])
                 & checks['check'].str.startswith('information')]['status'].iloc[0]
        for v in rg.DATES for b in ("B02", "B01"))
    pit += ("\n- Shares outstanding at 31 Dec 2025 carry an April 2026 publication date in the workbook, so the 31 Dec "
            "2025 run takes shares from the FY2024 annual report.\n- `tests/test_valuation_pit.py`: a FY2025 audited row "
            "cannot enter the 31 Dec 2025 run.")
    s = reg.groupby(["valuation", "bank"]).agg(rows=("id", "size"), inside=("inside_range", lambda x: (x == "yes").sum()),
                                               judgement=("inside_range", lambda x: (x != "yes").sum()))
    reg_txt = ("`config/assumption_register_proposed.csv`: " + f"{len(reg)} rows; " + "; ".join(
        f"{v} {b}: {r.rows} rows, {r.inside} inside their evidence range, {r.judgement} fixed / judgement"
        for (v, b), r in s.iterrows()) + ". Ranges: HIST from data eligible at 31 Dec 2025 (lambda, ER* and betas "
        "re-estimated on FY<=2024); CURR from `calibration_inputs.csv` where it covers the driver. Every row carries "
        "its rationale and evidence; status PROPOSED.")
    agg = checks.groupby("status").size().to_dict()
    fails = checks[checks["status"] == "FAIL"]
    chk = (f"{len(checks)} checks: {agg.get('PASS', 0)} PASS, {agg.get('FAIL', 0)} FAIL "
           "(`outputs/valuation/dryrun_engine_checks.csv`).\n\n"
           + ("Failed checks:\n\n" + "\n".join(f"- {r.valuation} {r.bank} {r.scenario}: {r.check} (worst "
                                               f"{r.worst_deviation:.4g}) {r.detail}" for r in fails.itertuples())
              if len(fails) else "All checks pass."))
    points = "\n".join([
        "- Approve or edit the register; it is then saved as `config/assumption_register.csv` with a version number.",
        f"- Capital-feasibility guard: {sum(1 for n in feas if n['cut'] > 0)} cut(s) "
        + ("(" + "; ".join(f"{n['valuation']} {n['bank']} {n['scenario']} -{n['cut'] * 100:.0f} pp" for n in feas
                           if n["cut"] > 0) + ")" if any(n["cut"] > 0 for n in feas) else "")
        + "; scenarios are set from evidence (bear growth from the 2015-18 stress) and the guard only logs residual "
          "breaches; a base cut above 2 pp would stop the run for the owner.",
        "- Terminal g (placeholder 5/7/9%) and the subsidiary FX bear path (10% a year) are judgement values awaiting "
        "the macro inputs.",
        "- Bank buffer in k* (D56, owner): base = min(mean FY2015-anchor headroom, anchor-year headroom), STOP if the "
        "anchor headroom is > 1 pp below the mean; bull = half the anchor headroom, at most 2 pp; bear = base + the "
        "largest D-SIB scenario where the designation is visible at the date (NMB at 31 Oct 2026). Year 1 (nowcast) "
        "is checked against the requirement only; k* is phased in from the year-1 ratio when the gap is <= 1 pp; "
        "base growth from year 2 is capped by internal capital generation (payout floor 20%).",
        "- Where the anchor already sits at the edge of the evidence range (e.g. NMB NIM and cost-to-income at their "
        "FY2019-25 extremes), the bear or bull value equals the anchor: the range binds; rate-driven NIM moves come on "
        "top through the betas."])
    files.append(write_text(REPORT.format(pit=pit, reg=reg_txt, checks=chk, points=points),
                            path("reports", root) / "valuation_09a_report.md"))
    m = {"chapter": "09a", "title": "Valuation engine and proposed assumption register", "status":
         "dry run only - STOP for owner approval; no valuation results published",
         "tables": [{"path": rel(files[1], root), "description": "Dry-run engine-check log", "evidence_class": "Derived"}],
         "registers": [rel(files[0], root)], "reports": [rel(files[2], root)],
         "module": "tzbank.valuation.run", "run_log_stage": "valuation"}
    files.append(write_json(m, vdir / "manifest_09a.json"))
    return files
