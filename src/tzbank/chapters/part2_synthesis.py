"""Part II synthesis and calibration inputs (prompt 08; blueprint §0 cross-cutting question, chapters 4-8; D8).

Reads only the chapter 4-8 output tables. Produces:

- ``reports/part2_consistency.md`` / ``.csv``: the same quantity computed in two chapters must agree (funding 4.0/4.2
  vs 6, cost of risk and return on RWA 4.6 vs 8, leverage and ROE 4.0 vs 7 vs 5); differences are resolved in code or
  explained by a stated definition difference.
- ``outputs/tables/part2_rent_sources.csv``: per bank and blueprint §0 explanation, the evidence (chapter, table, row),
  direction, strength and evidence class, plus an overall answer per bank (genuine advantage, structural features or
  cycle).
- ``outputs/tables/hypotheses_status.csv``: every H4-H8 result.
- ``outputs/tables/calibration_inputs.csv``: evidence-based RANGES for 09a, never chosen values (D8).
- ``reports/part2_executive_summary.md``: one-page draft for owner review; ``outputs/notes/part2.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import FOCUS, pct, tickers
from tzbank.config import find_root, path
from tzbank.io import rel, write_csv, write_json, write_text

EXPLANATIONS = ["scale economies", "low-cost deposits", "customer switching costs", "limited competition",
                "regulatory advantages", "distribution and market access", "temporary credit-cycle conditions"]
D8 = "range only: the value is chosen in 09a, not here (D8)"
TOL = 1e-9


def _t(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, dtype={"bank_id": str}, low_memory=False)


# --- 1. consistency ----------------------------------------------------------------------------------------------


def _cmp(name: str, area: str, a: pd.DataFrame, b: pd.DataFrame, ca: str, cb: str, keys: list[str], tol: float,
         la: str, lb: str, explain: str = "", resolved: str = "") -> dict:
    m = a[keys + [ca]].merge(b[keys + [cb]], on=keys, suffixes=("_a", "_b"))
    xa, xb = (ca + "_a", cb + "_b") if ca == cb else (ca, cb)
    m = m.dropna(subset=[xa, xb])
    d = (m[xa] - m[xb]).abs()
    n_diff = int((d > tol).sum())
    worst = m.loc[d.idxmax()] if len(m) else None
    status = "consistent" if n_diff == 0 else ("explained (definition differs)" if explain else
                                               "CONTRADICTION (unresolved)")
    if resolved and n_diff == 0:
        status = "resolved in code"
    return {"area": area, "check": name, "left": la, "right": lb, "n_compared": len(m), "n_differ": n_diff,
            "max_abs_diff": float(d.max()) if len(m) else np.nan,
            "worst_case": ", ".join(f"{k}={worst[k]}" for k in keys) if worst is not None and n_diff else "",
            "status": status, "explanation": resolved or explain, "evidence_class": "Derived"}


def consistency(root: Path) -> pd.DataFrame:
    k = ["bank_id", "fiscal_year"]
    c40 = _t(root, "ch04_4_0_components.csv")
    f42 = _t(root, "ch04_4_2_funding.csv")
    d6 = _t(root, "ch06_deposits.csv")
    fr6 = _t(root, "ch06_franchise.csv")
    r46 = _t(root, "ch04_4_6_risk.csv")
    h8 = _t(root, "ch08_credit_history.csv")
    e8 = _t(root, "ch08_capital_efficiency.csv")
    c7 = _t(root, "ch07_capital.csv")
    er5 = _t(root, "ch05_excess_returns.csv")
    rows = [
        _cmp("cost of deposits", "funding (4.2 vs 6)", f42, d6, "cost_of_deposits", "cost_of_deposits", k, 1e-9,
             "ch04_4_2_funding.cost_of_deposits", "ch06_deposits.cost_of_deposits"),
        _cmp("gross deposit funding advantage (TZS)", "funding (4.2 vs 6)", f42, fr6, "gdfa_proxy_tzs", "gdfa", k,
             1.0, "ch04_4_2_funding.gdfa_proxy_tzs", "ch06_franchise.gdfa",
             explain="4.2 uses the T-bill yield in its own annual-average series and all average deposits; chapter 6 "
                     "uses ch06_rates_used (mean of quarter-end auctions); differences listed are the rate series"),
    ]
    # 4.0 funding term against chapter 6's reconciliation of it
    c = c40[c40["tree"] == "full"].copy()
    c["funding_term"] = -(c["ibl_ta"] * c["c_fund"]) * (1 - c["tax_rate"]) * c["ta_e"] * c["attr_share"]
    rows.append(_cmp("4.0 funding term in ROE points", "funding (4.0 vs 6)", c, fr6, "funding_term",
                     "funding_term_40_roe", k, 1e-9, "ch04_4_0_components (-ibl_ta x c_fund x (1-t) x TA/E x attr)",
                     "ch06_franchise.funding_term_40_roe"))
    res = fr6["reconciliation_residual"].abs().max()
    rows.append({"area": "funding (4.0 vs 6)", "check": "chapter 6 decomposition of the 4.0 funding term adds up",
                 "left": "ch06_franchise components", "right": "ch06_franchise.funding_term_40_roe",
                 "n_compared": int(fr6["reconciliation_residual"].notna().sum()),
                 "n_differ": int((fr6["reconciliation_residual"].abs() > 1e-9).sum()), "max_abs_diff": float(res),
                 "worst_case": "", "status": "consistent" if res < 1e-9 else "CONTRADICTION (unresolved)",
                 "explanation": "", "evidence_class": "Derived"})
    rows.append(_cmp("cost of risk", "credit (4.6 vs 8)", r46, h8, "cost_of_risk", "cost_of_risk", k, 1e-12,
                     "ch04_4_6_risk.cost_of_risk", "ch08_credit_history.cost_of_risk"))
    rows.append(_cmp("NPL ratio", "credit (4.6 vs 8)", r46, h8, "npl_ratio", "npl_ratio", k, 1e-12,
                     "ch04_4_6_risk.npl_ratio", "ch08_credit_history.npl_ratio"))
    rows.append(_cmp("return on RWA", "credit (4.6 vs 8)", r46[r46["bank_id"].isin(FOCUS)],
                     e8[e8["bank_id"].isin(FOCUS)], "return_on_rwa", "return_on_rwa", k, 1e-12,
                     "ch04_4_6_risk.return_on_rwa", "ch08_capital_efficiency.return_on_rwa",
                     resolved="4.6 previously divided by the bank_year_wide RWA (e.g. CRDB FY2018 TZS 4,084bn); it now "
                              "uses the D2/D17 capital-basis RWA of chapters 7-8 (CRDB FY2018 TZS 4,305bn) "
                              "(`ch04.risk.with_capital_basis_rwa`, prompt 08)"))
    imp = h8.merge(c40[k + ["imp_ta"]], on=k)
    rows.append(_cmp("impairment term 4.0 vs chapter 8 cost of risk x loans / assets", "credit (4.0 vs 8)",
                     c40, r46, "imp_ta", "impairment_to_avg_assets", k, 1e-12, "ch04_4_0_components.imp_ta",
                     "ch04_4_6_risk.impairment_to_avg_assets (= ch08 impairment / avg TA)"))
    del imp
    # leverage: 4.0 TA/E (average balances, attributable equity) vs chapter 7 balances
    c7s = c7.sort_values(k).copy()
    g = c7s.groupby("bank_id")
    prev = g["fiscal_year"].diff() == 1
    c7s["ta_e_from_ch07"] = ((c7s["total_assets"] + g["total_assets"].shift(1)) / 2 /
                             ((c7s["equity"] + g["equity"].shift(1)) / 2)).where(prev)
    rows.append(_cmp("leverage TA/E (average balances)", "leverage (4.0 vs 7)", c40, c7s, "ta_e", "ta_e_from_ch07",
                     k, 1e-9, "ch04_4_0_components.ta_e", "ch07_capital (avg total_assets / avg equity)"))
    c7s["e_ta_year_end"] = c7s["equity_to_assets"]
    c40["e_ta_avg_attr"] = 1 / c40["ta_e"]
    rows.append(_cmp("equity / assets", "leverage (4.0 vs 7)", c40, c7s, "e_ta_avg_attr", "e_ta_year_end", k, 1e-9,
                     "1 / ch04_4_0_components.ta_e (average, attributable equity)",
                     "ch07_capital.equity_to_assets (year-end, total equity incl. NCI)",
                     explain="4.0 uses average attributable equity; chapter 7's leverage decomposition uses year-end "
                             "total equity (incl. NCI) because it is matched to year-end Tier 1 and RWA. Both are "
                             "correct for their use; the decomposition is not an input to the 4.0 tree"))
    rows.append(_cmp("ROE (attributable, average equity)", "leverage (4.0 vs 7)", c40, c7, "roe_reported", "roe", k,
                     1e-9, "ch04_4_0_components.roe_reported", "ch07_capital.roe"))
    rows.append(_cmp("ROE (attributable, average equity)", "ROE (4.0 vs 5)", c40[c40["bank_id"].isin(FOCUS)],
                     er5[er5["bank_id"].isin(FOCUS)], "roe_reported", "roe", k, 1e-9,
                     "ch04_4_0_components.roe_reported", "ch05_excess_returns.roe",
                     explain="chapter 5 uses the latest restatement published by the valuation date (D5) and the "
                             "opening equity convention of the excess-return model where it differs"))
    out = pd.DataFrame(rows)
    # an 'explained' row that in fact agrees is plainly consistent
    same = (out["n_differ"] == 0) & out["status"].str.startswith("explained")
    out.loc[same, "status"] = "consistent"
    out.loc[same, "explanation"] = ""  # identical: no difference left to explain
    return out


def consistency_md(cons: pd.DataFrame, root: Path) -> str:
    f42 = _t(root, "ch04_4_2_funding.csv")
    fr6 = _t(root, "ch06_franchise.csv")
    rates = _t(root, "ch06_rates_used.csv")
    lines = ["# Part II cross-chapter consistency", "",
             "Generated by `tzbank.chapters.part2_synthesis.consistency` (stage `synthesis`, prompt 08). Each check "
             "compares the same quantity as computed in two chapters, bank-year by bank-year. Status: consistent "
             "(identical within tolerance), resolved in code (a difference was found and removed in this prompt), "
             "explained (a stated definition difference), or contradiction (unresolved: none allowed).", "",
             "| Area | Check | Left | Right | Compared | Differ | Max abs diff | Status |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in cons.itertuples():
        lines.append(f"| {r.area} | {r.check} | `{r.left}` | `{r.right}` | {r.n_compared} | {r.n_differ} | "
                     f"{r.max_abs_diff:.3g} | {r.status} |")
    lines += ["", "## Resolutions and explanations", ""]
    for r in cons[cons["explanation"].astype(str) != ""].itertuples():
        lines.append(f"- **{r.check}** ({r.status}; worst case {r.worst_case or 'n/a'}): {r.explanation}.")
    g = f42.merge(fr6[["bank_id", "fiscal_year", "gdfa", "tbill"]], on=["bank_id", "fiscal_year"]).dropna(
        subset=["gdfa_proxy_tzs", "gdfa"])
    g = g[(g["gdfa_proxy_tzs"] - g["gdfa"]).abs() > 1]
    if len(g):
        lines += ["", "GDFA differences by bank-year (4.2 T-bill vs chapter 6 T-bill):", ""]
        for r in g.itertuples():
            lines.append(f"- {r.bank_id} FY{r.fiscal_year}: 4.2 T-bill {pct(r.tbill_yield_annual_avg, 2)} vs ch. 6 "
                         f"{pct(r.tbill, 2)}; GDFA TZS {r.gdfa_proxy_tzs / 1e9:,.1f}bn vs {r.gdfa / 1e9:,.1f}bn.")
    lines += ["", f"T-bill series used by chapter 6: `ch06_rates_used.csv` ({len(rates)} years).", "",
              "Basis note (open issue 5): NMB's return on RWA and RWA density combine Group profit / assets with "
              "Bank-solo RWA (D2); CRDB's FY2015 average RWA mixes Bank-solo (FY2014) and Group (FY2015) and is "
              "flagged in chapter 8. These are basis effects, not contradictions.", ""]
    bad = cons[cons["status"].str.startswith("CONTRADICTION")]
    lines.append("**Unresolved contradictions: none.**" if bad.empty else
                 "**Unresolved contradictions:** " + "; ".join(bad["check"]))
    return "\n".join(lines) + "\n"


# --- 2. rent sources -------------------------------------------------------------------------------------------------


def rent_sources(root: Path, t: dict[str, str]) -> pd.DataFrame:
    fr6 = _t(root, "ch06_franchise.csv")
    dep = _t(root, "ch06_deposits.csv")
    beta = _t(root, "ch06_deposit_beta.csv")
    sc = _t(root, "ch04_4_5_scale_elasticity.csv").set_index("bank_id")
    gap = _t(root, "ch04_4_0_gap_attribution.csv")
    cal = _t(root, "ch05_calibration_candidates.csv")
    h5 = _t(root, "ch05_hypotheses.csv").set_index("hypothesis")
    h81 = _t(root, "ch08_h81_cost_of_risk_gap.csv")
    sl = _t(root, "ch08_stress_lens.csv")
    tr = _t(root, "ch04_4_4_transactions.csv")
    comp = _t(root, "ch04_4_1_composition.csv")
    rk = _t(root, "ch05_rank_persistence.csv")
    ds = _t(root, "ch07_dsib_search.csv")
    rows: list[dict] = []

    def ev(b, x, evidence, ch, table, row, direction, strength, ec):
        rows.append({"bank_id": b, "bank": t[b], "explanation": x, "evidence": evidence, "chapter": ch,
                     "table": table, "row": row, "direction": direction, "strength": strength, "evidence_class": ec})

    small = dep[~dep["bank_id"].isin(FOCUS)].groupby("bank_id")["cost_of_deposits"].mean()
    gm = gap.groupby("subsection")["contribution"].mean()
    span = f"FY{int(gap['fiscal_year'].min())}-{str(int(gap['fiscal_year'].max()))[2:]}"

    def gsub(code: str) -> float:
        return float(gm[[s for s in gm.index if s.startswith(code)]].sum())

    for b in FOCUS:
        sign = 1 if b == "B02" else -1  # gap contributions are NMB - CRDB
        # scale economies
        s = sc.loc[b]
        ev(b, "scale economies", f"OpEx elasticity to assets {s['elasticity']:.2f} [{s['ci_low']:.2f}, "
           f"{s['ci_high']:.2f}], {s['years']}: {s['reading']}", "4.5", "ch04_4_5_scale_elasticity.csv",
           f"bank_id={b}", "supports" if s["ci_high"] < 1 else "contradicts",
           "strong" if s["ci_high"] < 1 else "moderate", "Derived")
        # low-cost deposits
        cd = dep[dep["bank_id"] == b]["cost_of_deposits"].mean()
        ev(b, "low-cost deposits", f"mean cost of deposits {pct(cd, 2)} vs smaller banks' median "
           f"{pct(small.median(), 2)} (n={len(small)})", "6", "ch06_deposits.csv", f"bank_id={b}", "supports",
           "strong", "Derived")
        f = fr6[fr6["bank_id"] == b].dropna(subset=["net_franchise_central_roe_pts"])
        lf = f.iloc[-1]
        ev(b, "low-cost deposits", f"net deposit franchise (central) FY{int(lf['fiscal_year'])} "
           f"{pct(lf['net_franchise_central_roe_pts'])} of ROE; mean FY2019-25 "
           f"{pct(f.loc[f['fiscal_year'] >= 2019, 'net_franchise_central_roe_pts'].mean())} (sign varies with the "
           "T-bill rate)", "6", "ch06_franchise.csv", f"bank_id={b}, net_franchise_central_roe_pts",
           "supports" if lf["net_franchise_central_roe_pts"] > 0 else "contradicts", "moderate", "Estimated")
        g42 = gsub("4.2")
        ev(b, "low-cost deposits", f"4.0 funding terms, mean NMB - CRDB ROE-gap contribution {span} "
           f"{g42 * 100:+.2f} pp (in {'NMB' if g42 > 0 else 'CRDB'}'s favour)", "4.0",
           "ch04_4_0_gap_attribution.csv", "subsection=4.2*", "supports" if sign * g42 > 0 else "contradicts",
           "moderate", "Derived")
        ev(b, "low-cost deposits", f"H5.3 (persistence higher with a stronger franchise): {h5.loc['H5.3', 'verdict']}"
           " (6 banks)", "5, 6", "ch05_hypotheses.csv", "hypothesis=H5.3", "contradicts", "weak", "Interpretation")
        # switching costs
        bb = beta[(beta["bank_id"] == b) & (beta["deposits"] == "all deposits")
                  & beta["rate"].str.startswith("T-bill")].iloc[0]
        ev(b, "customer switching costs", f"deposit beta on the T-bill {bb['beta']:.3f} [{bb['beta_lo']:.3f}, "
           f"{bb['beta_hi']:.3f}], {bb['years']}: deposit rates barely follow market rates", "6",
           "ch06_deposit_beta.csv", f"bank_id={b}, all deposits, T-bill", "supports", "moderate", "Estimated")
        cs = dep[dep["bank_id"] == b].dropna(subset=["casa_share"])
        if len(cs):
            ev(b, "customer switching costs", f"CASA share FY{int(cs['fiscal_year'].iloc[-1])} "
               f"{pct(cs['casa_share'].iloc[-1])}", "6", "ch06_deposits.csv", f"bank_id={b}, casa_share",
               "supports", "weak", "Derived")
        ev(b, "customer switching costs", "transaction fees and transactional deposits exposed to mobile-money "
           "competition (4.7 question 8)", "4.7", "ch04_business_model_map.csv",
           "dimension=question 8: exposure to disruption", "contradicts", "moderate", "Interpretation")
        # limited competition
        c = cal[(cal["bank_id"] == b)].set_index("specification")
        ev(b, "limited competition", f"break-adjusted lambda {c.loc['ar1_break_episode', 'lambda']:.2f} "
           f"[{c.loc['ar1_break_episode', 'lambda_lo']:.2f}, {c.loc['ar1_break_episode', 'lambda_hi']:.2f}]: "
           "excess returns persist once 2015-18 is dummied out; H5.1 conditionally supported", "5",
           "ch05_calibration_candidates.csv",
           f"bank_id={b}, specification=ar1_break_episode", "supports", "moderate", "Estimated")
        ev(b, "limited competition", f"rank persistence of ROE across the panel: mean Spearman "
           f"{rk['spearman'].mean():.2f} (lag {int(rk['lag'].min())}-{int(rk['lag'].max())})", "5",
           "ch05_rank_persistence.csv", "all rows", "supports" if rk["spearman"].mean() > 0.5 else "contradicts",
           "weak", "Derived")
        ev(b, "limited competition", "concentration and entry (chapter 1) not yet built: Part II alone cannot "
           "separate limited competition from efficiency", "1", "(chapter 1 pending)", "", "contradicts", "weak",
           "Interpretation")
        # regulatory advantages
        dsr = ds[(ds["kind"] == "dsib_search") & (ds["scope"] == b)]
        ev(b, "regulatory advantages", "the statutory minimum reserve is a cost borne by all banks (netted in the "
           "chapter 6 franchise); no regulatory advantage specific to the bank found in Part II; D-SIB search: "
           + (f"{int(dsr['value'].iloc[0])} passage(s)" if len(dsr) else "n/a")
           + (" (NMB designated Dec 2025: a higher requirement, not an advantage)" if b == "B02" else ""),
           "6, 7", "ch06_franchise.csv; ch07_dsib_search.csv", f"bank_id={b}", "contradicts", "weak",
           "Interpretation")
        # distribution and market access
        g43 = gsub("4.3")
        ev(b, "distribution and market access", f"4.0 asset deployment (yield and earning-asset share), mean NMB - "
           f"CRDB gap contribution {span} {g43 * 100:+.2f} pp", "4.0", "ch04_4_0_gap_attribution.csv",
           "subsection=4.3*", "supports" if sign * g43 > 0 else "contradicts", "moderate", "Derived")
        tx = tr[tr["bank_id"] == b].dropna(subset=["fee_to_avg_assets"])
        co = comp[comp["bank_id"] == b].dropna(subset=["nonii_share"])
        ev(b, "distribution and market access", f"fees / avg assets FY{int(tx['fiscal_year'].iloc[-1])} "
           f"{pct(tx['fee_to_avg_assets'].iloc[-1], 2)}; non-interest income share "
           f"{pct(co['nonii_share'].iloc[-1])}", "4.1, 4.4", "ch04_4_4_transactions.csv; ch04_4_1_composition.csv",
           f"bank_id={b}, latest year", "supports", "weak", "Derived")
        # cycle
        h = h81[(h81["bank_id"] == b) & (h81["current_definition"] == "FY2025")].iloc[0]
        ev(b, "temporary credit-cycle conditions", f"FY2025 cost of risk {pct(h['current_cost_of_risk'], 2)} vs TTC "
           f"{pct(h['ttc_primary'], 2)}, gap {pct(h['gap'], 2, sign=True)} [{pct(h['gap_lo_block'], 2, sign=True)}, "
           f"{pct(h['gap_hi_block'], 2, sign=True)}]: H8.1 {h['verdict']}", "8", "ch08_h81_cost_of_risk_gap.csv",
           f"bank_id={b}, current_definition=FY2025", "supports", "strong", "Estimated")
        s18 = sl[(sl["bank_id"] == b) & (sl["fiscal_year"] == 2025)].set_index("cost_of_risk_scenario")
        ev(b, "temporary credit-cycle conditions", f"stress lens: FY2025 ROE {pct(s18.loc['actual', 'roe_scenario'])} "
           f"-> {pct(s18.loc['2015-18 mean (own)', 'roe_scenario'])} at the 2015-18 mean cost of risk "
           f"({pct(s18.loc['2015-18 mean (own)', 'roe_change_vs_actual'], sign=True)}); "
           f"{pct(s18.loc['TTC primary (FY2010-25 mean)', 'roe_change_vs_actual'], sign=True)} at TTC",
           "8", "ch08_stress_lens.csv", f"bank_id={b}, fiscal_year=2025", "supports", "strong", "Scenario")
        ttc, brk = c.loc["through_cycle_mean", "er_star"], c.loc["ar1_break_episode", "er_star"]
        ev(b, "temporary credit-cycle conditions", f"long-run excess return: through-the-cycle mean {pct(ttc)} "
           f"[{pct(c.loc['through_cycle_mean', 'er_star_lo'])}, {pct(c.loc['through_cycle_mean', 'er_star_hi'])}] "
           f"vs break-adjusted (stress excluded) {pct(brk)}: {pct(1 - ttc / brk, 0)} of the break-adjusted ER* "
           "depends on excluding the stress", "5", "ch05_calibration_candidates.csv",
           f"bank_id={b}, through_cycle_mean vs ar1_break_episode", "supports",
           "strong" if ttc < 0.02 else "moderate", "Estimated")
        if b == "B01":
            ev(b, "temporary credit-cycle conditions", "H7.1: raise-adjusted RORE clears COE + 200 bp only in windows ending in "
               "the post-2018 recovery (FY2021-25); excluding stress-start windows the median is below COE (open issue 4)", "7",
               "ch07_rore_summary.csv; ch07_rore_windows.csv", "bank_id=B01, k=5", "supports", "moderate",
               "Derived")
        # overall answer
        gen = sc.loc[b, "ci_high"] < 1 and ttc > 0.02
        if gen:
            ans = (f"GENUINE ADVANTAGE with a cyclical component: through-the-cycle excess return {pct(ttc)} is "
                   f"positive (interval reaches {pct(c.loc['through_cycle_mean', 'er_star_lo'])}), scale economies "
                   "and a low-cost deposit franchise are evidenced; today's ROE is flattered by a cost of risk below "
                   "its through-the-cycle level")
        else:
            ans = (f"STRUCTURAL FEATURES AND CYCLE: the deposit franchise is real, but the through-the-cycle excess "
                   f"return is about zero ({pct(ttc)}), scale economies are not evidenced and the ROE edge over NMB "
                   "comes from leverage; recent excess returns rest on the post-2018 credit recovery")
        ev(b, "OVERALL: genuine advantage, structural features or cycle?", ans, "4-8",
           "part2_rent_sources.csv (rows above)", f"bank_id={b}", "supports" if gen else "contradicts",
           "moderate", "Interpretation")
        if b == "B01":
            ev(b, "financial structure (not a §0 explanation)", f"4.0 leverage TA/E: mean NMB - CRDB gap "
               f"contribution {gsub('4.7 Leverage') * 100:+.2f} pp (CRDB's main ROE edge); leverage raises ROE without "
               "creating rent", "4.0, 7", "ch04_4_0_gap_attribution.csv; ch07_leverage.csv", "component=ta_e",
               "contradicts", "strong", "Derived")
    return pd.DataFrame(rows)


# --- 3. hypotheses -----------------------------------------------------------------------------------------------


def hypotheses_status(root: Path) -> pd.DataFrame:
    stm = {}
    for line in (root / "docs" / "RESEARCH_HYPOTHESES.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"- (H[4-8]\.\d) (.+)", line.strip())
        if m:
            stm[m.group(1)] = m.group(2)
    rows = []
    bm = _t(root, "ch04_business_model_map.csv")
    for h in ("H4.1", "H4.2"):
        r = bm[bm["dimension"] == f"hypothesis {h}"].iloc[0]
        v = str(r["item"])
        verdict = "not supported" if v.upper().startswith("NOT SUPPORTED") else "supported" if v.upper().startswith(
            "SUPPORTED") else "inconclusive"
        rows.append({"hypothesis": h, "chapter": "4", "verdict": verdict, "verdict_by_bank": "joint",
                     "evidence": v, "source_table": "ch04_business_model_map.csv", "review_note": ""})
    for ch, f in (("5", "ch05_hypotheses.csv"), ("6", "ch06_hypotheses.csv"), ("7", "ch07_hypotheses.csv"),
                  ("8", "ch08_hypotheses.csv")):
        x = _t(root, f)
        for r in x.itertuples():
            rows.append({"hypothesis": r.hypothesis, "chapter": ch, "verdict": r.verdict,
                         "verdict_by_bank": getattr(r, "verdict_by_bank", getattr(r, "scope", "joint")),
                         "evidence": r.evidence, "source_table": f, "review_note": ""})
    out = pd.DataFrame(rows)
    out.insert(1, "statement", out["hypothesis"].str.extract(r"(H\d\.\d)")[0].map(stm))
    # open issue 4: H7.1 CRDB wording
    w = _t(root, "ch07_rore_windows.csv")
    x = w[(w["bank_id"] == "B01") & (w["k"] == 5) & w["in_h71"].astype(bool) & ~w["stress_start_flag"].astype(bool)]
    allw = w[(w["bank_id"] == "B01") & (w["k"] == 5) & w["in_h71"].astype(bool)]
    above = allw[allw["rore_adjusted"] > allw["coe_window"] + 0.02].sort_values("start_year")
    s = _t(root, "ch07_rore_summary.csv")
    med = s[(s["bank_id"] == "B01") & (s["k"] == 5) & (s["series"] == "rore_adjusted")]["median_excl_stress_start"]
    out.loc[out["hypothesis"] == "H7.1", "review_note"] = (
        f"Open issue 4: for CRDB, excluding windows that start in FY2015-18 leaves {len(x)} windows with a "
        f"raise-adjusted median RORE of {pct(med.iloc[0])} and a median excess over the window COE of "
        f"{pct((x['rore_adjusted'] - x['coe_window']).median())}; windows clearing COE + 200 bp start in "
        f"{', '.join('FY' + str(int(v)) for v in above['start_year']) or 'none'} (ending FY"
        f"{int(above['end_year'].min()) if len(above) else 0}-{int(above['end_year'].max()) if len(above) else 0}). "
        "Wording accepted by the owner (2026-11-09): 'supported for NMB; for CRDB above COE only in windows that end in the "
        "post-2018 recovery, so not supported through the cycle; inconclusive overall'.")
    out.loc[out["hypothesis"] == "H4.1", "review_note"] = (
        "Tests the NMB - CRDB gap, not the level of excess return: the deposit franchise still lifts both banks' "
        "ROE (chapter 6), but it is not what separates them.")
    out.loc[out["hypothesis"] == "H5.2", "review_note"] = (
        "Through-the-cycle AR(1) lambda = 1.00 for NMB and CRDB, so ER* is undefined there; the break-adjusted ER* is "
        "positive for both (open issue 3).")
    out.loc[out["hypothesis"] == "H8.2", "review_note"] = (
        "NMB's RWA is Bank-solo and its profit Group (D2): the gap includes a small basis effect (open issue 5).")
    out["evidence_class"] = "Interpretation"
    return out


# --- 4. calibration ranges -----------------------------------------------------------------------------------------


def calibration_inputs(root: Path, t: dict[str, str]) -> pd.DataFrame:
    cal = _t(root, "ch05_calibration_candidates.csv")
    coe = _t(root, "ch05_coe_annual.csv")
    ttc = _t(root, "ch08_ttc_cost_of_risk.csv")
    risk = _t(root, "ch04_4_6_risk.csv")
    ol = _t(root, "ch04_4_5_operating_leverage.csv")
    pay = _t(root, "ch07_payout.csv")
    d1 = _t(root, "ch07_distributable_capacity.csv")
    d2 = _t(root, "ch07_distributable_capacity_total.csv")
    cap = _t(root, "ch07_capital.csv")
    beta = _t(root, "ch06_deposit_beta.csv")
    cv = {r.date_label: r.coe for r in coe.itertuples() if r.date_label in ("FY2025", "valuation_current")}
    rows: list[dict] = []

    def add(b, param, unit, lo, hi, ilo, ihi, refs, src, srow, ec, note=""):
        rows.append({"bank_id": b, "bank": t[b], "parameter": param, "unit": unit, "range_low": lo, "range_high": hi,
                     "interval_low": ilo, "interval_high": ihi, "reference_points": refs, "source_table": src,
                     "source_rows": srow, "note": note, "selection": D8, "evidence_class": ec})

    for b in FOCUS:
        c = cal[cal["bank_id"] == b].set_index("specification")
        cases = {"bear (D50)": c.loc["bear_zero_rent"], "base (D50, TTC mean)": c.loc["through_cycle_mean"],
                 "bull (D50, break-adjusted)": c.loc["ar1_break_episode"]}
        pts = {k: float(v["er_star"]) for k, v in cases.items()}
        ilo = min(0.0, float(c.loc["through_cycle_mean", "er_star_lo"]))
        ihi = float(c.loc["ar1_break_episode", "er_star_hi"])
        add(b, "long-run excess return ER*", "ratio (ROE - COE)", min(pts.values()), max(pts.values()), ilo, ihi,
            "; ".join(f"{k} {pct(v)}" for k, v in pts.items()) + "; D51 alternative beta: "
            "ch05_beta_sensitivity.csv",
            "ch05_calibration_candidates.csv", "bear_zero_rent, through_cycle_mean, ar1_break_episode", "Estimated",
            "D50 cases to be confirmed at 09a (open issue 2)" + ("; CRDB base is close to bear" if b == "B01" else ""))
        lo_c, hi_c = min(cv.values()), max(cv.values())
        add(b, "sustainable ROE (ER* + COE)", "ratio", lo_c + min(pts.values()), hi_c + max(pts.values()),
            lo_c + ilo, hi_c + ihi, f"COE FY2025 {pct(cv.get('FY2025'))}, valuation-current "
            f"{pct(cv.get('valuation_current'))}; ER* cases as above", "ch05_calibration_candidates.csv; "
            "ch05_coe_annual.csv", "FY2025, valuation_current", "Estimated")
        be = c.loc["ar1_break_episode"]
        add(b, "fade speed lambda", "AR(1) coefficient", float(be["lambda_lo"]), float(be["lambda_hi"]),
            float(be["lambda_lo"]), float(be["lambda_hi"]),
            f"break-adjusted Andrews lambda {be['lambda']:.2f}; through-the-cycle lambda 1.00 (no mean reversion "
            "established, cannot serve as a fade speed)", "ch05_calibration_candidates.csv",
            "ar1_break_episode", "Estimated", "fade uses the break-adjusted lambda by necessity (open issue 3)")
        tt = ttc[ttc["bank_id"] == b].set_index("definition")
        grid = ttc[(ttc["bank_id"] == b) & ttc["definition_type"].isin(["primary", "boundary grid"])]
        prim = tt[tt["definition_type"] == "primary"].iloc[0]
        add(b, "through-the-cycle cost of risk", "ratio of avg gross loans", float(grid["ttc_mean"].min()),
            float(grid["ttc_mean"].max()), float(tt.loc["IFRS 9 regime FY2018-25", "ttc_mean"]),
            float(tt.loc["stress only FY2015-18", "ttc_mean"]),
            f"primary FY2010-25 {pct(prim['ttc_mean'], 2)} (loan-weighted {pct(prim['ttc_pooled'], 2)}); IAS 39 "
            f"{pct(tt.loc['IAS 39 regime FY2010-17', 'ttc_mean'], 2)}; IFRS 9 "
            f"{pct(tt.loc['IFRS 9 regime FY2018-25', 'ttc_mean'], 2)}; stress {pct(tt.loc['stress only FY2015-18', 'ttc_mean'], 2)}",
            "ch08_ttc_cost_of_risk.csv", "primary, boundary grid, sub-periods", "Derived",
            "range = 25 cycle boundaries; interval columns = IFRS 9-regime mean to stress mean")
        r = risk[risk["bank_id"] == b].dropna(subset=["nim"])
        rr = r[r["fiscal_year"] >= 2019]
        add(b, "NIM (NII / avg interest-earning assets)", "ratio", float(rr["nim"].min()), float(rr["nim"].max()),
            float(r["nim"].min()), float(r["nim"].max()),
            f"FY2025 {pct(r['nim'].iloc[-1], 2)}; mean FY2010-25 {pct(r.loc[r['fiscal_year'] >= 2010, 'nim'].mean(), 2)}"
            f"; trend FY2019-25 {np.polyfit(rr['fiscal_year'], rr['nim'], 1)[0] * 100:+.2f} pp a year",
            "ch04_4_6_risk.csv", f"bank_id={b}, nim", "Derived", "range = FY2019-25; interval = full history")
        o = ol[ol["bank_id"] == b].dropna(subset=["cost_to_income"])
        oo = o[o["fiscal_year"] >= 2019]
        slope = np.polyfit(oo["fiscal_year"], oo["cost_to_income"], 1)[0]
        add(b, "cost-to-income path", "ratio", float(oo["cost_to_income"].min()), float(oo["cost_to_income"].max()),
            float(o["cost_to_income"].min()), float(o["cost_to_income"].max()),
            f"FY2025 {pct(o['cost_to_income'].iloc[-1])}; trend FY2019-25 {slope * 100:+.2f} pp a year; FY2017-18 "
            f"peak {pct(o.loc[o['fiscal_year'].between(2015, 2018), 'cost_to_income'].max())}",
            "ch04_4_5_operating_leverage.csv", f"bank_id={b}, cost_to_income", "Derived",
            "range = FY2019-25 levels; the path is the trend, not a forecast")
        p = pay[pay["bank_id"] == b].dropna(subset=["payout_total"])
        dd = pd.concat([d1[d1["bank_id"] == b], d2[d2["bank_id"] == b]])
        add(b, "payout ratio", "ratio of attributable profit",
            float(p.loc[p["fiscal_year"] >= 2010, "payout_total"].min()),
            float(p.loc[p["fiscal_year"] >= 2010, "payout_total"].max()),
            float(dd["payout_max_steady_state"].min()), float(dd["payout_max_steady_state"].max()),
            f"actual mean FY2010-25 {pct(p.loc[p['fiscal_year'] >= 2010, 'payout_total'].mean())}; FY2025 "
            f"{pct(p['payout_total'].iloc[-1])}; steady-state ceiling over Tier 1 / total capital, 0-4 pp buffers"
            + (", readings A/B" if b == "B01" else ""),
            "ch07_payout.csv; ch07_distributable_capacity.csv; ch07_distributable_capacity_total.csv",
            f"bank_id={b}", "Scenario", "range = actual history; interval = capital-constrained ceilings")
        k = cap[(cap["bank_id"] == b)].dropna(subset=["tier1_ratio"]).iloc[-1]
        hr = [k["tier1_headroom"], k["total_headroom"]]
        hb = [k["tier1_headroom_reading_b"], k["total_headroom_reading_b"]] if b == "B01" else hr
        add(b, "capital headroom (FY2025)", "ratio of RWA", float(min(hr)), float(max(hr)), float(np.nanmin(hb)),
            float(np.nanmax(hr)), f"binding constraint {k['binding_constraint']}; Tier 1 {pct(k['tier1_headroom'])}, "
            f"total {pct(k['total_headroom'])}" + (f"; reading B {pct(hb[0])} / {pct(hb[1])}" if b == "B01" else "")
            + f"; basis {k['capital_basis']}", "ch07_capital.csv", f"bank_id={b}, fiscal_year=2025",
            "Derived", ("NMB Bank-solo capital vs Group profit (open issue 5); D-SIB surcharge unknown (open "
                        "issue 1)") if b == "B02" else "reading B is a sensitivity (D52 accepted)")
        bt = beta[(beta["bank_id"] == b) & (beta["deposits"] == "all deposits")]
        tb = bt[bt["rate"].str.startswith("T-bill")].iloc[0]
        po = bt[bt["rate"].str.startswith("policy")].iloc[0]
        add(b, "deposit beta (all deposits)", "d cost of deposits / d rate", float(min(tb["beta"], po["beta"])),
            float(max(tb["beta"], po["beta"])), float(min(tb["beta_lo"], po["beta_lo"])),
            float(max(tb["beta_hi"], po["beta_hi"])),
            f"T-bill {tb['beta']:.3f} [{tb['beta_lo']:.3f}, {tb['beta_hi']:.3f}], {tb['years']}; policy rate "
            f"{po['beta']:.3f}", "ch06_deposit_beta.csv", f"bank_id={b}, all deposits", "Estimated",
            "annual data, about 14 observations: intervals wide")
    return pd.DataFrame(rows)


# --- 5. summary and note ----------------------------------------------------------------------------------------


def summary_md(rs: pd.DataFrame, hs: pd.DataFrame, ci: pd.DataFrame, cons: pd.DataFrame, t: dict[str, str],
               root: Path) -> str:
    sl = _t(root, "ch08_stress_lens.csv")
    d = sl[(sl["fiscal_year"] == 2025) & (sl["cost_of_risk_scenario"] == "2015-18 mean (own)")].set_index("bank_id")
    dd = (-d["roe_change_vs_actual"] * 100).round(1)
    cap = _t(root, "ch07_capital.csv")
    nmb = cap[cap["bank_id"] == "B02"].dropna(subset=["tier1_ratio"]).iloc[-1]
    ov = rs[rs["explanation"].str.startswith("OVERALL")].set_index("bank_id")["evidence"]
    cv = ci.set_index(["bank_id", "parameter"])
    hv = hs.set_index("hypothesis")["verdict"]

    def rng(b, p, d=1):
        r = cv.loc[(b, p)]
        return f"{pct(r['range_low'], d)} to {pct(r['range_high'], d)}"

    lines = ["# Part II executive summary (DRAFT for owner review)", "",
             "> DRAFT generated by `tzbank.chapters.part2_synthesis` (prompt 08). Evidence classes as in the source "
             "tables; interpretation is marked. Not for circulation before owner review.", "",
             "**Why do NMB and CRDB earn returns above their cost of equity, and will they keep doing so?**", ""]
    for b in ("B02", "B01"):
        lines.append(f"- **{t[b]}** (Interpretation): {ov[b]}.")
    lines += ["", "**What separates them (chapter 4).** The banks run different profit engines (H4.2 "
              f"{hv.get('H4.2')}): NMB earns more on its assets and pays less for funding, CRDB earns its ROE edge "
              f"from leverage. Funding is not the main source of the gap (H4.1 {hv.get('H4.1')}), though the deposit "
              "franchise lifts both banks' ROE.", "",
              f"**Persistence (chapter 5).** Excess returns persist (H5.1 {hv.get('H5.1')}) once the 2015-18 episode "
              "is treated as a break; through the cycle, mean reversion is not established (lambda 1.00), so the "
              f"long-run excess return is only identified on the break-adjusted model (H5.2 {hv.get('H5.2')}).", "",
              f"**Funding (chapter 6).** Both banks fund far below the T-bill rate and well below smaller banks; the "
              f"net deposit franchise is positive in recent years but its sign follows the T-bill rate (H6.1 "
              f"{hv.get('H6.1')}).", "",
              f"**Capital (chapter 7).** NMB has reinvested above its cost of equity; CRDB only in windows ending in the post-2018 "
              f"recovery (H7.1 {hv.get('H7.1')}). CRDB runs close to its requirement and total capital binds for "
              f"both banks in recent years; NMB's FY2025 headroom is {pct(nmb['binding_headroom'])} on its binding "
              f"constraint.", "",
              f"**Credit (chapter 8).** Today's cost of risk is below the through-the-cycle level for both banks (H8.1 "
              f"{hv.get('H8.1')}); at 2015-18 credit costs FY2025 ROE would be "
              f"{dd.get('B02'):.1f} pp (NMB) and {dd.get('B01'):.1f} pp (CRDB) lower. After "
              f"normalising credit cost NMB still earns more per unit of RWA (H8.2 {hv.get('H8.2')}).", "",
              "**Calibration ranges for 09a (D8: ranges, not choices).**", "",
              "| Input | NMB | CRDB |", "| --- | --- | --- |"]
    for p, d in (("long-run excess return ER*", 1), ("sustainable ROE (ER* + COE)", 1), ("through-the-cycle cost of "
                 "risk", 2), ("NIM (NII / avg interest-earning assets)", 1), ("cost-to-income path", 1),
                 ("payout ratio", 0), ("capital headroom (FY2025)", 1)):
        lines.append(f"| {p} | {rng('B02', p, d)} | {rng('B01', p, d)} |")
    for p in ("fade speed lambda", "deposit beta (all deposits)"):
        lines.append(f"| {p} | {cv.loc[('B02', p), 'range_low']:.2f} to {cv.loc[('B02', p), 'range_high']:.2f} | "
                     f"{cv.loc[('B01', p), 'range_low']:.2f} to {cv.loc[('B01', p), 'range_high']:.2f} |")
    bad = cons["status"].str.startswith("CONTRADICTION").sum()
    lines += ["", f"**Consistency.** {len(cons)} cross-chapter checks; unresolved contradictions: {bad} "
              "(`reports/part2_consistency.md`).", "",
              "**For the owner.** Confirm the D50 cases (CRDB's base case is close to its bear case); the H7.1 "
              "wording for CRDB; the D-SIB surcharge treatment for NMB (open issues 1-5, `docs/OPEN_ISSUES.md`)."]
    return "\n".join(lines) + "\n"


NOTE = """# Part II synthesis: why the banks earn rents, and what that implies for 09a

Generated by `tzbank.chapters.part2_synthesis` (stage `synthesis`, prompt 08). Blueprint §0 cross-cutting question;
chapters 4-8; D8 (calibration ranges only).

## Research question

Are NMB's and CRDB's economic rents generated by genuine competitive advantages, by structural features of Tanzanian
banking, or by the credit cycle (blueprint §0)? Which evidence-based ranges should 09a use?

## Variable definitions

Every number is read from a chapter 4-8 table (column `table` / `source_table`); no new financial quantity is
computed except the consistency differences and the range bounds (min / max over the listed reference points).
Direction: whether the evidence supports or contradicts the explanation for that bank. Strength: strong (a test
with an interval that excludes the null, or a large mechanical effect), moderate (a clear point estimate or a
ranking), weak (descriptive, indirect or untested).

## Sources and transformation trail

Inputs: `outputs/tables/ch04_*.csv` to `ch08_*.csv`, `docs/RESEARCH_HYPOTHESES.md`. Functions: `consistency`,
`rent_sources`, `hypotheses_status`, `calibration_inputs`, `summary_md`. Outputs: `reports/part2_consistency.md`,
`outputs/tables/part2_rent_sources.csv`, `hypotheses_status.csv`, `calibration_inputs.csv`,
`reports/part2_executive_summary.md`.

## Validation test

`tests/test_part2.py`: no unresolved contradiction; 4.6 and chapter 8 return on RWA agree; every bank has a row for
each §0 explanation and an overall answer; every H4-H8 hypothesis is in the register; every calibration parameter has
a range with low <= high for both banks and no chosen value.

## Results

### Consistency (Derived)

{cons}

### Rent sources (Interpretation over Derived / Estimated evidence)

{rent}

### Hypothesis register (Interpretation)

{hyp}

### Calibration ranges for 09a (D8)

{cal}

## Evidence class

Consistency checks and ranges drawn from Derived tables: **Derived**; ER*, lambda, deposit beta and sustainable ROE:
**Estimated**; payout ceilings: **Scenario**; directions, strengths, verdicts and overall answers: **Interpretation**.

## Limitations

- **D54 (accepted) is a judgement rule**, not a statistic: strength labels follow its stated rule, and the overall
  answer calls a bank a 'genuine advantage' only when scale economies are evidenced and the through-the-cycle mean
  excess return exceeds a **2 pp threshold**.
- Limited competition and regulatory advantages are Part I questions (chapters 1-3, not yet built); Part II evidence
  on them is indirect.
- Ranges span the chapter evidence; they are not probability intervals for 09a inputs.
- Basis effects (NMB Bank-solo capital with Group profit) and the IAS 39 / IFRS 9 span remain (open issues 5, D53).

## Interpretation

See `reports/part2_executive_summary.md`.

## Cross-cutting explanations of rent (blueprint §0)

This note is the cross-chapter answer: see the rent-source table and the overall rows per bank.
"""


def write_synthesis(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    cons = consistency(root)
    rs = rent_sources(root, t)
    hs = hypotheses_status(root)
    ci = calibration_inputs(root, t)
    rep, tab = path("reports", root), path("tables", root)
    files = [write_csv(cons, rep / "part2_consistency.csv", require_evidence=True),
             write_text(consistency_md(cons, root), rep / "part2_consistency.md"),
             write_csv(rs, tab / "part2_rent_sources.csv", require_evidence=True),
             write_csv(hs, tab / "hypotheses_status.csv", require_evidence=True),
             write_csv(ci, tab / "calibration_inputs.csv", require_evidence=True),
             write_text(summary_md(rs, hs, ci, cons, t, root), rep / "part2_executive_summary.md")]
    cons_txt = "\n".join(f"- {r.area}: {r.check}: {r.status} ({r.n_compared} compared, {r.n_differ} differ)"
                         for r in cons.itertuples())
    rent_txt = "\n".join(f"- {r.bank} / {r.explanation}: {r.direction}, {r.strength} ({r.evidence_class}): "
                         f"{r.evidence} [`{r.table}`]" for r in rs.itertuples())
    hyp_txt = "\n".join(f"- {r.hypothesis}: {r.verdict}" + (f" ({r.verdict_by_bank})" if isinstance(
        r.verdict_by_bank, str) and r.verdict_by_bank not in ("joint", "") else "")
        + (f". Review: {r.review_note}" if r.review_note else "") for r in hs.itertuples())
    cal_txt = "\n".join(f"- {r.bank} {r.parameter}: {r.range_low:.4f} to {r.range_high:.4f} (interval "
                        f"{r.interval_low:.4f} to {r.interval_high:.4f}); {r.reference_points} [`{r.source_table}`]"
                        for r in ci.itertuples())
    nf = write_text(NOTE.format(cons=cons_txt, rent=rent_txt, hyp=hyp_txt, cal=cal_txt), path("notes", root) /
                    "part2.md")
    files.append(nf)
    m = {"chapter": "part2", "title": "Part II synthesis and calibration inputs", "hypotheses": [],
         "status": "built",
         "tables": [{"path": rel(f, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for f, d, c in ((files[2], "Rent sources by bank and §0 explanation", "Interpretation"),
                                    (files[3], "Hypothesis register H4-H8", "Interpretation"),
                                    (files[4], "Calibration ranges for 09a (D8)", "Estimated"),
                                    (files[0], "Cross-chapter consistency checks", "Derived"))],
         "figures": [], "notes": [rel(nf, root)], "reports": [rel(files[1], root), rel(files[5], root)],
         "inputs": ["outputs/tables/ch04_*.csv ... ch08_*.csv", "docs/RESEARCH_HYPOTHESES.md"],
         "module": "tzbank.chapters.part2_synthesis", "run_log_stage": "synthesis"}
    files.append(write_json(m, path("manifests", root) / "part2.json"))
    return files

