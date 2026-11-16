"""Chapter 8: credit risk and capital efficiency (blueprint §2 credit and capital efficiency, §4 chapter 8; H8.1, H8.2;
limitation L4, IAS 39 -> IFRS 9 in FY2018).

cost of risk_t          = impairment_t / average gross loans_t                       (blueprint §2)
TTC cost of risk        = mean of the annual cost of risk over a stated cycle that includes FY2015-18
return on RWA           = net profit / average RWA                                    (blueprint §2)
risk-adjusted RoRWA     = (PPOP_t - TTC cost of risk x average gross loans_t) / average RWA_t,  PPOP = PBT + impairment
RWA density             = RWA / total assets
stress lens (Scenario)  = the 4.0 tree ROE with Imp/TA replaced by CoR_stress x avg gross loans / avg TA

The IFRS 9 break is never smoothed: annual values are reported as computed, the regime is a column, figures draw the
IAS 39 and IFRS 9 segments separately, and the FY2018 transition adjustment (taken to equity on 1 January 2018, not
through impairment) is reported in its own table.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, FOCUS, avg, col, div, lag, pct, plt_subplots, save, style, tickers, wide
from tzbank.chapters.ch04.dupont_attribution import components, roe_full
from tzbank.chapters.ch07_capital import capital_basis
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text

IFRS9_YEAR = 2018
STRESS = (2015, 2018)
PRIMARY = (2010, 2025)  # full cycle: expansion 2010-14, stress 2015-18, recovery 2019-25
CURRENT = 2025
LEVEL = 0.90


def regime(y: int) -> str:
    return "IFRS 9" if y >= IFRS9_YEAR else "IAS 39"


# --- cost of risk and asset quality ---------------------------------------------------------------------------------


def credit_history(w: pd.DataFrame) -> pd.DataFrame:
    """Annual cost of risk, NPL ratio and coverage per bank-year, as computed (no smoothing across FY2018)."""
    imp = col(w, "impairment_charges")
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"].astype(int)})
    out["impairment"] = imp
    out["avg_gross_loans"] = avg(w, "gross_loans")
    out["cost_of_risk"] = div(imp, out["avg_gross_loans"])
    out["npl_ratio"] = div(col(w, "stage3_loans_gross"), col(w, "gross_loans"))
    out["npl_ratio_reported"] = col(w, "npl_ratio")
    out["coverage"] = div(col(w, "loan_loss_allowance").abs(), col(w, "stage3_loans_gross"))
    out["coverage_reported"] = col(w, "npl_coverage_ratio")
    out["accounting_regime"] = [regime(y) for y in out["fiscal_year"]]
    out["npl_definition"] = np.where(out["fiscal_year"] >= IFRS9_YEAR, "IFRS 9 stage 3 gross loans",
                                     "IAS 39 impaired (non-performing) gross loans")
    out["ifrs9_transition_year"] = out["fiscal_year"] == IFRS9_YEAR
    out["break_note"] = np.where(out["ifrs9_transition_year"],
                                 "FY2018: first IFRS 9 year (expected-loss impairment); transition adjustment to equity "
                                 "on 1 Jan 2018 is NOT in this year's impairment (ch08_ifrs9_transition.csv)", "")
    out["evidence_class"] = "Derived"
    keep = out[["cost_of_risk", "npl_ratio", "npl_ratio_reported", "coverage", "coverage_reported"]].notna().any(axis=1)
    return out[keep].reset_index(drop=True)


def _ttc(x: pd.DataFrame, start: int, end: int) -> dict:
    s = x[x["fiscal_year"].between(start, end)].dropna(subset=["cost_of_risk"])
    pooled = s["impairment"].sum() / s["avg_gross_loans"].sum() if len(s) else np.nan
    missing = sorted(set(range(start, end + 1)) - set(s["fiscal_year"]))
    return {"start": start, "end": end, "n_years": len(s), "ttc_mean": s["cost_of_risk"].mean() if len(s) else np.nan,
            "ttc_pooled": pooled, "years_missing": ", ".join(map(str, missing)) or "none",
            "covers_2015_18": all(y in set(s["fiscal_year"]) for y in range(STRESS[0], STRESS[1] + 1))}


def ttc_table(hist: pd.DataFrame) -> pd.DataFrame:
    """Through-the-cycle cost of risk per bank under the primary cycle and every alternative boundary (quality gate:
    sensitivity to cycle boundaries). All boundary-grid cycles include FY2015-18."""
    rows = []
    for b, x in hist.groupby("bank_id"):
        first = int(x.dropna(subset=["cost_of_risk"])["fiscal_year"].min()) if x["cost_of_risk"].notna().any() else None
        if first is None:
            continue
        defs = [("primary", "full cycle FY2010-25: expansion 2010-14, stress 2015-18, recovery 2019-25", *PRIMARY)]
        defs += [("boundary grid", f"FY{s}-{e}", s, e) for s in range(2010, 2015) for e in range(2021, 2026)
                 if (s, e) != PRIMARY]
        defs += [("sub-period", "stress only FY2015-18", *STRESS),
                 ("sub-period", "IAS 39 regime FY2010-17", 2010, IFRS9_YEAR - 1),
                 ("sub-period", "IFRS 9 regime FY2018-25", IFRS9_YEAR, 2025),
                 ("sub-period", "post-stress FY2019-25", 2019, 2025)]
        for kind, label, s, e in defs:
            rows.append({"bank_id": b, "definition_type": kind, "definition": label, **_ttc(x, s, e)})
        # excluding the IFRS 9 transition year from the primary cycle (robustness, L4)
        xe = x[x["fiscal_year"] != IFRS9_YEAR]
        rows.append({"bank_id": b, "definition_type": "robustness", "definition": "primary cycle excluding FY2018",
                     **_ttc(xe, *PRIMARY)})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


def ifrs9_transition(root: Path) -> pd.DataFrame:
    """FY2018 opening IFRS 9 adjustment, from the notes already extracted into the inputs workbook rows (TZS million in
    the source text; converted to full TZS). Reported separately, never added to the FY2018 cost of risk."""
    lg = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False, dtype={"bank_id": str})
    x = lg[lg["notes"].astype(str).str.contains("IFRS 9 transition", case=False)
           & (lg["canonical_name"] == "loan_loss_allowance")]
    num = r"([\d,]+)"
    pats = {"retained_earnings_ias39_close": r"Retained earnings " + num + r" \(",
            "ifrs9_adjustment_retained_earnings": r"less IFRS 9 (?:adjustments|impact) \(" + num + r"\)",
            "fair_value_reserve_adjustment": r"plus " + num + r" to fair value reserve",
            "equity_impact_net": r"\(net -" + num + r"\)",
            "allowance_ias39_close": r"(?:loan loss allowance " + num + r" ->|Loan allowance " + num + r" at 31 Dec 2017)",
            "allowance_ifrs9_open": r"-> " + num + r" \(\+",
            "allowance_increase": r"\(\+" + num + r"\)"}
    rows = []
    for r in x.itertuples():
        txt = str(r.notes)
        src = re.search(r"(PDF[^.|]*(?:\.\s*\d+[^.|]*)?)", txt.split("IFRS 9 transition")[-1])
        for item, p in pats.items():
            m = re.search(p, txt)
            if not m:
                continue
            v = float(next(g for g in m.groups() if g).replace(",", "")) * 1e6
            if item in ("ifrs9_adjustment_retained_earnings", "equity_impact_net"):
                v = -v
            rows.append({"bank_id": r.bank_id, "item": item, "value": v, "unit": "TZS", "basis": "Group",
                         "date": "2018-01-01 (opening)", "source_document": r.source_document,
                         "source_location": src.group(1).strip() if src else "transition note",
                         "evidence_class": "Observed"})
    out = pd.DataFrame(rows)
    if len(out):
        # equity impact: net where stated, else the retained-earnings adjustment
        for b in out["bank_id"].unique():
            o = out[out["bank_id"] == b].set_index("item")["value"]
            if "equity_impact_net" not in o.index and "ifrs9_adjustment_retained_earnings" in o.index:
                out = pd.concat([out, pd.DataFrame([{
                    "bank_id": b, "item": "equity_impact_net", "value": o["ifrs9_adjustment_retained_earnings"],
                    "unit": "TZS", "basis": "Group", "date": "2018-01-01 (opening)", "source_document": "derived",
                    "source_location": "= retained-earnings adjustment (no other reserve adjustment stated)",
                    "evidence_class": "Derived"}])], ignore_index=True)
    return out


def transition_ratios(tr: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Scale of the transition adjustment: % of 31 Dec 2017 equity and, as a memo, the cost of risk FY2018 would show
    if the allowance increase had gone through profit and loss (not used anywhere: the break is not smoothed)."""
    rows = []
    for b in tr["bank_id"].unique() if len(tr) else []:
        o = tr[tr["bank_id"] == b].set_index("item")["value"]
        g = w[(w["bank_id"] == b)].set_index("fiscal_year")
        eq17 = g["total_equity"].get(2017, np.nan) if "total_equity" in g else np.nan
        al = avg(w[w["bank_id"] == b], "gross_loans").set_axis(g.index).get(IFRS9_YEAR, np.nan)
        inc = o.get("allowance_increase", np.nan)
        if pd.isna(inc) and "ifrs9_adjustment_retained_earnings" in o.index:
            inc = np.nan  # NMB: allowance after transition not stated in the extracted note
        rows += [{"bank_id": b, "item": "equity_impact_pct_of_fy2017_equity", "value": o.get("equity_impact_net",
                                                                                              np.nan) / eq17,
                  "unit": "ratio", "evidence_class": "Derived"},
                 {"bank_id": b, "item": "memo_allowance_increase_pct_of_fy2018_avg_loans", "value": inc / al,
                  "unit": "ratio (memo only; not added to cost of risk)", "evidence_class": "Derived"}]
    return pd.DataFrame(rows)


# --- H8.1 ------------------------------------------------------------------------------------------------------------


def _block_means(v: np.ndarray, reps: int, block: int, rng: np.random.Generator) -> np.ndarray:
    from tzbank.chapters.ch05.models import block_indices

    return np.array([v[block_indices(len(v), block, rng)].mean() for _ in range(reps)])


def h81(hist: pd.DataFrame, ttc: pd.DataFrame, boot: dict, t: dict[str, str]) -> pd.DataFrame:
    """Current cost of risk (FY2025; 2023-25 mean as robustness) against the primary TTC level; interval from the
    moving-block bootstrap over the years of the cycle (block 3, D46), i.i.d. year bootstrap alongside."""
    rows = []
    for i, b in enumerate(FOCUS):
        x = hist[(hist["bank_id"] == b) & hist["fiscal_year"].between(*PRIMARY)].dropna(subset=["cost_of_risk"])
        v = x.sort_values("fiscal_year")["cost_of_risk"].to_numpy()
        if len(v) < 3:
            continue
        rng = np.random.default_rng([*boot["seed"], i])
        bm = _block_means(v, boot["reps"], boot["block"], rng)
        iid = np.array([rng.choice(v, len(v)).mean() for _ in range(boot["reps"])])
        a = (1 - LEVEL) / 2
        grid = ttc[(ttc["bank_id"] == b) & ttc["definition_type"].isin(["primary", "boundary grid"])]
        for label, cur in [(f"FY{CURRENT}", x.loc[x["fiscal_year"] == CURRENT, "cost_of_risk"].mean()),
                           (f"FY{CURRENT - 2}-{CURRENT} mean", x.loc[x["fiscal_year"] >= CURRENT - 2,
                                                                     "cost_of_risk"].mean())]:
            gap = cur - v.mean()
            lo, hi = np.quantile(cur - bm, [a, 1 - a])
            ilo, ihi = np.quantile(cur - iid, [a, 1 - a])
            verdict = "supported" if hi < 0 else "rejected" if lo > 0 else "inconclusive"
            rows.append({"bank_id": b, "bank": t[b], "current_definition": label, "current_cost_of_risk": cur,
                         "ttc_primary": v.mean(), "ttc_years": f"FY{PRIMARY[0]}-{PRIMARY[1]} ({len(v)} years)",
                         "gap": gap, "gap_lo_block": lo, "gap_hi_block": hi, "gap_lo_iid": ilo, "gap_hi_iid": ihi,
                         "share_of_cycle_definitions_with_current_below_ttc": float((cur < grid["ttc_mean"]).mean()),
                         "ttc_range_over_definitions": f"{grid['ttc_mean'].min():.4f}-{grid['ttc_mean'].max():.4f}",
                         "verdict": verdict, "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- capital efficiency --------------------------------------------------------------------------------------------


def capital_efficiency(w: pd.DataFrame, cb: pd.DataFrame, hist: pd.DataFrame, ttc: pd.DataFrame) -> pd.DataFrame:
    """Return on RWA, PPOP / RWA, risk-adjusted RoRWA at the bank's own TTC cost of risk, RWA density. RWA on the
    D2/D17 basis for NMB / CRDB (capital_long), the reported (Bank = Group) RWA for B03-B07."""
    base = w[["bank_id", "fiscal_year"]].copy()
    base["fiscal_year"] = base["fiscal_year"].astype(int)
    base["net_profit"] = col(w, "net_profit").to_numpy()
    base["ppop"] = (col(w, "profit_before_tax") + col(w, "impairment_charges")).to_numpy()
    base["total_assets"] = col(w, "total_assets").to_numpy()
    base["rwa_wide"] = col(w, "rwa_total").to_numpy()
    cb = cb.assign(fiscal_year=cb["fiscal_year"].astype(int))
    out = base.merge(cb[["bank_id", "fiscal_year", "rwa_total", "capital_basis"]], on=["bank_id", "fiscal_year"],
                     how="left")
    foc = out["bank_id"].isin(FOCUS)
    out["rwa"] = np.where(foc, out["rwa_total"], out["rwa_wide"])
    out["rwa_basis"] = np.where(foc, out["capital_basis"].fillna("not held"), "Bank (= Group), reported")
    out = out.drop(columns=["rwa_total", "rwa_wide", "capital_basis"]).sort_values(["bank_id", "fiscal_year"])
    out = out.reset_index(drop=True)
    prev_basis = out.groupby("bank_id")["rwa_basis"].shift(1)
    out["avg_rwa"] = (out["rwa"] + lag(out, out["rwa"])) / 2
    out["rwa_basis_mixed_in_average"] = (prev_basis.notna() & (prev_basis != out["rwa_basis"])
                                         & out["avg_rwa"].notna())
    out = out.merge(hist[["bank_id", "fiscal_year", "avg_gross_loans", "cost_of_risk"]],
                    on=["bank_id", "fiscal_year"], how="left")
    prim = ttc[ttc["definition_type"] == "primary"].set_index("bank_id")
    out["ttc_cost_of_risk"] = out["bank_id"].map(prim["ttc_mean"])
    out["ttc_coverage"] = out["bank_id"].map(prim["n_years"].astype(str) + " years; FY2015-18 covered: "
                                             + prim["covers_2015_18"].map({True: "yes", False: "NO"}))
    out["return_on_rwa"] = div(out["net_profit"], out["avg_rwa"])
    out["ppop_to_rwa"] = div(out["ppop"], out["avg_rwa"])
    out["risk_adjusted_rorwa"] = div(out["ppop"] - out["ttc_cost_of_risk"] * out["avg_gross_loans"], out["avg_rwa"])
    out["rwa_density"] = div(out["rwa"], out["total_assets"])
    out["rwa_density_basis_note"] = np.where(out["bank_id"] == "B02", "NMB: Bank-solo RWA over Group assets (D2)", "")
    out["evidence_class"] = "Derived"
    keep = out[["return_on_rwa", "rwa_density", "risk_adjusted_rorwa"]].notna().any(axis=1)
    return out[keep].reset_index(drop=True)


def panel_summary(eff: pd.DataFrame, ttc: pd.DataFrame, t: dict[str, str]) -> pd.DataFrame:
    """Seven-bank panel where RWA is held, with coverage flags."""
    rows = []
    prim = ttc[ttc["definition_type"] == "primary"].set_index("bank_id")
    for b, x in eff.groupby("bank_id"):
        r = x.dropna(subset=["return_on_rwa"])
        ra = x.dropna(subset=["risk_adjusted_rorwa"])
        rows.append({"bank_id": b, "bank": t.get(b, b), "years_with_return_on_rwa": len(r),
                     "years": f"{r['fiscal_year'].min()}-{r['fiscal_year'].max()}" if len(r) else "none",
                     "mean_return_on_rwa": r["return_on_rwa"].mean() if len(r) else np.nan,
                     "mean_return_on_rwa_2019_25": r.loc[r["fiscal_year"] >= 2019, "return_on_rwa"].mean(),
                     "years_with_risk_adjusted": len(ra),
                     "mean_risk_adjusted_rorwa": ra["risk_adjusted_rorwa"].mean() if len(ra) else np.nan,
                     "ttc_cost_of_risk": prim["ttc_mean"].get(b, np.nan),
                     "ttc_years": int(prim["n_years"].get(b, 0)),
                     "ttc_covers_2015_18": bool(prim["covers_2015_18"].get(b, False)),
                     "mean_rwa_density": x["rwa_density"].mean(),
                     "coverage_flag": "full" if b in FOCUS else (
                         "LOW: RWA held for few years" if len(r) < 5 else "PARTIAL") + (
                         "" if bool(prim["covers_2015_18"].get(b, False)) else
                         "; TTC cost of risk does not cover 2015-18 (gross loans not held)"),
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def h82(eff: pd.DataFrame, boot: dict) -> tuple[pd.DataFrame, dict]:
    """NMB - CRDB difference in risk-adjusted RoRWA by year (both on their D2/D17 RWA basis, no mixed-basis
    averages), mean and moving-block bootstrap interval over years."""
    e = eff[eff["bank_id"].isin(FOCUS) & ~eff["rwa_basis_mixed_in_average"]]
    p = e.pivot(index="fiscal_year", columns="bank_id", values=["risk_adjusted_rorwa", "return_on_rwa"]).dropna()
    d = pd.DataFrame({"fiscal_year": p.index,
                      "nmb_risk_adjusted_rorwa": p[("risk_adjusted_rorwa", "B02")].to_numpy(),
                      "crdb_risk_adjusted_rorwa": p[("risk_adjusted_rorwa", "B01")].to_numpy(),
                      "nmb_return_on_rwa": p[("return_on_rwa", "B02")].to_numpy(),
                      "crdb_return_on_rwa": p[("return_on_rwa", "B01")].to_numpy()})
    d["diff_risk_adjusted"] = d["nmb_risk_adjusted_rorwa"] - d["crdb_risk_adjusted_rorwa"]
    d["diff_unadjusted"] = d["nmb_return_on_rwa"] - d["crdb_return_on_rwa"]
    d["evidence_class"] = "Derived"
    rng = np.random.default_rng([*boot["seed"], 82])
    a = (1 - LEVEL) / 2
    res = {}
    for k in ("diff_risk_adjusted", "diff_unadjusted"):
        v = d[k].to_numpy()
        bm = _block_means(v, boot["reps"], boot["block"], rng)
        res[k] = (float(v.mean()), float(np.quantile(bm, a)), float(np.quantile(bm, 1 - a)), len(v))
    return d, res


# --- stress lens -----------------------------------------------------------------------------------------------------


def stress_lens(w: pd.DataFrame, hist: pd.DataFrame, ttc: pd.DataFrame) -> pd.DataFrame:
    """ROE today at other cost-of-risk levels, mechanically through the 4.0 tree (Scenario input for 09a): Imp/TA is
    replaced by CoR x avg gross loans / avg TA; every other component (yields, funding, fees, costs, other items,
    effective tax rate, leverage, attributable share) is held at its reported level."""
    comp = components(w)
    rows = []
    for b in FOCUS:
        c = comp[(comp["bank_id"] == b) & (comp["tree"] == "full")].set_index("fiscal_year")
        h = hist[hist["bank_id"] == b].set_index("fiscal_year")
        tb = ttc[ttc["bank_id"] == b].set_index("definition")
        st = h.loc[STRESS[0]:STRESS[1], "cost_of_risk"]
        levels = {"actual": None,
                  "TTC primary (FY2010-25 mean)": tb.loc[tb["definition_type"] == "primary", "ttc_mean"].iloc[0],
                  "2015-18 mean (own)": st.mean(),
                  "2015-18 peak year (own)": st.max()}
        ta_avg = avg(w[w["bank_id"] == b], "total_assets").set_axis(w.loc[w["bank_id"] == b, "fiscal_year"])
        for y in range(2019, CURRENT + 1):
            if y not in c.index or y not in h.index:
                continue
            x = c.loc[y].copy()
            for label, cor in levels.items():
                z = x.copy()
                if cor is not None:
                    z["imp_ta"] = cor * h.loc[y, "avg_gross_loans"] / ta_avg.get(y)
                roe = roe_full(z)
                rows.append({"bank_id": b, "fiscal_year": y, "cost_of_risk_scenario": label,
                             "cost_of_risk": h.loc[y, "cost_of_risk"] if cor is None else cor,
                             "imp_ta": z["imp_ta"], "roe_reported": x["roe_reported"], "roe_tree": x["roe_tree"],
                             "roe_scenario": roe, "roe_change_vs_actual": roe - x["roe_tree"],
                             "held_constant": "y_iea, iea_ta, c_fund, ibl_ta, nonii_ta, opex_ta, other_ta, tax_rate, "
                                              "ta_e, attr_share at reported levels",
                             "evidence_class": "Derived" if cor is None else "Scenario"})
    return pd.DataFrame(rows)


# --- note and outputs ------------------------------------------------------------------------------------------------

NOTE = """# Chapter 8: credit risk and capital efficiency

Generated by `tzbank.chapters.ch08_credit` (stage `credit`). Blueprint §2 (credit and capital efficiency) and §4
chapter 8; hypotheses H8.1, H8.2; limitation L4 (IAS 39 -> IFRS 9 in FY2018). Rules: D53 (accepted).

## Research question

How efficiently is risk capital deployed through the cycle? Is today's low cost of risk below the through-the-cycle
level, and do NMB and CRDB earn different returns on risk-weighted assets once credit cost is normalised?

## Variable definitions

- Cost of risk = impairment charge / average gross loans (blueprint §2), annual, as computed in every year.
- Through-the-cycle (TTC) cost of risk = mean of the annual cost of risk over the **primary cycle FY2010-25**
  (expansion 2010-14, stress 2015-18 including CRDB's episode, recovery 2019-25). Sensitivity: every cycle starting
  FY2010-14 and ending FY2021-25 (all include 2015-18), sub-periods, the loan-weighted (pooled) mean and the cycle
  without FY2018.
- NPL ratio = stage-3 (IFRS 9) or impaired (IAS 39) gross loans / gross loans; coverage = loss allowance / NPL.
- Return on RWA = net profit / average RWA; PPOP = profit before tax + impairment; risk-adjusted RoRWA = (PPOP - TTC
  cost of risk x average gross loans) / average RWA (pre-tax); RWA density = RWA / total assets. RWA on the D2/D17
  basis (NMB Bank-solo; CRDB Bank-solo to FY2014, Group from FY2015).
- Stress lens: the 4.0 tree ROE with Imp/TA replaced by a cost-of-risk level x average gross loans / average total
  assets, all other components held at reported levels.

## Sources and transformation trail

`data/processed/bank_year_wide.csv` (impairment, gross loans, stage-3 loans, allowance, PBT, net profit, total assets),
`data/processed/capital_long.csv` via `tzbank.chapters.ch07_capital.capital_basis` (RWA, D2/D17),
`data/processed/bank_period_long.csv` (IFRS 9 transition notes, FY2018 rows; source documents NMB-AR-FY2018,
CRDB-AR-FY2018), the 4.0 tree (`tzbank.chapters.ch04.dupont_attribution.components`). Functions: `credit_history`,
`ttc_table`, `ifrs9_transition`, `transition_ratios`, `h81`, `capital_efficiency`, `panel_summary`, `h82`,
`stress_lens`.

## Validation test

`tests/test_ch08.py`: cost-of-risk and risk-adjusted RoRWA hand cases; the primary cycle includes FY2015-18 and every
boundary-grid cycle does too; the FY2018 cost of risk equals the raw ratio (no smoothing) and excludes the transition
adjustment; transition amounts match the source notes; the stress lens at actual cost of risk reproduces the tree ROE;
bootstrap reproducibility; manifest and note sections.

## Results

### Cost of risk through the cycle (Derived)

{cor}

{ttc}

### Asset quality and the IFRS 9 break (Derived; transition amounts Observed)

{aq}

{trans}

### H8.1: current cost of risk against the through-the-cycle level (Estimated)

{h81}

### Capital efficiency (Derived)

{eff}

Seven-bank panel (where RWA is held; coverage flagged):

{panel}

### Stress lens: ROE today at the 2015-18 cost of risk (Scenario, input to 09a)

{stress}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

Cost of risk, NPL, coverage, return on RWA, RWA density: **Derived**. IFRS 9 transition amounts: **Observed**. TTC
intervals and the H8.1 gap: **Estimated**. Stress-lens ROE: **Scenario**. Verdicts: **Interpretation**.

## Limitations

- One full credit cycle in the sample; the TTC level depends on where the cycle is cut (see the boundary grid).
- IAS 39 (incurred loss) and IFRS 9 (expected loss) impairment are not the same quantity; the primary TTC spans both
  regimes and the regime means are shown separately. FY2018 values are reported as computed; the transition adjustment
  went to equity and is reported separately.
- NPL definitions change in FY2018 (impaired loans -> stage 3); coverage is not comparable across the break.
- RWA bases differ (NMB Bank-solo, D2; CRDB Group from FY2015); NMB's return on RWA divides Group profit by Bank-solo
  RWA, and its RWA density Bank-solo RWA by Group assets. The CRDB FY2015 average RWA mixes bases and is excluded from
  H8.2.
- The stress lens is mechanical: it holds margins, costs, tax and leverage fixed, which overstates the ROE effect if
  pricing or growth adjust in a downturn and understates it if NII also falls.
- The five smaller banks hold gross loans and RWA for few years; their TTC cost of risk does not cover 2015-18 and is
  not comparable (flagged).

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

A current cost of risk below the through-the-cycle level means part of today's excess return is **cyclical**, not
rent: the stress lens shows how much ROE would fall at stress-period credit costs. A risk-adjusted RoRWA gap that
survives normalising credit cost points to **pricing power or funding advantage** (chapters 4.2, 6) rather than
under-provisioning; a gap that disappears points to **risk-taking**, which is not a durable rent.
"""


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def write_credit(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    cfg = settings(root)
    pc = cfg["persistence"]
    boot = {"reps": int(pc["bootstrap_reps"]), "block": int(pc["block_length"]), "seed": [int(cfg["random_seed"]), 8]}
    w = wide(root)
    hist = credit_history(w)
    ttc = ttc_table(hist)
    tr = ifrs9_transition(root)
    trr = transition_ratios(tr, w)
    h1 = h81(hist, ttc, boot, t)
    cb = capital_basis(root)
    eff = capital_efficiency(w, cb, hist, ttc)
    panel = panel_summary(eff, ttc, t)
    d82, r82 = h82(eff, boot)
    sl = stress_lens(w, hist, ttc)
    # hypotheses
    per = {r.bank: r.verdict for r in h1[h1["current_definition"] == f"FY{CURRENT}"].itertuples()}
    v1 = "supported" if all(v == "supported" for v in per.values()) else "rejected" if all(
        v == "rejected" for v in per.values()) else "inconclusive"
    ev1 = "; ".join(f"{r.bank} {r.current_definition}: {pct(r.current_cost_of_risk, 2)} vs TTC "
                    f"{pct(r.ttc_primary, 2)} ({r.ttc_years}), gap {pct(r.gap, 2, sign=True)} [block bootstrap "
                    f"{pct(r.gap_lo_block, 2, sign=True)}, {pct(r.gap_hi_block, 2, sign=True)}; i.i.d. "
                    f"{pct(r.gap_lo_iid, 2, sign=True)}, {pct(r.gap_hi_iid, 2, sign=True)}]; current below TTC under "
                    f"{r.share_of_cycle_definitions_with_current_below_ttc:.0%} of cycle definitions"
                    for r in h1.itertuples())
    m, lo, hi, n = r82["diff_risk_adjusted"]
    mu, lou, hiu, _ = r82["diff_unadjusted"]
    v2 = "supported" if lo > 0 or hi < 0 else "inconclusive"
    hyp = pd.DataFrame([
        {"hypothesis": "H8.1", "verdict": v1, "verdict_by_bank": ", ".join(f"{k} {v}" for k, v in per.items()),
         "evidence": ev1,
         "rule": "per bank: supported when the 90% moving-block bootstrap interval of (FY2025 cost of risk - primary TTC "
                 "mean) lies below zero; rejected when above; joint verdict needs both banks. FY2023-25 mean shown as "
                 "robustness"},
        {"hypothesis": "H8.2", "verdict": v2, "verdict_by_bank": "joint (NMB - CRDB)",
         "evidence": f"NMB - CRDB risk-adjusted RoRWA (pre-tax, own TTC cost of risk), mean over {n} years "
                     f"(FY{int(d82['fiscal_year'].min())}-{int(d82['fiscal_year'].max())}): {pct(m, 2, sign=True)} "
                     f"[{pct(lo, 2, sign=True)}, {pct(hi, 2, sign=True)}]; unadjusted return on RWA (after tax) "
                     f"{pct(mu, 2, sign=True)} [{pct(lou, 2, sign=True)}, {pct(hiu, 2, sign=True)}]",
         "rule": "supported when the 90% moving-block bootstrap interval (block 3, over years) of the mean NMB - CRDB "
                 "difference in risk-adjusted RoRWA excludes zero; inconclusive otherwise"}])
    hyp["evidence_class"] = "Interpretation"
    tab, figd = path("tables", root), path("figures", root)
    tables = [(hist, "ch08_credit_history.csv", "Cost of risk, NPL and coverage by bank-year, regime marked",
               "Derived"),
              (ttc, "ch08_ttc_cost_of_risk.csv", "Through-the-cycle cost of risk: primary cycle, boundary grid, "
                                                 "sub-periods", "Derived"),
              (tr, "ch08_ifrs9_transition.csv", "FY2018 IFRS 9 transition adjustment (opening equity, allowance)",
               "Observed"),
              (trr, "ch08_ifrs9_transition_ratios.csv", "Transition adjustment scaled by equity and loans (memo)",
               "Derived"),
              (h1, "ch08_h81_cost_of_risk_gap.csv", "H8.1: current vs TTC cost of risk with bootstrap intervals",
               "Estimated"),
              (eff, "ch08_capital_efficiency.csv", "Return on RWA, PPOP / RWA, risk-adjusted RoRWA, RWA density",
               "Derived"),
              (panel, "ch08_panel_summary.csv", "Seven-bank panel with coverage flags", "Derived"),
              (d82, "ch08_h82_rorwa_difference.csv", "H8.2: NMB - CRDB risk-adjusted RoRWA by year", "Derived"),
              (sl, "ch08_stress_lens.csv", "ROE at TTC and 2015-18 cost of risk through the 4.0 tree (09a input)",
               "Scenario"),
              (hyp, "ch08_hypotheses.csv", "H8.1-H8.2 verdicts", "Interpretation")]
    files = [write_csv(df, tab / name, require_evidence=True) for df, name, _, _ in tables]
    figs = []
    # figure 1: cost of risk with the IFRS 9 break (segments drawn separately) and TTC levels
    fig, ax = plt_subplots(figsize=(8, 4.4))
    prim = ttc[ttc["definition_type"] == "primary"].set_index("bank_id")
    for b in FOCUS:
        x = hist[hist["bank_id"] == b].dropna(subset=["cost_of_risk"])
        for reg, ls in (("IAS 39", "-"), ("IFRS 9", "-")):
            s = x[x["accounting_regime"] == reg]
            ax.plot(s["fiscal_year"], s["cost_of_risk"] * 100, color=COLORS[b], lw=2, ls=ls, marker="o", ms=3,
                    label=f"{t[b]} cost of risk ({reg})")
        ax.axhline(prim.loc[b, "ttc_mean"] * 100, color=COLORS[b], lw=1, ls="--", label=f"{t[b]} TTC FY2010-25")
    ax.axvspan(STRESS[0] - 0.5, STRESS[1] + 0.5, color="#ecebe6", zorder=0)
    ax.axvline(IFRS9_YEAR - 0.5, color="#5a5a55", lw=0.8, ls=":")
    ax.text(IFRS9_YEAR - 0.4, ax.get_ylim()[1] * 0.95, "IFRS 9", fontsize=7, color="#5a5a55")
    style(ax, "Cost of risk, IAS 39 and IFRS 9 segments, against the through-the-cycle level", "percent of loans")
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch08_cost_of_risk.png", "Evidence class: Derived. Impairment / average gross loans; shaded: "
               "2015-18 stress; the IFRS 9 break (FY2018) is not smoothed: segments are drawn separately.")
    data = write_csv(hist[hist["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "cost_of_risk", "accounting_regime",
                                                        "evidence_class"]], figd / "ch08_cost_of_risk_data.csv",
                     require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Cost of risk with the IFRS 9 break and TTC levels",
                 "evidence_class": "Derived"})
    # figure 2: NPL and coverage
    fig, (a1, a2) = plt_subplots(1, 2, figsize=(9, 4))
    for b in FOCUS:
        x = hist[hist["bank_id"] == b]
        for reg in ("IAS 39", "IFRS 9"):
            s = x[x["accounting_regime"] == reg]
            a1.plot(s["fiscal_year"], s["npl_ratio"] * 100, color=COLORS[b], lw=2, marker="o", ms=3,
                    label=f"{t[b]}" if reg == "IFRS 9" else None)
            a2.plot(s["fiscal_year"], s["coverage"] * 100, color=COLORS[b], lw=2, marker="o", ms=3)
    for a in (a1, a2):
        a.axvline(IFRS9_YEAR - 0.5, color="#5a5a55", lw=0.8, ls=":")
    style(a1, "NPL ratio (impaired / stage 3)", "percent")
    style(a2, "Coverage (allowance / NPL)", "percent")
    a1.legend(fontsize=7, frameon=False)
    png = save(fig, figd / "ch08_asset_quality.png", "Evidence class: Derived. Dotted line: IFRS 9 from FY2018 (NPL "
               "definition changes; segments not joined).")
    data = write_csv(hist[hist["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "npl_ratio", "coverage",
                                                        "accounting_regime", "evidence_class"]],
                     figd / "ch08_asset_quality_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "NPL ratio and coverage", "evidence_class": "Derived"})
    # figure 3: return on RWA and risk-adjusted
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in FOCUS:
        x = eff[(eff["bank_id"] == b) & ~eff["rwa_basis_mixed_in_average"]]
        ax.plot(x["fiscal_year"], x["return_on_rwa"] * 100, color=COLORS[b], lw=2, label=f"{t[b]} return on RWA")
        ax.plot(x["fiscal_year"], x["risk_adjusted_rorwa"] * 100, color=COLORS[b], lw=1.2, ls="--",
                label=f"{t[b]} risk-adjusted (pre-tax, TTC CoR)")
    ax.axvline(IFRS9_YEAR - 0.5, color="#5a5a55", lw=0.8, ls=":")
    style(ax, "Return on RWA and risk-adjusted return on RWA", "percent of average RWA")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch08_rorwa.png", "Evidence class: Derived. RWA on the D2/D17 basis; CRDB FY2015 (mixed-basis "
               "average) omitted.")
    data = write_csv(eff[eff["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "return_on_rwa", "risk_adjusted_rorwa",
                                                      "rwa_density", "evidence_class"]],
                     figd / "ch08_rorwa_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Return on RWA and risk-adjusted RoRWA",
                 "evidence_class": "Derived"})
    files += [f["path"] for f in figs] + [f["data"] for f in figs]
    # --- note text ---
    hf = hist[hist["bank_id"].isin(FOCUS)].copy()
    hf["bank"] = hf["bank_id"].map(t)
    pv = hf.pivot(index="fiscal_year", columns="bank", values="cost_of_risk").sort_index()
    rg = hf.drop_duplicates("fiscal_year").set_index("fiscal_year")["accounting_regime"]
    cor_md = _md(pd.DataFrame({"FY": pv.index, "Regime": rg.reindex(pv.index).to_numpy(),
                               **{b: [pct(v, 2) for v in pv[b]] for b in pv.columns}}))
    tp = ttc[ttc["bank_id"].isin(FOCUS)].copy()
    tl = []
    for b in FOCUS:
        x = tp[tp["bank_id"] == b].set_index("definition")
        g = tp[(tp["bank_id"] == b) & tp["definition_type"].isin(["primary", "boundary grid"])]
        p = x[x["definition_type"] == "primary"].iloc[0]
        tl.append(f"- {t[b]}: TTC FY2010-25 {pct(p['ttc_mean'], 2)} (loan-weighted {pct(p['ttc_pooled'], 2)}; "
                  f"{int(p['n_years'])} years, missing: {p['years_missing']}); across the 25 boundary cycles "
                  f"{pct(g['ttc_mean'].min(), 2)} to {pct(g['ttc_mean'].max(), 2)}; stress 2015-18 "
                  f"{pct(x.loc['stress only FY2015-18', 'ttc_mean'], 2)}; IAS 39 regime "
                  f"{pct(x.loc['IAS 39 regime FY2010-17', 'ttc_mean'], 2)} vs IFRS 9 regime "
                  f"{pct(x.loc['IFRS 9 regime FY2018-25', 'ttc_mean'], 2)}; excluding FY2018 "
                  f"{pct(x.loc['primary cycle excluding FY2018', 'ttc_mean'], 2)}.")
    ttc_txt = ("**Through-the-cycle definition.** Primary cycle FY2010-25 (16 annual observations, simple mean of the "
               "annual ratios); it contains one full cycle with the 2015-18 stress. Boundary sensitivity: starts "
               "FY2010-14 x ends FY2021-25 (25 cycles, all containing 2015-18). `ch08_ttc_cost_of_risk.csv`. "
               "**Accounting span (D53 accepted):** the FY2010-25 average mixes 8 IAS 39 years (FY2010-17, incurred "
               "loss) and 8 IFRS 9 years (FY2018-25, expected loss); the regime means are shown below and the FY2018 "
               "transition adjustment is outside it.\n\n"
               + "\n".join(tl))
    aq = []
    for b in FOCUS:
        x = hist[hist["bank_id"] == b].set_index("fiscal_year")
        pk = x.loc[x.index <= IFRS9_YEAR - 1, "npl_ratio"]
        aq.append(f"- {t[b]}: NPL ratio peak (IAS 39) {pct(pk.max())} in FY{int(pk.idxmax())}; FY2017 "
                  f"{pct(x['npl_ratio'].get(2017))} -> FY2018 {pct(x['npl_ratio'].get(2018))} (first IFRS 9 year, "
                  f"stage 3) -> FY{CURRENT} {pct(x['npl_ratio'].get(CURRENT))}; coverage FY2017 "
                  f"{pct(x['coverage'].get(2017))} -> FY2018 {pct(x['coverage'].get(2018))} -> FY{CURRENT} "
                  f"{pct(x['coverage'].get(CURRENT))}.")
    tx = []
    for b in tr["bank_id"].unique() if len(tr) else []:
        o = tr[tr["bank_id"] == b].set_index("item")
        rr = trr[trr["bank_id"] == b].set_index("item")["value"]
        vals = o["value"]

        def g(k: str, v: pd.Series = vals) -> float:
            return v.get(k, np.nan)
        tx.append(f"- {t[b]} (1 Jan 2018, Group, {o['source_document'].iloc[0]}, {o['source_location'].iloc[0]}): "
                  f"retained-earnings adjustment TZS {g('ifrs9_adjustment_retained_earnings') / 1e9:,.1f}bn"
                  + (f", fair-value reserve +{g('fair_value_reserve_adjustment') / 1e9:,.1f}bn"
                     if pd.notna(g("fair_value_reserve_adjustment")) else "")
                  + f"; net equity impact TZS {g('equity_impact_net') / 1e9:,.1f}bn "
                    f"({pct(rr.get('equity_impact_pct_of_fy2017_equity'), 2)} of FY2017 equity)"
                  + (f"; loan allowance {g('allowance_ias39_close') / 1e9:,.1f}bn -> "
                     f"{g('allowance_ifrs9_open') / 1e9:,.1f}bn (+{g('allowance_increase') / 1e9:,.1f}bn; memo: "
                     f"{pct(rr.get('memo_allowance_increase_pct_of_fy2018_avg_loans'), 2)} of FY2018 average loans, "
                     "not added to the FY2018 cost of risk)" if pd.notna(g("allowance_increase")) else
                     f"; loan allowance at 31 Dec 2017 TZS {g('allowance_ias39_close') / 1e9:,.1f}bn (opening IFRS 9 "
                     "allowance not in the extracted note)") + ".")
    trans_txt = ("**FY2018 IFRS 9 transition (reported separately; the break is not smoothed).**\n\n" + "\n".join(tx)
                 + "\n\nThe adjustment went to opening equity, so it never appears in impairment or cost of risk; "
                   "FY2018 cost of risk is the first expected-loss year as reported.")
    h81_md = _md(pd.DataFrame({
        "Bank": h1["bank"], "Current": h1["current_definition"], "Cost of risk": [pct(v, 2) for v in
                                                                                   h1["current_cost_of_risk"]],
        "TTC FY2010-25": [pct(v, 2) for v in h1["ttc_primary"]], "Gap": [pct(v, 2, sign=True) for v in h1["gap"]],
        "90% block bootstrap": [f"[{pct(a, 2, sign=True)}, {pct(b_, 2, sign=True)}]"
                                for a, b_ in zip(h1["gap_lo_block"], h1["gap_hi_block"], strict=True)],
        "90% i.i.d.": [f"[{pct(a, 2, sign=True)}, {pct(b_, 2, sign=True)}]"
                       for a, b_ in zip(h1["gap_lo_iid"], h1["gap_hi_iid"], strict=True)],
        "Below TTC (share of cycles)": [f"{v:.0%}" for v in h1["share_of_cycle_definitions_with_current_below_ttc"]],
        "Verdict": h1["verdict"]}))
    el = []
    for b in FOCUS:
        x = eff[(eff["bank_id"] == b) & ~eff["rwa_basis_mixed_in_average"]].set_index("fiscal_year")
        dens = eff[eff["bank_id"] == b].set_index("fiscal_year")["rwa_density"]  # point-in-time: no averaging
        el.append(f"- {t[b]}: return on RWA FY{CURRENT} {pct(x['return_on_rwa'].get(CURRENT), 2)} (mean 2019-25 "
                  f"{pct(x.loc[x.index >= 2019, 'return_on_rwa'].mean(), 2)}); risk-adjusted RoRWA FY{CURRENT} "
                  f"{pct(x['risk_adjusted_rorwa'].get(CURRENT), 2)} (pre-tax, at TTC cost of risk "
                  f"{pct(x['ttc_cost_of_risk'].iloc[-1], 2)}); PPOP / RWA {pct(x['ppop_to_rwa'].get(CURRENT), 2)}; RWA "
                  f"density FY2015 {pct(dens.get(2015))} -> FY{CURRENT} "
                  f"{pct(dens.get(CURRENT))}" + (" (Bank-solo RWA over Group assets)" if b == "B02" else
                                                             "") + ".")
    pm = _md(pd.DataFrame({
        "Bank": panel["bank"], "RoRWA years": panel["years_with_return_on_rwa"], "Span": panel["years"],
        "Mean RoRWA": [pct(v, 2) for v in panel["mean_return_on_rwa"]],
        "Mean risk-adj. RoRWA": [pct(v, 2, reason="TTC not computable") for v in panel["mean_risk_adjusted_rorwa"]],
        "TTC CoR": [pct(v, 2, reason="gross loans not held") for v in panel["ttc_cost_of_risk"]],
        "TTC years": panel["ttc_years"], "Coverage": panel["coverage_flag"]}))
    sf = sl[sl["fiscal_year"] == CURRENT].copy()
    sf["bank"] = sf["bank_id"].map(t)
    sl_md = _md(pd.DataFrame({
        "Bank": sf["bank"], "Cost of risk": sf["cost_of_risk_scenario"], "Level": [pct(v, 2) for v in
                                                                                    sf["cost_of_risk"]],
        f"ROE FY{CURRENT}": [pct(v) for v in sf["roe_scenario"]],
        "Change vs actual": [pct(v, sign=True) for v in sf["roe_change_vs_actual"]]}))
    sl_md += ("\n\nAll other 4.0 components held at their reported FY2025 levels (effective tax rate, leverage and "
              "attributable share included). Years FY2019-25 in `ch08_stress_lens.csv`.")
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}** ({r.verdict_by_bank}). {r.evidence}. Rule: {r.rule}."
                        for r in hyp.itertuples())
    s25 = sf.set_index(["bank_id", "cost_of_risk_scenario"])["roe_change_vs_actual"]
    interp = ("**Interpretation.** Both banks run today below their through-the-cycle cost of "
              "risk (see H8.1 for whether the gap is distinguishable from cycle noise), so part of current ROE is "
              f"cyclical: at their own 2015-18 average cost of risk, FY{CURRENT} ROE would change by "
              f"{pct(s25.get(('B01', '2015-18 mean (own)')), sign=True)} (CRDB) and "
              f"{pct(s25.get(('B02', '2015-18 mean (own)')), sign=True)} (NMB), holding everything else fixed. "
              "The risk-adjusted return on RWA compares the banks once credit cost is normalised; whether a gap "
              "survives (H8.2) separates franchise economics from risk appetite. The IFRS 9 break limits how far the "
              "IAS 39 years can be read as a guide to future expected-loss charges.")
    text = NOTE.format(cor=cor_md, ttc=ttc_txt, aq="\n".join(aq), trans=trans_txt, h81=h81_md, eff="\n".join(el),
                       panel=pm, stress=sl_md, hyp=hyp_txt, interp=interp)
    nf = write_text(text, path("notes", root) / "ch08.md")
    files.append(nf)
    m = {"chapter": "08", "title": "Credit risk and capital efficiency", "hypotheses": ["H8.1", "H8.2"],
         "status": "built",
         "tables": [{"path": rel(tab / name, root), "description": d, "evidence_class": c_, "note": rel(nf, root)}
                    for _, name, d, c_ in tables],
         "figures": [{"path": rel(f["path"], root), "data": rel(f["data"], root), "description": f["description"],
                      "evidence_class": f["evidence_class"], "note": rel(nf, root)} for f in figs],
         "notes": [rel(nf, root)],
         "inputs": ["data/processed/bank_year_wide.csv", "data/processed/capital_long.csv",
                    "data/processed/bank_period_long.csv"],
         "module": "tzbank.chapters.ch08_credit", "run_log_stage": "credit"}
    files.append(write_json(m, path("manifests", root) / "ch08.json"))
    return files


__all__ = ["credit_history", "ttc_table", "ifrs9_transition", "h81", "capital_efficiency", "h82", "stress_lens",
           "write_credit"]
