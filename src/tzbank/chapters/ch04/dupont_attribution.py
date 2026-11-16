"""Chapter 4.0: extended DuPont tree and ROE-gap attribution (blueprint §4, 4.0; hypotheses H4.1, H4.2).

All terms use average balances (opening + closing) / 2. IEA = interest-earning assets; IBL = interest-bearing
liabilities, measured as customer deposits + borrowings (the lines held for every year; deposits from banks and lease
liabilities are not held separately, so their interest sits in c_fund over this base):

    ROE_attr = [ IEA/TA * y_IEA - IBL/TA * c_fund + NonII/TA - OpEx/TA - Imp/TA + Other/TA ] * (1 - t) * TA/E * NP_attr/NP

with y_IEA = interest income / IEA, c_fund = interest expense / IBL, t = 1 - NP/PBT (effective) and Other = PBT -
(II - IE + NonII - OpEx - Imp), the reconciling items (associates, one-offs), so the tree reconstructs the reported ROE
exactly. E is average equity attributable to shareholders (NMB, CRDB); for B03-B07 a reduced tree (NII/TA in place of
the yield and funding terms, total equity, attributable share 1) is used and flagged.

Attribution: an exact Shapley decomposition (all 2^k orderings of the k components) of (1) the NMB - CRDB ROE gap in
each year and (2) each bank's change between periods (2010-14 -> 2015-18 stress -> 2019-25, and 2019 -> 2025),
evaluated on period-mean components. Contributions are grouped by the subsection that explains them (4.1-4.7), which
ranks the subsections; 4.7 must agree with this ranking.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, FOCUS, avg, col, div, manifest, note, pct, plt_subplots, save, style, tickers, wide
from tzbank.config import find_root, path
from tzbank.io import write_csv

FULL = ["y_iea", "iea_ta", "c_fund", "ibl_ta", "nonii_ta", "opex_ta", "imp_ta", "other_ta", "tax_rate", "ta_e",
        "attr_share"]
REDUCED = ["nii_ta", "nonii_ta", "opex_ta", "imp_ta", "other_ta", "tax_rate", "ta_e", "attr_share"]
SUBSECTION = {"y_iea": "4.3 Asset deployment (asset yield)", "iea_ta": "4.3 Asset deployment (earning-asset share)",
              "c_fund": "4.2 Funding economics (cost of funds)", "ibl_ta": "4.2 Funding economics (funding mix)",
              "nii_ta": "4.2/4.3 Net interest margin (reduced tree)",
              "nonii_ta": "4.1/4.4 Revenue and transaction economics (non-interest income)",
              "opex_ta": "4.5 Operating leverage (operating costs)", "imp_ta": "4.6 Risk economics (impairment)",
              "other_ta": "4.7 Other items (reconciling)", "tax_rate": "4.7 Tax", "ta_e": "4.7 Leverage (capital, ch. 7)",
              "attr_share": "4.7 Minority interests (attributable share)"}
# Ranking codes: the capital-structure block is split into its own terms (03b review fix 1) so that leverage, which
# chapter 7 analyses on its own, is isolated from tax, the NCI share and reconciling items.
SUBCODE = {"y_iea": "4.3", "iea_ta": "4.3", "c_fund": "4.2", "ibl_ta": "4.2", "nii_ta": "4.2/4.3",
           "nonii_ta": "4.1/4.4", "opex_ta": "4.5", "imp_ta": "4.6", "other_ta": "4.7 other items",
           "tax_rate": "4.7 tax (1-t)", "ta_e": "4.7 leverage (TA/E)", "attr_share": "4.7 NCI share"}
PERIODS = {"2010-14": range(2010, 2015), "2015-18 (stress)": range(2015, 2019), "2019-25": range(2019, 2026)}


def components(w: pd.DataFrame) -> pd.DataFrame:
    """Every bank-year's tree components, the reconstructed ROE and the tree type."""
    ta = avg(w, "total_assets")
    eq_attr, eq_tot = avg(w, "equity_attributable"), avg(w, "total_equity")
    ii, ie = col(w, "interest_income"), col(w, "interest_expense").abs()
    nonii, opex, imp = col(w, "non_interest_income"), col(w, "operating_expenses").abs(), col(
        w, "impairment_charges").abs()
    pbt, np_, npa = col(w, "profit_before_tax"), col(w, "net_profit"), col(w, "profit_attributable")
    iea = avg(w, "interest_earning_assets")
    ibl = avg(w.assign(_ibl=col(w, "customer_deposits") + col(w, "borrowings").fillna(0)), "_ibl")
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"]})
    attr = npa.notna() & eq_attr.notna()
    full = attr & iea.notna() & ibl.notna()
    e = eq_attr.where(attr, eq_tot)
    out["tree"] = np.where(full, "full", "reduced (no IEA / cost of funds / NCI split)")
    out["y_iea"], out["iea_ta"] = div(ii, iea), div(iea, ta)
    out["c_fund"], out["ibl_ta"] = div(ie, ibl), div(ibl, ta)
    out["nii_ta"] = div(ii - ie, ta)
    out["nonii_ta"], out["opex_ta"], out["imp_ta"] = div(nonii, ta), div(opex, ta), div(imp, ta)
    out["other_ta"] = div(pbt - (ii - ie + nonii - opex - imp), ta)
    out["tax_rate"] = 1 - div(np_, pbt)
    out["ta_e"] = div(ta, e)
    out["attr_share"] = div(npa, np_).where(attr, 1.0)
    out["roe_reported"] = div(npa.where(attr, np_), e)
    out["roe_tree"] = [roe_full(r) if t == "full" else roe_reduced(r) for t, (_, r) in zip(out["tree"], out.iterrows(),
                                                                                         strict=True)]
    out["reconstruction_error"] = out["roe_tree"] - out["roe_reported"]
    out.loc[out["roe_tree"].isna(), "tree"] = "not computable (a component is missing)"
    out["evidence_class"] = "Derived"
    return out[out["roe_reported"].notna()].reset_index(drop=True)


def roe_full(x) -> float:
    pre = x["iea_ta"] * x["y_iea"] - x["ibl_ta"] * x["c_fund"] + x["nonii_ta"] - x["opex_ta"] - x["imp_ta"] + x[
        "other_ta"]
    return pre * (1 - x["tax_rate"]) * x["ta_e"] * x["attr_share"]


def roe_reduced(x) -> float:
    pre = x["nii_ta"] + x["nonii_ta"] - x["opex_ta"] - x["imp_ta"] + x["other_ta"]
    return pre * (1 - x["tax_rate"]) * x["ta_e"] * x["attr_share"]


def shapley(f, base: dict, target: dict, names: list[str]) -> dict[str, float]:
    """Exact Shapley values of moving ``names`` from ``base`` to ``target``; they sum to f(target) - f(base)."""
    k = len(names)
    w = {s: factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)}
    cache: dict[frozenset, float] = {}

    def v(sub: frozenset) -> float:
        if sub not in cache:
            cache[sub] = f({n: (target[n] if n in sub else base[n]) for n in names})
        return cache[sub]

    out = {}
    for n in names:
        others = [m for m in names if m != n]
        tot = 0.0
        for s in range(k):
            for c in combinations(others, s):
                fs = frozenset(c)
                tot += w[s] * (v(fs | {n}) - v(fs))
        out[n] = tot
    return out


def gap_attribution(comp: pd.DataFrame) -> pd.DataFrame:
    """NMB - CRDB ROE gap in each year with both full trees, Shapley by component."""
    a = comp[(comp["bank_id"] == "B02") & (comp["tree"] == "full")].set_index("fiscal_year")
    b = comp[(comp["bank_id"] == "B01") & (comp["tree"] == "full")].set_index("fiscal_year")
    rows = []
    for y in sorted(set(a.index) & set(b.index)):
        sv = shapley(roe_full, b.loc[y, FULL].to_dict(), a.loc[y, FULL].to_dict(), FULL)
        gap = a.loc[y, "roe_tree"] - b.loc[y, "roe_tree"]
        for n, v in sv.items():
            rows.append({"comparison": "NMB - CRDB", "fiscal_year": y, "component": n, "subsection": SUBSECTION[n],
                         "contribution": v, "gap": gap, "nmb_value": a.loc[y, n], "crdb_value": b.loc[y, n]})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


def change_attribution(comp: pd.DataFrame) -> pd.DataFrame:
    """Each bank's change between periods (period-mean components) and 2019 -> 2025. NMB and CRDB get the full tree
    on the years where it is complete (coverage recorded) and the reduced tree on all years; B03-B07 the reduced
    tree only."""
    rows = []
    for bank, g0 in comp.groupby("bank_id"):
        variants = [("reduced", REDUCED, roe_reduced, g0)]
        if bank in FOCUS:
            variants.insert(0, ("full", FULL, roe_full, g0[g0["tree"] == "full"]))
        for tree, names, f, g in variants:
            g = g.set_index("fiscal_year")
            means = {p: g.loc[g.index.isin(list(yrs)), names].mean() for p, yrs in PERIODS.items()}
            cover = {p: int(g.index.isin(list(yrs)).sum()) for p, yrs in PERIODS.items()}
            steps = [(f"{x} -> {y}", means[x], means[y], f"{cover[x]}/{len(PERIODS[x])} and {cover[y]}/"
                      f"{len(PERIODS[y])} years") for x, y in [("2010-14", "2015-18 (stress)"),
                                                              ("2015-18 (stress)", "2019-25")]]
            if 2019 in g.index and 2025 in g.index:
                steps.append(("2019 -> 2025", g.loc[2019, names], g.loc[2025, names], "point years"))
            for label, x0, x1, cov in steps:
                if x0.isna().any() or x1.isna().any():
                    continue
                sv = shapley(f, x0.to_dict(), x1.to_dict(), names)
                change = f(x1.to_dict()) - f(x0.to_dict())
                for n, v in sv.items():
                    rows.append({"bank_id": bank, "comparison": label, "tree": tree, "component": n,
                                 "subsection": SUBSECTION[n], "contribution": v, "roe_change": change,
                                 "coverage": cov,
                                 "flag": "" if tree == "full" else "reduced tree (NII/TA; no yield-funding split)"})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


def by_subsection(att: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    s = att.assign(sub=att["component"].map(SUBCODE))
    g = s.groupby([*keys, "sub"], as_index=False)["contribution"].sum()
    g["rank"] = g.groupby(keys)["contribution"].transform(lambda x: x.abs().rank(ascending=False, method="first"))
    g["evidence_class"] = "Derived"
    return g.rename(columns={"sub": "subsection"})


PLAIN = {"y_iea": "asset yield", "iea_ta": "earning-asset share", "c_fund": "cost of funds", "ibl_ta": "funding mix",
         "nonii_ta": "non-interest income", "opex_ta": "operating costs", "imp_ta": "impairment",
         "other_ta": "other items", "tax_rate": "tax", "ta_e": "leverage", "attr_share": "minority interests"}


def plain_label(component: str, subsection: str | None = None) -> str:
    """Plain-language axis label for a DuPont component, with its subsection number (figure fix, prompt 15c)."""
    lab = PLAIN.get(component, component)
    num = subsection.split(" ")[0] if subsection else ""
    return f"{lab} ({num})" if num else lab


def waterfall(ax, labels: list[str], values: list[float], total_label: str) -> None:
    """Shapley waterfall in percentage points; the y-axis spans the whole running path (no clipped bars)."""
    assert len(labels) == len(values)
    run, path_ = 0.0, [0.0]
    for i, v in enumerate(values):
        ax.bar(i, v * 100, bottom=run * 100, color="#1baf7a" if v >= 0 else "#d1453b", width=0.6)
        ax.annotate(f"{v * 100:+.1f}", (i, (run + max(v, 0)) * 100), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6, color="#1f1f1d")
        run += v
        path_.append(run)
    ax.bar(len(values), run * 100, color="#5a5a55", width=0.6)
    ax.annotate(f"{run * 100:+.1f}", (len(values), max(run, 0) * 100), xytext=(0, 2), textcoords="offset points",
                ha="center", fontsize=6, color="#1f1f1d")
    lo, hi = min(path_) * 100, max(path_) * 100
    pad = 0.08 * (hi - lo or 1)
    ax.set_ylim(lo - pad, hi + 2 * pad)
    ax.set_xticks(range(len(values) + 1), [*labels, total_label], rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="#8a8a85", lw=0.8)


NOTE = """# Chapter 4.0: extended DuPont and ROE-gap attribution

Generated by `tzbank.chapters.ch04.dupont_attribution` (stage `profitability`). Blueprint §4, 4.0; hypotheses H4.1
(funding advantage drives excess return) and H4.2 (different profit engines). Decision D11 (amended).

## Research question

Which components of ROE explain each bank's level, its changes between periods, and the NMB - CRDB gap? The
attribution ranks the subsections 4.1-4.7; every claim in 4.7 must agree with it.

## Variable definitions

Average balances throughout. IEA = interest-earning assets (inputs workbook); IBL = customer deposits + borrowings
(wholesale funding); y_IEA = interest income / IEA; c_fund = interest expense / IBL; NonII, OpEx, Imp over average
total assets; Other = PBT - (NII + NonII - OpEx - Imp), the reconciling items; t = 1 - NP/PBT; TA/E with E = average
equity attributable; NP_attr/NP the attributable share. Formula in `docs/RESEARCH_BLUEPRINT.md` §4 (4.0).

## Sources and transformation trail

`data/processed/bank_year_wide.csv` (original vintage, the bank's own entity); functions `components`, `shapley`,
`gap_attribution`, `change_attribution`. Tables: `outputs/tables/ch04_4_0_components.csv`,
`ch04_4_0_gap_attribution.csv`, `ch04_4_0_change_attribution.csv`, `ch04_4_0_subsection_ranking.csv`.

## Validation test

`tests/test_ch04_03a.py`: the tree reconstructs reported ROE exactly (|error| < 1e-12) for every bank-year; Shapley
contributions sum to each gap and change; hand-checked two-component Shapley case.

## Results (Derived)

{results}

## Evidence class

Tables and figures: **Derived** (computed from Observed workbook and BoT data); the sector book is
**Observed**. Statements under *Draft interpretation* are **Interpretation**.

## Limitations

- IBL excludes deposits from banks and lease liabilities (not held separately for all years), so c_fund is interest
  expense over a slightly narrow base; the product IBL/TA x c_fund is exact (= IE/TA), the split between rate and mix
  is approximate.
- Period comparisons use period-mean components; the ROE of mean components differs from the mean ROE.
- Shapley values attribute interaction terms equally; they describe, not cause (blueprint D12).
- B03-B07 use the reduced tree (flagged): their NII cannot be split into yield and funding cost.
- Accounting breaks (IFRS 9 in FY2018, CRDB's FY2024 Islamic-income presentation) move components between lines.

## Interpretation

{interpretation}

## Cross-cutting explanations of rent (blueprint §0)

{explanations}
"""


def _results(comp, gap, chg, rank_gap, rank_chg, t) -> tuple[str, str, str]:
    g25 = gap[gap["fiscal_year"] == gap["fiscal_year"].max()]
    y = int(g25["fiscal_year"].iloc[0]) if len(g25) else None
    lines = []
    if y:
        top = g25.reindex(g25["contribution"].abs().sort_values(ascending=False).index).head(4)
        lines.append(f"- NMB - CRDB ROE gap in FY{y}: {pct(g25['gap'].iloc[0], 1, sign=True)}. Largest contributors: "
                     + "; ".join(f"{r.component} ({r.subsection}) {pct(r.contribution, 1, sign=True)}" for r in top.itertuples()))
    avg_gap = rank_gap.groupby("subsection")["contribution"].mean().sort_values(key=abs, ascending=False)
    lines.append("- Mean contribution to the NMB - CRDB gap across years, by subsection: "
                 + "; ".join(f"{k} {pct(v, 1, sign=True)}" for k, v in avg_gap.items()))
    for bank in FOCUS:
        for comp_label in ("2015-18 (stress) -> 2019-25", "2019 -> 2025"):
            s = chg[(chg["bank_id"] == bank) & (chg["comparison"] == comp_label) & (chg["tree"] == "full")]
            if s.empty:
                continue
            top = s.reindex(s["contribution"].abs().sort_values(ascending=False).index).head(3)
            lines.append(f"- {t[bank]} {comp_label} (full tree, {s['coverage'].iloc[0]}): ROE change "
                         f"{pct(s['roe_change'].iloc[0], 1, sign=True)}; largest: "
                         + "; ".join(f"{r.component} {pct(r.contribution, 1, sign=True)}" for r in top.itertuples()))
    err = comp["reconstruction_error"].abs().max()
    lines.append(f"- Reconstruction: max |tree ROE - reported ROE| = {err:.1e} over {len(comp)} bank-years "
                 f"({(comp['tree'] == 'full').sum()} full trees).")
    interp = ("**Interpretation.** The subsection ranking above, not the level of any single "
              "ratio, is the evidence for H4.1 and H4.2. If the funding terms (4.2) dominate the gap and the recovery, "
              "H4.1 is supported; if the leading contributors differ between the banks, H4.2 is supported. The "
              "ranking is descriptive: it says where the ROE arithmetic differs, not why.")
    expl = ("Supports or contradicts (to be confirmed in 4.2-4.6): **low-cost deposits** (c_fund, ibl_ta); **scale "
            "economies** (opex_ta); **temporary credit-cycle conditions** (imp_ta, the 2015-18 stress period); "
            "**distribution and market access** (nonii_ta). The attribution cannot speak to switching costs, limited "
            "competition or regulatory advantage, which need chapters 1, 6 and 8.")
    return "\n".join(lines), interp, expl


def write_dupont_attribution(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    comp = components(wide(root))
    gap = gap_attribution(comp)
    chg = change_attribution(comp)
    rank_gap = by_subsection(gap, ["fiscal_year"])
    rank_chg = by_subsection(chg, ["bank_id", "comparison", "tree"])
    tab = path("tables", root)
    files = [write_csv(comp, tab / "ch04_4_0_components.csv", require_evidence=True),
             write_csv(gap, tab / "ch04_4_0_gap_attribution.csv", require_evidence=True),
             write_csv(chg, tab / "ch04_4_0_change_attribution.csv", require_evidence=True),
             write_csv(pd.concat([rank_gap.assign(comparison="NMB - CRDB"), rank_chg], ignore_index=True),
                       tab / "ch04_4_0_subsection_ranking.csv", require_evidence=True)]
    figs = []
    fig_dir = path("figures", root)
    if len(gap):
        y = int(gap["fiscal_year"].max())
        g = gap[gap["fiscal_year"] == y]
        fig, ax = plt_subplots(figsize=(8, 4.4))
        waterfall(ax, [plain_label(c, s) for s, c in zip(g["subsection"], g["component"], strict=True)],
                  list(g["contribution"]), "NMB - CRDB gap")
        style(ax, f"NMB - CRDB ROE gap, FY{y}: Shapley contributions by subsection", "percentage points")
        png = save(fig, fig_dir / "ch04_4_0_gap_waterfall.png", "Evidence class: Derived. Extended DuPont tree on "
                   "average balances; exact Shapley decomposition (tzbank.chapters.ch04.dupont_attribution).")
        data = write_csv(g, fig_dir / "ch04_4_0_gap_waterfall_data.csv", require_evidence=True)
        figs.append({"path": png, "data": data, "description": f"NMB - CRDB ROE gap FY{y}, by subsection",
                     "evidence_class": "Derived"})
        files += [png, data]
    for bank in FOCUS:
        s = chg[(chg["bank_id"] == bank) & (chg["comparison"] == "2015-18 (stress) -> 2019-25")
                & (chg["tree"] == "full")]
        if s.empty:
            continue
        fig, ax = plt_subplots(figsize=(8, 4.4))
        waterfall(ax, [plain_label(c, x) for c, x in zip(s["component"], s["subsection"], strict=True)],
                  list(s["contribution"]), "ROE change")
        style(ax, f"{t[bank]}: ROE change 2015-18 (stress) -> 2019-25, Shapley by component", "percentage points")
        png = save(fig, fig_dir / f"ch04_4_0_change_waterfall_{t[bank]}.png", "Evidence class: Derived. "
                   "Period-mean components; exact Shapley decomposition. Colours: green raises ROE, red lowers it.")
        data = write_csv(s, fig_dir / f"ch04_4_0_change_waterfall_{t[bank]}_data.csv", require_evidence=True)
        figs.append({"path": png, "data": data, "description": f"{t[bank]} ROE recovery after the 2015-18 stress",
                     "evidence_class": "Derived"})
        files += [png, data]
    res, interp, expl = _results(comp, gap, chg, rank_gap, rank_chg, t)
    nf = note(root, "4.0", NOTE.format(results=res, interpretation=interp, explanations=expl))
    tables = [{"path": files[0], "description": "Extended DuPont components and reconstructed ROE, every bank-year",
               "evidence_class": "Derived"},
              {"path": files[1], "description": "NMB - CRDB ROE gap: Shapley contributions by component, each year",
               "evidence_class": "Derived"},
              {"path": files[2], "description": "Each bank's ROE change between periods: Shapley contributions",
               "evidence_class": "Derived"},
              {"path": files[3], "description": "Subsection ranking (sum of contributions by subsection)",
               "evidence_class": "Derived"}]
    files += [nf, manifest(root, "4.0", "Extended DuPont and ROE-gap attribution", tables, figs, nf,
                           "tzbank.chapters.ch04.dupont_attribution", ["data/processed/bank_year_wide.csv"],
                           "built - attribution descriptive")]
    return files


__all__ = ["COLORS", "components", "shapley", "gap_attribution", "change_attribution", "write_dupont_attribution"]
