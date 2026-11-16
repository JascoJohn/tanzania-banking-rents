"""Chapter 4.7: comparative business-model map (blueprint §4, 4.7 and the deliverable; H4.1, H4.2).

One structured CSV (``outputs/tables/ch04_business_model_map.csv``) with, for NMB and CRDB: profit engines ranked by
the 4.0 attribution, competitive advantages mapped to a blueprint §0 explanation, sources of rent, vulnerabilities
(incl. digital disruption from NPS/H12 and FinScope), drivers of future profitability, answers to the blueprint §4
questions, and the H4.1/H4.2 verdicts. Every row carries evidence for and against, its evidence class and the 4.0
number it rests on.

Consistency rule (prompt 03b): a claim that names a winner on a dimension mapped to a 4.0 subsection must agree in
sign with that subsection's mean contribution to the NMB - CRDB gap; ``contradiction_flag`` is computed from both
files and must be False everywhere (tests read the map and the 4.0 tables independently).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.chapters.ch04.common import manifest, note, num, pct, tickers
from tzbank.config import find_root, path
from tzbank.io import write_csv

SUBS = {"4.2": "funding economics", "4.3": "asset deployment", "4.1/4.4": "revenue and transaction economics",
        "4.5": "operating leverage", "4.6": "risk economics", "4.7 leverage (TA/E)": "balance-sheet leverage (TA/E)",
        "4.7 tax (1-t)": "effective tax rate", "4.7 NCI share": "attributable (non-NCI) share of profit",
        "4.7 other items": "other reconciling items"}
LEV, TAX, NCI = "4.7 leverage (TA/E)", "4.7 tax (1-t)", "4.7 NCI share"
EXPLAIN = {"4.2": "low-cost deposits", "4.3": "distribution and market access", "4.1/4.4":
           "distribution and market access; customer switching costs", "4.5": "scale economies",
           "4.6": "temporary credit-cycle conditions",
           "4.7 leverage (TA/E)": "none of the §0 explanations (financial structure; chapter 7)",
           "4.7 tax (1-t)": "none of the §0 explanations (tax)", "4.7 NCI share": "none of the §0 explanations "
           "(ownership structure)", "4.7 other items": "none (reconciling items)"}
MATERIAL = 0.005  # a subsection names a winner only if its mean gap contribution is at least 0.5 pp
COLUMNS = ["bank", "dimension", "rank", "item", "attribution_subsection", "attribution_window",
           "attribution_mean_gap_pp", "winner", "explanation_s0", "evidence_for", "evidence_against",
           "evidence_class", "contradiction_flag"]


def load(root: Path) -> dict[str, pd.DataFrame]:
    t = path("tables", root)
    names = {"rank": "ch04_4_0_subsection_ranking.csv", "gap": "ch04_4_0_gap_attribution.csv",
             "chg": "ch04_4_0_change_attribution.csv", "fund": "ch04_4_2_funding.csv",
             "dep": "ch04_4_3_deployment.csv", "rev": "ch04_4_1_composition.csv", "tx": "ch04_4_4_transactions.csv",
             "mkt": "ch04_4_4_market_context.csv", "lev": "ch04_4_5_operating_leverage.csv",
             "el": "ch04_4_5_scale_elasticity.csv", "risk": "ch04_4_6_risk.csv"}
    return {k: pd.read_csv(t / v) for k, v in names.items()}


def mean_gap(d: dict[str, pd.DataFrame], years: tuple[int, int] | None = None) -> pd.Series:
    """Mean contribution of each subsection to the NMB - CRDB ROE gap across years (4.0), optionally a window."""
    r = d["rank"]
    r = r[r["comparison"] == "NMB - CRDB"]
    if years:
        r = r[r["fiscal_year"].between(*years)]
    return r.groupby("subsection")["contribution"].mean()


def winner_of(v: float | None) -> str:
    """NMB / CRDB when the 4.0 contribution is material (>= 0.5 pp), else 'neither'."""
    if v is None or pd.isna(v) or abs(v) < MATERIAL:
        return "neither (4.0 contribution below 0.5 pp)"
    return "NMB" if v > 0 else "CRDB"


def _mean(df: pd.DataFrame, bank: str, c: str, years=None) -> float:
    s = df[df["bank_id"] == bank]
    if years is not None:
        s = s[s["fiscal_year"].between(*years)]
    return s[c].mean()


def build(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    t = tickers(root)
    d = load(root)
    g = mean_gap(d)
    g_stress = mean_gap(d, (2015, 2018))
    rows: list[dict] = []

    def add(bank, dim, rank, item, sub, winner, expl, ev_for, ev_against, cls="Interpretation", window="all years"):
        gg = g_stress if window == "2015-2018" else g
        rows.append({"bank": bank, "dimension": dim, "rank": rank, "item": item, "attribution_subsection": sub,
                     "attribution_window": window,
                     "attribution_mean_gap_pp": round(float(gg.get(sub, float("nan"))) * 100, 2) if sub else None,
                     "winner": winner, "explanation_s0": expl, "evidence_for": ev_for,
                     "evidence_against": ev_against, "evidence_class": cls})

    # profit engines: NMB's are the subsections with a positive mean gap contribution, CRDB's the negative ones
    for bank, sign in (("NMB", 1), ("CRDB", -1)):
        eng = g[(g * sign > 0) & (g.abs() >= MATERIAL)].sort_values(key=lambda s: -s.abs())
        for i, (sub, v) in enumerate(eng.items(), start=1):
            add(bank, "profit engine (relative, 4.0)", i, f"{SUBS.get(sub, sub)} raises {bank}'s ROE relative to "
                f"the other bank", sub, bank, EXPLAIN.get(sub, ""),
                f"4.0 mean contribution to the NMB - CRDB gap {v * 100:+.2f} pp (ch04_4_0_subsection_ranking.csv)",
                "contributions vary by year; see ch04_4_0_gap_attribution.csv", "Derived")
    # funding (Q1)
    cn, cc = _mean(d["fund"], "B02", "cost_of_funds", (2010, 2025)), _mean(d["fund"], "B01", "cost_of_funds",
                                                                          (2010, 2025))
    dn, dc = _mean(d["fund"], "B02", "cost_of_deposits", (2010, 2025)), _mean(d["fund"], "B01", "cost_of_deposits",
                                                                             (2010, 2025))
    win = winner_of(g.get("4.2"))
    add(win, "question 1: stronger funding economics", None, f"stronger funding economics: {win}", "4.2", win,
        EXPLAIN["4.2"], f"mean cost of funds 2010-25: NMB {pct(cn, 2)}, CRDB {pct(cc, 2)}; cost of deposits NMB "
        f"{pct(dn, 2)}, CRDB {pct(dc, 2)}; 4.0 funding "
        f"contribution {g.get('4.2', 0) * 100:+.2f} pp", "FY2025 cost of deposits NMB 2.37% vs CRDB 2.19% (gap "
        "narrowing); GDFA is a proxy (D12)", "Derived")
    # deployment (Q2)
    win = winner_of(g.get("4.3"))
    ln, lc = _mean(d["dep"], "B02", "loan_yield", (2015, 2025)), _mean(d["dep"], "B01", "loan_yield", (2015, 2025))
    add(win, "question 2: deploys capital more productively", None, f"higher earnings on the asset base: {win}", "4.3",
        win, EXPLAIN["4.3"], f"mean loan yield 2015-25: NMB {pct(ln, 2)}, CRDB {pct(lc, 2)}; 4.0 asset-deployment "
        f"contribution {g.get('4.3', 0) * 100:+.2f} pp",
        f"NMB's yield advantage fell 2019-25 (4.0: y_iea is NMB's largest negative change); risk-adjusted NIM FY2025 "
        f"NMB {pct(_mean(d['risk'], 'B02', 'risk_adjusted_nim', (2025, 2025)), 2)} vs CRDB "
        f"{pct(_mean(d['risk'], 'B01', 'risk_adjusted_nim', (2025, 2025)), 2)}", "Derived")
    # operating leverage (Q3)
    en = d["el"].set_index("bank_id")
    ctn, ctc = _mean(d["lev"], "B02", "cost_to_income", (2021, 2025)), _mean(d["lev"], "B01", "cost_to_income",
                                                                             (2021, 2025))
    sub = g.get("4.5", 0)
    win = winner_of(sub)
    add(win if win in ("NMB", "CRDB") else "both", "question 3: better operating leverage", None,
        f"operating leverage: {win}; NMB's lower cost-to-income comes from higher income per unit of assets (4.3), "
        "not from lower cost per unit of assets", "4.5", win, EXPLAIN["4.5"],
        f"cost-to-income 2021-25 mean NMB {pct(ctn)}, CRDB {pct(ctc)}; scale elasticity NMB "
        f"{en.loc['B02', 'elasticity']:.2f} ({en.loc['B02', 'reading']}), CRDB {en.loc['B01', 'elasticity']:.2f} "
        f"({en.loc['B01', 'reading']}); 4.0 opex contribution {sub * 100:+.2f} pp",
        "the 4.0 cost term barely separates the banks on average; both cut OpEx / assets sharply 2019-25",
        "Derived")
    # diversification (Q4)
    nn, nc = _mean(d["rev"], "B02", "nonii_share", (2021, 2025)), _mean(d["rev"], "B01", "nonii_share", (2021, 2025))
    win = winner_of(g.get("4.1/4.4"))
    add(win if win in ("NMB", "CRDB") else "both", "question 4: more diversified income", None,
        f"more diversified income: {win}", "4.1/4.4", win,
        EXPLAIN["4.1/4.4"], f"non-interest income share 2021-25 mean NMB {pct(nn)}, CRDB {pct(nc)}; 4.0 "
        f"non-interest contribution {g.get('4.1/4.4', 0) * 100:+.2f} pp",
        "fee-type shares (incl. the digital share) are LOW confidence (label-matched) and excluded from this "
        "verdict until the fee-type mapping is verified", "Derived")
    # risk
    rn, rc = _mean(d["risk"], "B02", "cost_of_risk", (2015, 2018)), _mean(d["risk"], "B01", "cost_of_risk",
                                                                          (2015, 2018))
    win = winner_of(g_stress.get("4.6"))
    add(win if win in ("NMB", "CRDB") else "both", "question 5: lower credit cost through the stress", None,
        f"lower credit cost in the 2015-18 stress: {win}", "4.6", win, EXPLAIN["4.6"],
        f"cost of risk 2015-18 mean NMB {pct(rn, 2)}, CRDB {pct(rc, 2)}; 4.0 impairment contribution 2015-18 "
        f"{g_stress.get('4.6', 0) * 100:+.2f} pp",
        f"small on average over all years ({g.get('4.6', 0) * 100:+.2f} pp); one cycle only; chapter 8", "Derived",
        window="2015-2018")
    # capital structure
    win = winner_of(g.get(LEV))
    add(win, "question 6: capital structure", None, f"ROE gain from higher leverage (TA/E): {win}", LEV, win,
        EXPLAIN[LEV], f"4.0 leverage (TA/E) contribution {g.get(LEV, 0) * 100:+.2f} pp, isolated from tax "
        f"({g.get(TAX, 0) * 100:+.2f} pp) and the NCI share ({g.get(NCI, 0) * 100:+.2f} pp)",
        "higher leverage raises ROE without creating rent (risk, capital headroom: chapter 7)", "Derived")
    # durability and disruption (Q7, Q8)
    mm = d["mkt"][d["mkt"]["series"].str.startswith("Mobile money transactions")].sort_values("year")
    mm_g = (mm["value"].iloc[-1] / mm["value"].iloc[0]) ** (1 / (mm["year"].iloc[-1] - mm["year"].iloc[0])) - 1 \
        if len(mm) > 1 else float("nan")
    add("both", "question 7: durable advantages", None, "the deposit franchise and scale are the more durable "
        "advantages; the asset-yield edge is eroding", "4.2", "both", "low-cost deposits; scale economies",
        "cost of deposits below the T-bill rate in every year for both (4.2); NMB scale elasticity CI below 1 (4.5)",
        "NMB's asset-yield contribution shrank 2019-25 (4.0); persistence is a chapter 5 test", "Interpretation")
    add("both", "question 8: exposure to disruption", None, "transaction fees and transactional deposits are exposed "
        "to mobile-money competition", "4.1/4.4", "both", "customer switching costs (eroding)",
        f"market-level mobile money transaction value grew {pct(mm_g)} a year (NPS, 4.4); digital savings and credit "
        "in Table H12 grew fastest of all channels", "bank-level market shares are not held; FinScope usage evidence "
        "is in chapter 2", "Interpretation")
    # sources of rent, vulnerabilities, drivers
    for bank in ("NMB", "CRDB"):
        add(bank, "source of rent (candidate)", None, "deposit franchise (cost of deposits well below the T-bill rate)",
            "4.2", "both", EXPLAIN["4.2"], "GDFA proxy about 5% of average assets in FY2025 (4.2)",
            "gross proxy (D12); net franchise and rent test in chapter 6; COE needed (chapter 5)", "Interpretation")
        add(bank, "vulnerability", None, "falling asset yields as government-securities and loan rates fall", "4.3",
            "both", "temporary credit-cycle conditions", "4.0 2019-25: y_iea negative for both banks",
            "policy rate rose again from 2024 (D42); yields may stabilise", "Interpretation")
        add(bank, "driver of future profitability", None, "cost discipline (OpEx / assets) and credit costs",
            "4.5", "both", "scale economies; temporary credit-cycle conditions",
            "OpEx / assets was the largest positive contributor to both banks' 2019-25 ROE change (4.0)",
            "cost-of-risk normalisation would reverse part of the gain (4.6)", "Interpretation")
    # hypotheses
    rank = g.sort_values(key=lambda s: -s.abs())
    add("both", "hypothesis H4.1", None, "NOT SUPPORTED: funding is not the main source of the difference in "
        f"returns; funding ({g.get('4.2', 0) * 100:+.2f} pp) ranks behind asset deployment "
        f"({g.get('4.3', 0) * 100:+.2f} pp, NMB) and balance-sheet leverage TA/E alone "
        f"({g.get(LEV, 0) * 100:+.2f} pp, CRDB), with tax and the NCI share isolated as separate terms", "4.2", "n/a",
        EXPLAIN["4.2"],
        "4.0 ranking of mean gap contributions: " + "; ".join(f"{k} {v * 100:+.2f} pp" for k, v in rank.items()),
        "levels: the GDFA proxy is large (about 5% of assets), so funding matters for both banks' ROE level; the "
        "test of excess return needs COE (chapter 5) and the net franchise (chapter 6)", "Derived")
    add("both", "hypothesis H4.2", None, "SUPPORTED: the banks' profit engines differ (NMB: asset yield, funding, "
        "fees; CRDB: leverage)", LEV, "n/a", "",
        "opposite-signed mean gap contributions by subsection (4.0)", "differences in operating leverage and risk are "
        "small on average", "Derived")
    out = pd.DataFrame(rows, columns=COLUMNS[:-1])
    out["contradiction_flag"] = contradictions(out, g, g_stress)
    out["bank"] = out["bank"].replace({"B01": t["B01"], "B02": t["B02"]})
    return out


def contradictions(m: pd.DataFrame, g: pd.Series, g_stress: pd.Series | None = None) -> pd.Series:
    """True where a claim names NMB or CRDB as winner on a 4.0 subsection whose mean gap contribution (over the
    claim's window) has the opposite sign or is not material (positive means NMB ahead)."""
    flags = []
    for _, r in m.iterrows():
        sub, w = r["attribution_subsection"], r["winner"]
        gg = g_stress if r.get("attribution_window") == "2015-2018" and g_stress is not None else g
        v = gg.get(sub)
        if w not in ("NMB", "CRDB"):
            flags.append(False)
        elif v is None or pd.isna(v) or abs(v) < MATERIAL:
            flags.append(True)  # a named winner needs a material 4.0 contribution
        else:
            flags.append((w == "NMB") != (v > 0))
    return pd.Series(flags, index=m.index)


NOTE = """# Chapter 4.7: comparative business-model map

Generated by `tzbank.chapters.ch04.business_model_map` (stage `profitability`). Blueprint §4, 4.7 and the
deliverable; hypotheses H4.1 and H4.2.

## Research question

Why do NMB and CRDB generate different economic outcomes: which profit engines, advantages, sources of rent,
vulnerabilities and drivers distinguish them?

## Variable definitions

The map (`outputs/tables/ch04_business_model_map.csv`) has one row per claim: bank, dimension, item, the 4.0
subsection it rests on and that subsection's mean contribution to the NMB - CRDB ROE gap (percentage points,
positive = NMB ahead), the winner named, the blueprint §0 explanation, evidence for and against, evidence class and
a contradiction flag (claim versus 4.0 sign).

## Sources and transformation trail

4.0-4.6 tables in `outputs/tables/` (`ch04_4_0_*`, `ch04_4_1_composition`, `ch04_4_2_funding`, `ch04_4_3_deployment`,
`ch04_4_4_*`, `ch04_4_5_*`, `ch04_4_6_risk`); function `build`, `contradictions`.

## Validation test

`tests/test_ch04_03b.py`: reads the map and the 4.0 ranking independently and fails on any contradiction flag or
any claim without evidence.

## Results

### Hypotheses

{hypotheses}

### The questions of blueprint §4

{questions}

### Profit engines (ranked by the 4.0 attribution)

{engines}

## Evidence class

Map rows are labelled one by one: **Derived** for claims that are arithmetic on the 4.0-4.6 tables, **Interpretation**
for judgements (durability, disruption, rent candidates). No row is a rent estimate.

## Limitations

- The 4.0 attribution is descriptive (Shapley, D12); "engine" means a component that raises one bank's ROE relative
  to the other, not a cause.
- Rent needs the cost of equity (chapter 5) and the net deposit franchise (chapter 6); persistence is chapter 5.
- Disruption evidence is market-level (NPS, H12); bank-level market shares are not held.

## Interpretation

**Interpretation.** NMB's edge has come from what it earns on its assets and from cheaper funding; CRDB closes the
return gap by running more leverage. Both banks' recent ROE gains came mostly from lower costs relative to assets and,
for CRDB, from lower credit losses after 2018, which are partly cyclical. The difference in returns is therefore less
about a funding moat than the blueprint's H4.1 expected; the deposit franchise still matters for both banks' ROE level,
which chapter 6 measures net of its costs.

## Cross-cutting explanations of rent (blueprint §0)

Supported (for the difference between the banks): **distribution and market access** (asset yield, fees) and
**low-cost deposits** (smaller weight than expected). Partly supported: **scale economies** (NMB elasticity below 1).
Cyclical: **temporary credit-cycle conditions** (impairment in CRDB's recovery). Not assessable here: **limited
competition**, **regulatory advantages**, **switching costs** (chapters 1, 6, 8).
"""


def write_business_model_map(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    m = build(root)
    f = write_csv(m, path("tables", root) / "ch04_business_model_map.csv", require_evidence=True)
    hyp = m[m["dimension"].str.startswith("hypothesis")]
    q = m[m["dimension"].str.startswith("question")]
    e = m[m["dimension"].str.startswith("profit engine")]
    text = NOTE.format(
        hypotheses="\n".join(f"- **{r.dimension.replace('hypothesis ', '')}: {r.item}** Evidence: {r.evidence_for}. "
                             f"Against: {r.evidence_against}. ({r.evidence_class})" for r in hyp.itertuples()),
        questions="\n".join(f"- {r.dimension.split(': ', 1)[1].capitalize()}: **{r.winner}**. {r.evidence_for}. "
                            f"Caveat: {r.evidence_against}. ({r.evidence_class})" for r in q.itertuples()),
        engines="\n".join(f"- {r.bank} #{int(r.rank)}: {r.item} ({r.evidence_for})" for r in e.itertuples()))
    nf = note(root, "4.7", text)
    # chapter 4 overview index (pre-flight: ch04.json lists the subsections)
    files = [f, nf, manifest(root, "4.7", "Comparative business-model map",
                             [{"path": f, "description": "Comparative business-model map (claims, evidence, 4.0 "
                               "numbers, contradiction flag)", "evidence_class": "Derived"}], [], nf,
                             "tzbank.chapters.ch04.business_model_map",
                             ["outputs/tables/ch04_4_0_*.csv ... ch04_4_6_risk.csv"],
                             "built")]
    return files


__all__ = ["build", "contradictions", "write_business_model_map", "num"]
