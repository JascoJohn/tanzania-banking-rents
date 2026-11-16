"""Chapter 4 starter: DuPont profitability panel (blueprint section 4, chapter 4, hypotheses H4.1/H4.2; formulas
from blueprint section 2: ROE on average attributable equity, cost of risk = impairment / average gross loans).

This is the smoke test of the data layer, not the chapter 4 analysis: it reports levels and makes no claim about
which profit engine generates rent (that needs the funding inputs listed in reports/data_audit.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tzbank.chapters.ch04.common import year_ticks  # noqa: E402
from tzbank.config import banks, find_root, path  # noqa: E402
from tzbank.io import rel, write_csv, write_json, write_text  # noqa: E402

METRICS = ["roe", "roa", "nii_to_avg_assets", "non_interest_income_share", "cost_to_income", "cost_of_risk",
           "leverage"]
# Reference categorical palette (dataviz skill references/palette.md, light mode), fixed order by bank_id.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


SUBSECTIONS = [("4.0", "Extended DuPont and Shapley attribution"), ("4.1", "Revenue composition"),
               ("4.2", "Funding economics"), ("4.3", "Asset deployment"), ("4.4", "Transaction economics"),
               ("4.5", "Operating leverage"), ("4.6", "Risk economics"), ("4.7", "Comparative business-model map")]

def average(cur: pd.Series, prev: pd.Series) -> pd.Series:
    """Simple average of opening and closing balances; NaN unless both exist."""
    return (cur + prev) / 2


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with NaN where den is missing or zero."""
    den = den.where(den != 0)
    return num / den


def _lag(df: pd.DataFrame, col: str) -> pd.Series:
    """Prior-year value for the same bank, only when the prior year is exactly t-1."""
    s = df.sort_values(["bank_id", "fiscal_year"])
    prev = s.groupby("bank_id")[col].shift(1)
    gap = s["fiscal_year"] - s.groupby("bank_id")["fiscal_year"].shift(1)
    return prev.where(gap == 1).reindex(df.index)


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)


def dupont_panel(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute the seven starter ratios for every bank-year (blueprint §2 definitions).

    - ROE = profit attributable / average equity attributable (fallback: net profit / average total equity, flagged
      in ``roe_basis`` where the NCI split is not held).
    - ROA = net profit / average total assets.
    - NII / average assets.
    - Non-interest income share = non-interest income / (NII + non-interest income).
    - Cost-to-income = operating expenses / (NII + non-interest income).
    - Cost of risk = impairment charges / average gross loans; ``cost_of_risk_net_loans_proxy`` uses average net
      loans and is a separately labelled proxy for banks without gross loans.
    - Leverage = average total assets / average equity used in ROE (so ROE = ROA x leverage x attributable share).
    """
    w = wide.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    num = {c: _col(w, c) for c in ["profit_attributable", "net_profit", "equity_attributable", "total_equity",
                                   "total_assets", "net_interest_income", "non_interest_income",
                                   "operating_expenses", "impairment_charges", "gross_loans", "net_loans"]}
    w2 = pd.DataFrame(num)
    w2["bank_id"], w2["fiscal_year"] = w["bank_id"], w["fiscal_year"]
    lag = {c: _lag(w2, c) for c in num}
    avg = {c: average(w2[c], lag[c]) for c in num}
    attr = safe_div(w2["profit_attributable"], avg["equity_attributable"])
    fallback = safe_div(w2["net_profit"], avg["total_equity"])
    use_attr = attr.notna()
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"]})
    out["roe"] = attr.where(use_attr, fallback)
    out["roe_basis"] = np.where(use_attr, "attributable profit / avg attributable equity",
                                np.where(fallback.notna(), "net profit / avg total equity (no NCI split held)", None))
    out["roa"] = safe_div(w2["net_profit"], avg["total_assets"])
    out["nii_to_avg_assets"] = safe_div(w2["net_interest_income"], avg["total_assets"])
    income = w2["net_interest_income"] + w2["non_interest_income"]
    out["non_interest_income_share"] = safe_div(w2["non_interest_income"], income)
    out["cost_to_income"] = safe_div(w2["operating_expenses"], income)
    out["cost_of_risk"] = safe_div(w2["impairment_charges"], avg["gross_loans"])
    out["cost_of_risk_net_loans_proxy"] = safe_div(w2["impairment_charges"], avg["net_loans"])
    eq = avg["equity_attributable"].where(use_attr, avg["total_equity"])
    out["leverage"] = safe_div(avg["total_assets"], eq)
    missing = out[METRICS].isna()
    out["coverage_flag"] = [
        "complete" if not m.any() else "partial: missing " + ", ".join(np.array(METRICS)[m.to_numpy()])
        for _, m in missing.iterrows()
    ]
    out["reporting_basis"] = _col(w, "reporting_basis")
    out["non_default_basis_vars"] = _col(w, "non_default_basis_vars")
    out["provenance"] = _col(w, "provenance")
    out["evidence_class"] = "Derived"
    has_any = out[METRICS].notna().any(axis=1)
    return out[has_any].reset_index(drop=True)


def roe_figure(panel: pd.DataFrame, png: Path, root: Path) -> pd.DataFrame:
    """ROE by bank, 300 dpi PNG; returns the plotted data."""
    names = {b["bank_id"]: b["ticker"] for b in banks(root)}
    data = panel[["bank_id", "fiscal_year", "roe", "roe_basis"]].dropna(subset=["roe"]).copy()
    data["ticker"] = data["bank_id"].map(names)
    data["evidence_class"] = "Derived"
    from tzbank.chapters.ch05.persistence import EXCLUDE_SHOWN as excl

    data["flag"] = [excl.get((b, int(y)), "") for b, y in zip(data["bank_id"], data["fiscal_year"], strict=True)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, b in enumerate(sorted(names)):
        d = data[data["bank_id"] == b].sort_values("fiscal_year")
        if d.empty:
            continue
        # Reindex to every year so gaps break the line (missing stays missing, CLAUDE.md 3.7).
        d = d.set_index("fiscal_year").reindex(range(int(d.fiscal_year.min()), int(d.fiscal_year.max()) + 1))
        d = d.rename_axis("fiscal_year").reset_index()
        flagged = [(y, d.loc[d["fiscal_year"] == y, "roe"].iloc[0]) for (bb, y) in excl if bb == b
                   and (d["fiscal_year"] == y).any()]
        for y, v in flagged:  # shown as an unconnected hollow marker, outside the line (as in chapter 5)
            d.loc[d["fiscal_year"] == y, "roe"] = np.nan
            ax.scatter([y], [v * 100], s=22, facecolors="none", edgecolors=SERIES_COLORS[i], zorder=3)
            ax.annotate(f"{names[b]} FY{y}: {v:.0%} (flagged)", (y, v * 100), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color="#5a5a55")
        focus = b in ("B01", "B02")
        ax.plot(d["fiscal_year"], d["roe"] * 100, color=SERIES_COLORS[i], lw=2.0 if focus else 1.2,
                alpha=1.0 if focus else 0.85, marker="o", ms=3.5 if focus else 2.5, label=names[b])
        if focus:
            last = d.iloc[-1]
            ax.annotate(names[b], (last["fiscal_year"], last["roe"] * 100), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#3a3a38")
    ax.axhline(0, color="#8a8a85", lw=0.8)
    ax.set_ylabel("ROE (%)", fontsize=9, color="#3a3a38")
    ax.set_xlabel("Fiscal year", fontsize=9, color="#3a3a38")
    ax.set_title("Return on average equity by bank, FY2009-FY2025", fontsize=10, loc="left", color="#1f1f1d")
    ax.grid(axis="y", color="#e4e3dd", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#5a5a55")
    ax.legend(ncol=7, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    from tzbank.reporting.chart_theme import snapshot

    snapshot(fig, png)  # report-theme copy before the footnote (prompt 16)
    fig.text(0.01, 0.005, "Evidence class: Derived. NMB/CRDB: attributable profit / avg attributable equity "
             "(inputs workbook). Other banks: net profit / avg total equity (Phase 1).\nHollow marker: flagged "
             "year outside the line (MUCOBA FY2020 near-zero average equity; excluded from estimation in ch. 5).",
             fontsize=6.5, color="#5a5a55")
    year_ticks(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, metadata={"Software": None})
    plt.close(fig)
    return data


NOTE = """# Chapter 4 starter: profitability (DuPont) panel

Generated by `python -m tzbank run profitability` (`tzbank.chapters.ch04.profitability`). Prompt 01 section 9.

## Research question

Blueprint chapter 4, *The economics of banking profits and business models*: how does each bank make money, and
why do their economics differ?

- **H4.1** Most of NMB's and CRDB's excess return comes from their funding advantage, not lending margin or fees.
- **H4.2** The two banks' excess returns rest on different profit engines.

This starter table is the first step only: it decomposes profitability into its components for every bank-year
where inputs exist. **It does not test H4.1 or H4.2.** H4.1 needs the cost of deposits and the gross deposit
funding advantage (4.2), which need interest expense on deposits and T-bill yields. Neither is held yet
(`reports/data_audit.md`). Excess return (ROE - COE) needs the annual COE series (chapter 5 input).

## Variable definitions (docs/VARIABLE_DICTIONARY.md)

| Column | Definition | Blueprint |
| --- | --- | --- |
| roe | Profit attributable / average equity attributable; fallback net profit / average total equity (`roe_basis`) | §2 |
| roa | Net profit / average total assets | §4 (4.1) |
| nii_to_avg_assets | Net interest income / average total assets | §4 (4.1) |
| non_interest_income_share | Non-interest income / (NII + non-interest income) | §4 (4.1, 4.4) |
| cost_to_income | Operating expenses / (NII + non-interest income) | §4 (4.5) |
| cost_of_risk | Impairment charges / average gross loans | §2, §4 (4.6) |
| cost_of_risk_net_loans_proxy | Impairment charges / average net loans (proxy where gross loans are not held) | labelled proxy |
| leverage | Average total assets / average equity used for ROE | DuPont identity |

Averages are (opening + closing) / 2 and need the immediately preceding fiscal year, so each bank's first year has
no average-based ratio.

## Sources and transformation trail

- Input: `data/processed/bank_year_wide.csv` (original, not restated; FY; the bank's own entity), built by
  `tzbank.panel.build.bank_year_wide` from `bank_period_long.csv`.
- NMB (B02) and CRDB (B01): inputs workbook v0.6, Historical Financials (Doc IDs CRDB-AR-FY20xx, NMB-AR-FY20xx),
  which replaces Phase 1 values for the same variable, year and basis (CLAUDE.md 3.8).
- DCB, Maendeleo, Mkombozi, Mwalimu, Mucoba (B03-B07): Phase 1 v2 RAW_PROFITABILITY and RAW_BALANCE_SHEET only.
- Functions: `tzbank.chapters.ch04.profitability.dupont_panel`, `roe_figure`.

## Validation test

`tests/test_ch04_profitability.py` checks every formula against hand-computed values on a fixture, plus
first-year and gap-year handling. The upstream accounting identities are in
`reports/validation/identity_checks.csv`.

## Evidence class

Every row: **Derived** (computed deterministically from Observed data). Statements below are labelled.

## What the table shows

{summary}

## Limitations

- Basis differs: NMB/CRDB are Group (consolidated) with NCI deducted; the other five are single-entity Bank
  accounts. `non_default_basis_vars` flags any variable taken from a non-default basis.
- For B03-B07 ROE uses net profit over total equity, because Phase 1 holds no NCI split. This is close to the
  attributable measure only if NCI is negligible (Interpretation).
- Gross loans are not held for B03-B07, so `cost_of_risk` is blank for them; the net-loans proxy overstates cost
  of risk relative to the gross definition.
- Accounting breaks: IAS 39 to IFRS 9 in FY2018 (blueprint L4); restatements are excluded from this original-
  vintage table.
- Levels only: the table cannot say whether high ROE is rent (needs COE, chapter 5) or which structural advantage
  produces it (needs funding and fee detail, chapter 4.2-4.4).

## Cross-cutting explanations of rent (blueprint §0)

Not assessable from this starter table. It neither supports nor contradicts any of the candidate explanations
(scale, low-cost deposits, switching costs, limited competition, regulatory advantage, distribution, credit cycle).
The cost-to-income and NII-margin columns are the inputs that sections 4.2 and 4.5 will use to test the scale and
deposit-franchise explanations.
"""


def _summary(panel: pd.DataFrame, root: Path) -> str:
    names = {b["bank_id"]: b["ticker"] for b in banks(root)}
    rows = []
    for b, g in panel.groupby("bank_id"):
        r = g.dropna(subset=["roe"])
        if r.empty:
            continue
        rows.append(f"| {names[b]} | {int(r.fiscal_year.min())}-{int(r.fiscal_year.max())} | {len(r)} | "
                    f"{r.roe.median():.1%} | {r.roe.min():.1%} | {r.roe.max():.1%} | "
                    f"{g.cost_to_income.median():.1%} | {(g.coverage_flag == 'complete').sum()} |")
    table = "\n".join(["| Bank | ROE years | n | Median ROE | Min | Max | Median cost-to-income | Complete rows |",
                       "| --- | --- | --- | --- | --- | --- | --- | --- |", *rows])
    med = panel.dropna(subset=["roe"]).groupby("bank_id")["roe"].median()
    focus = med.reindex(["B01", "B02"]).dropna()
    others = med.drop(index=[b for b in ["B01", "B02"] if b in med.index])
    text = ""
    if len(focus) and len(others):
        text = (f"\n\n**Derived.** Over the years shown, the median ROE of NMB and CRDB "
                f"({focus.min():.1%}-{focus.max():.1%}) is above the median ROE of every one of the five smaller "
                f"listed banks ({others.min():.1%} to {others.max():.1%}). The smaller banks' cost-to-income ratios "
                "are also higher. These are level comparisons on different bases (Group vs Bank, attributable vs "
                "total equity), not estimates of rent.")
    return (table + text + "\n\n**Interpretation (not yet tested).** Whether the gap is economic rent needs the "
            "annual COE series (chapter 5). Whether it comes from funding, margin or fees needs sections 4.1-4.4.")


def write_profitability(root: Path | None = None) -> list[Path]:
    """Write the ch04 starter table, figure (+ data), note and manifest."""
    root = root or find_root()
    wide = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    panel = dupont_panel(wide)
    t = path("tables", root) / "ch04_dupont_panel.csv"
    png = path("figures", root) / "ch04_roe_by_bank.png"
    data_csv = path("figures", root) / "ch04_roe_by_bank_data.csv"
    note = path("notes", root) / "ch04_starter.md"
    files = [write_csv(panel, t, require_evidence=True)]
    plotted = roe_figure(panel, png, root)
    files += [png, write_csv(plotted, data_csv, require_evidence=True)]
    files.append(write_text(NOTE.replace("{summary}", _summary(panel, root)), note))
    manifest = {
        "chapter": "04",
        "title": "The economics of banking profits and business models",
        "subsection": "starter (profitability / DuPont panel)",
        "hypotheses": ["H4.1", "H4.2"],
        "status": "starter only - hypotheses not yet tested",
        "tables": [{"path": rel(t, root), "description": "DuPont profitability panel, all bank-years with inputs",
                    "evidence_class": "Derived", "note": rel(note, root)}],
        "figures": [{"path": rel(png, root), "data": rel(data_csv, root),
                     "description": "ROE on average equity by bank", "evidence_class": "Derived",
                     "note": rel(note, root)}],
        "notes": [rel(note, root)],
        "inputs": ["data/processed/bank_year_wide.csv"],
        "module": "tzbank.chapters.ch04.profitability",
        "run_log_stage": "profitability",
        # overview index of the subsection manifests 4.0-4.7 (prompt 03b pre-flight; written by their modules)
        "subsections": [{"subsection": s, "title": tl, "manifest": f"outputs/manifests/ch04_{s.replace('.', '_')}.json"}
                        for s, tl in SUBSECTIONS],
    }
    files.append(write_json(manifest, path("manifests", root) / "ch04.json"))
    return files
