"""Chapter 2: credit allocation patterns (blueprint §4 chapter 2; H2.1, H2.2; descriptive only, L1).

Loan books by industry: NMB and CRDB from the IFRS 7 industry-concentration note (``data/interim/loans_by_sector.csv``,
prompt 10) and the commercial-bank system from BoT credit by economic activity (``credit_by_activity.csv``, December,
FY2010 onward when the BoT activity list is stable). Both are mapped to eight common categories (``CATEGORIES``) and
compared with the dissimilarity (Duncan) index D = 1/2 x sum |s_a - s_b| (0 = identical mixes, 1 = no overlap): the
share of one book that would have to move to match the other. Context: credit by provider (banks, SACCOs, tier-2
microfinance; inclusion reports) and digital credit (BoT NPS report Table H12, a transaction flow, not a stock).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, pct, plt_subplots, save, style, tickers
from tzbank.config import path
from tzbank.io import rel, write_csv, write_json, write_text

MATERIAL = 0.20  # D59 (accepted, amended): primary threshold; verdicts also at 0.15 and 0.25
THRESHOLDS = (0.15, 0.20, 0.25)
BASIS = ("no published benchmark exists for loan-mix dissimilarity; D is the share of one book that would have to move "
         "between the eight categories to match the other, so 0.20 = one fifth of the book (a judgement threshold, "
         "bracketed by 0.15 and 0.25); the coarse eight-category mapping biases D downwards")
BANK_MAP = {"agriculture": "agriculture", "manufacturing": "manufacturing", "trading": "trade",
            "trading and commercial": "trade", "trading and commerce": "trade", "wholesale & retail": "trade",
            "wholesale and retail": "trade", "transport and communication": "transport and communication",
            "transport & communication": "transport and communication",
            "transport and communications": "transport and communication",
            "hotel and restaurant": "hotels and tourism", "hotels and restaurants": "hotels and tourism",
            "hotel & restaurant": "hotels and tourism", "tourism": "hotels and tourism",
            "financial institutions": "financial institutions", "individuals": "households and personal",
            "individual": "households and personal", "personal": "households and personal",
            "households": "households and personal"}
BOT_MAP = {"AGRICULTURE": "agriculture", "FISHING": "agriculture", "FOREST": "agriculture", "HUNTING": "agriculture",
           "MANUFACTURING": "manufacturing", "TRADE": "trade", "TRANSPORT_COMMUNICATION": "transport and communication",
           "HOTEL_RESTAURANT": "hotels and tourism", "TOURISM": "hotels and tourism",
           "FINANCIAL_INTERMEDIARIE": "financial institutions", "PERSONAL_OTHER_SERVICE": "households and personal",
           "BUILDING_CONSTRUCTION": "other", "EDUCATION": "other", "ELECTRICITY": "other", "GAS": "other",
           "HEALTH": "other", "LEASING": "other", "MINING_QUARRYING": "other", "MORTGAGE": "other",
           "REAL_ESTATE": "other", "WAREHOUSING_STORAGE": "other", "WATER": "other"}
CATEGORIES = ["agriculture", "manufacturing", "trade", "transport and communication", "hotels and tourism",
              "financial institutions", "households and personal", "other"]


def dissimilarity(a: pd.Series, b: pd.Series) -> float:
    """Duncan dissimilarity index between two share vectors over the same categories."""
    a, b = a.reindex(CATEGORIES).fillna(0), b.reindex(CATEGORIES).fillna(0)
    return float(0.5 * (a / a.sum() - b / b.sum()).abs().sum())


def books(root: Path) -> pd.DataFrame:
    """Category values and shares per book (NMB, CRDB, NMB + CRDB, BoT system) and year."""
    s = pd.read_csv(path("interim", root) / "loans_by_sector.csv", dtype={"bank_id": str})
    s["category"] = s["sector_label"].map(BANK_MAP).fillna("other")
    t = tickers(root)
    bk = s.groupby(["bank_id", "fiscal_year", "category"])["value"].sum().reset_index()
    bk["book"] = bk["bank_id"].map(t)
    both = bk.groupby(["fiscal_year", "category"])["value"].sum().reset_index()
    nyrs = bk.groupby("fiscal_year")["bank_id"].nunique()
    both = both[both["fiscal_year"].map(nyrs) == 2].assign(book="NMB + CRDB")
    c = pd.read_csv(path("processed", root) / "credit_by_activity.csv")
    c = c[c["year"] >= 2010].copy()
    c["key"] = c["series_id"].str.replace("BOTDB_CREDIT_BY_ACTIVITY__", "", regex=False)
    c["category"] = c["key"].map(BOT_MAP)
    sysb = c.dropna(subset=["category"]).groupby(["year", "category"])["value"].sum().reset_index()
    sysb = sysb.rename(columns={"year": "fiscal_year"}).assign(book="BoT commercial banks", value=lambda d: d["value"] * 1e6)
    tot = c[c["key"] == "TOTAL"].set_index("year")["value"] * 1e6
    out = pd.concat([bk.drop(columns="bank_id"), both, sysb], ignore_index=True)
    out["share"] = out["value"] / out.groupby(["book", "fiscal_year"])["value"].transform("sum")
    out["bot_total_coverage"] = np.where(out["book"] == "BoT commercial banks", out.groupby(
        ["book", "fiscal_year"])["value"].transform("sum") / out["fiscal_year"].map(tot), np.nan)
    out["evidence_class"] = "Derived"
    return out.sort_values(["book", "fiscal_year", "category"]).reset_index(drop=True)


def indices(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    piv = b.pivot_table(index=["book", "fiscal_year"], columns="category", values="value", aggfunc="sum")
    for a_, b_, lab in (("NMB", "CRDB", "NMB vs CRDB (H2.1)"), ("NMB + CRDB", "BoT commercial banks",
                                                                 "NMB + CRDB vs system (H2.2)"),
                        ("NMB", "BoT commercial banks", "NMB vs system"), ("CRDB", "BoT commercial banks",
                                                                         "CRDB vs system")):
        if a_ not in piv.index.get_level_values(0) or b_ not in piv.index.get_level_values(0):
            continue
        ya, yb = set(piv.loc[a_].index), set(piv.loc[b_].index)
        for y in sorted(ya & yb):
            rows.append({"comparison": lab, "fiscal_year": int(y), "dissimilarity": dissimilarity(piv.loc[(a_, y)],
                                                                                                 piv.loc[(b_, y)]),
                         "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def providers(root: Path) -> pd.DataFrame:
    inc = pd.read_csv(path("processed", root) / "inclusion_panel.csv", low_memory=False)
    keep = {"incl_crd_bank_loan_amt": "banks (loan amount)", "incl_crd_sacco_main_loan_amt": "SACCOs, Mainland",
            "incl_crd_sacco_znz_loan_amt": "SACCOs, Zanzibar", "incl_crd_t2_loan_amt": "tier-2 microfinance"}
    x = inc[inc["canonical_name"].isin(keep)][["canonical_name", "Year", "value", "Std_Unit", "source_report"]].copy()
    x["provider"] = x["canonical_name"].map(keep)
    x = x.rename(columns={"Year": "year", "Std_Unit": "unit"})
    n = pd.read_csv(path("processed", root) / "nps_panel.csv", low_memory=False)
    d = n[n["canonical_name"] == "code_nps_table_h12_digital_credit_value_of_transactions_tzs_billions_"][
        ["canonical_name", "year", "value", "unit", "source_report"]].copy()
    d["provider"] = "digital credit (value of transactions, a flow)"
    out = pd.concat([x, d], ignore_index=True)
    out["evidence_class"] = "Observed"
    return out.sort_values(["provider", "year"]).reset_index(drop=True)


def _verdict(d: pd.Series, t: float) -> str:
    """D > t in every common year: supported; D <= t in every year: rejected; else inconclusive (D59 amended)."""
    m = d > t
    return "supported" if m.all() else "rejected" if (~m).all() else "inconclusive"


def hypotheses(ix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, lab, stmt in (("H2.1", "NMB vs CRDB (H2.1)", "the two banks' loan mixes"),
                         ("H2.2", "NMB + CRDB vs system (H2.2)", "the combined mix and the system's")):
        x = ix[ix["comparison"] == lab]
        if x.empty:
            rows.append({"hypothesis": h, "verdict": "inconclusive", "evidence": "no common years", "rule": ""})
            continue
        d = x["dissimilarity"]

        def verdict(t: float, d: pd.Series = d) -> str:
            return _verdict(d, t)

        rows.append({"hypothesis": h, "verdict": verdict(MATERIAL),
                     "verdict_by_threshold": "; ".join(f"{t:.2f}: {verdict(t)}" for t in THRESHOLDS),
                     "d_mean": d.mean(), "d_min": d.min(), "d_max": d.max(),
                     "evidence": f"dissimilarity between {stmt} (continuous): " + ", ".join(
                         f"FY{r.fiscal_year} {r.dissimilarity:.2f}" for r in x.itertuples())
                     + f"; mean {d.mean():.2f}, range {d.min():.2f}-{d.max():.2f}; verdict at "
                     + ", ".join(f"{t:.2f} {verdict(t)}" for t in THRESHOLDS),
                     "rule": f"descriptive (L1): supported when D > threshold in every common year, rejected when "
                             f"D <= threshold in every year, inconclusive otherwise; primary threshold {MATERIAL:.2f}, "
                             f"also 0.15 and 0.25 (D59 amended). Threshold basis: {BASIS}"})
    return pd.DataFrame(rows).assign(evidence_class="Interpretation")


NOTE = """# Chapter 2: credit allocation patterns

Generated by `tzbank.chapters.ch02_credit_allocation` (stage `part1`). Blueprint §4 chapter 2; H2.1, H2.2.
**Descriptive only: no economy-wide causal claim (L1).**

## Research question

How do NMB's and CRDB's loan books differ by industry, from each other and from the commercial-bank system, and where
do banks sit among credit providers?

## Variable definitions

Category shares of each loan book over eight common categories ({cats}); dissimilarity D = 1/2 x sum |s_a - s_b|.
Bank books: IFRS 7 industry-concentration note, loans and advances to customers at carrying amount (Group where
printed); system: BoT credit by economic activity, December, FY2010 onward ('personal and other services' is the BoT
line for households).

## Sources and transformation trail

`data/interim/loans_by_sector.csv` (`tzbank.extract.sector_loans`, review `reports/validation/loans_by_sector_review.csv`),
`data/processed/credit_by_activity.csv`, `data/processed/inclusion_panel.csv`, `data/processed/nps_panel.csv`.
Functions: `books`, `dissimilarity`, `indices`, `providers`, `hypotheses`.

## Validation test

`tests/test_part1.py`: shares sum to 1 per book and year; D hand case; D within [0, 1]; extracted sector totals
reconcile to net loans.

## Results (Derived)

### Coverage

{cov}

### Loan mixes (latest common year)

{mix}

### Dissimilarity

{dis}

### Credit by provider (context, Observed)

{prov}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

Shares and indices: **Derived**; provider figures: **Observed**; verdicts: **Interpretation**.

## Limitations

- L1: seven listed banks of the system; no claim about how credit allocation affects the economy.
- Bank books are extracted for {nyears}; other years did not reconcile or print the matrix (collection task).
- Category mapping is coarse: the banks' 'others' and the BoT's 'personal and other services' are broad; the BoT
  system includes NMB and CRDB.
- Digital credit is a flow of disbursements, not an outstanding balance.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

A household-heavy book (salary loans) points to **distribution and market access** and **customer switching costs**
(payroll relationships); an agriculture-heavy book reflects a specialised franchise (CRDB's cooperative and
agribusiness lending). Neither shows rent by itself.
"""


def write_ch02(root: Path) -> list[Path]:
    b = books(root)
    ix = indices(b)
    pv = providers(root)
    hyp = hypotheses(ix)
    tab, figd = path("tables", root), path("figures", root)
    tables = [(b, "ch02_loan_mix.csv", "Loan mix by category: NMB, CRDB, combined, BoT system", "Derived"),
              (ix, "ch02_dissimilarity.csv", "Dissimilarity indices", "Derived"),
              (pv, "ch02_credit_providers.csv", "Credit by provider (inclusion reports, NPS digital credit)",
               "Observed"),
              (hyp, "ch02_hypotheses.csv", "H2.1-H2.2", "Interpretation")]
    cv = b.loc[b["book"] == "BoT commercial banks", ["fiscal_year", "bot_total_coverage"]].dropna()  # as in the note
    tables.insert(1, (pd.DataFrame([{"measure": "BoT total covered by the mapped categories",
                                     "years": f"FY{int(cv['fiscal_year'].min())}-{int(cv['fiscal_year'].max())}",
                                     "mean": cv["bot_total_coverage"].mean(), "min": cv["bot_total_coverage"].min(),
                                     "max": cv["bot_total_coverage"].max(), "evidence_class": "Derived"}]),
                      "ch02_bot_coverage.csv", "Share of BoT credit by activity covered by the mapped categories",
                      "Derived"))
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    t = tickers(root)
    yrs = sorted(set(b[b["book"] == "NMB + CRDB"]["fiscal_year"]) & set(b[b["book"] == "BoT commercial banks"]
                                                                       ["fiscal_year"]))
    y = yrs[-1] if yrs else int(b["fiscal_year"].max())
    x = b[b["fiscal_year"] == y].pivot_table(index="category", columns="book", values="share").reindex(CATEGORIES)
    fig, ax = plt_subplots(figsize=(8.5, 4.4))
    books_ = [c for c in ("NMB", "CRDB", "BoT commercial banks") if c in x.columns]
    w = 0.8 / max(len(books_), 1)
    colors = {"NMB": COLORS["B02"], "CRDB": COLORS["B01"], "BoT commercial banks": "#8a8a84"}
    for i, bk in enumerate(books_):
        ax.bar(np.arange(len(CATEGORIES)) + i * w, x[bk].fillna(0) * 100, width=w, color=colors[bk], label=bk)
    ax.set_xticks(np.arange(len(CATEGORIES)) + w, [c.replace(" and ", " & ") for c in CATEGORIES], rotation=30,
                  ha="right", fontsize=7)
    style(ax, f"Loan mix by industry, FY{y}", "percent of loans")
    ax.legend(fontsize=7, frameon=False)
    png = save(fig, figd / "ch02_loan_mix.png", "Evidence class: Derived. Banks: IFRS 7 industry note; system: BoT credit "
               "by economic activity (December).")
    fd = write_csv(b[b["fiscal_year"] == y][["book", "fiscal_year", "category", "share", "evidence_class"]],
                   figd / "ch02_loan_mix_data.csv", require_evidence=True)
    files += [png, fd]
    s = pd.read_csv(path("interim", root) / "loans_by_sector.csv", dtype={"bank_id": str})
    sysy = b[b["book"] == "BoT commercial banks"]["fiscal_year"]
    cov = "\n".join(f"- {t[k]}: FY{', FY'.join(str(v) for v in sorted(g['fiscal_year'].unique()))}"
                    for k, g in s.groupby("bank_id"))
    cov += (f"\n- BoT system: FY{int(sysy.min())}-{int(sysy.max())} (mapped categories cover "
            f"{pct(b['bot_total_coverage'].dropna().mean())} of the BoT total on average)")
    mix = "\n".join(f"- {c}: " + "; ".join(f"{bk} {pct(x.loc[c, bk])}" for bk in x.columns if pd.notna(x.loc[c, bk]))
                    for c in CATEGORIES)
    dis = "\n".join(f"- {lab}: " + ", ".join(f"FY{r.fiscal_year} {r.dissimilarity:.2f}" for r in g.itertuples())
                    for lab, g in ix.groupby("comparison", sort=False))
    prov = "\n".join(f"- {p}: " + ", ".join(f"{int(r.year)} {r.value:,.0f} {r.unit}" for r in g.itertuples())
                     for p, g in pv.groupby("provider"))
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Rule: {r.rule}." for r in hyp.itertuples())
    interp = ("**Interpretation.** The industry mixes describe where each bank's loan book sits; "
              "they are descriptive and say nothing about the banks' effect on credit to the economy (L1).")
    nf = write_text(NOTE.format(cats=", ".join(CATEGORIES), cov=cov, mix=f"FY{y}:\n\n" + mix, dis=dis, prov=prov,
                                hyp=hyp_txt, nyears="the years listed under Coverage", interp=interp),
                    path("notes", root) / "ch02.md")
    files.append(nf)
    m = {"chapter": "02", "title": "Credit allocation patterns", "hypotheses": ["H2.1", "H2.2"],
         "status": "built - descriptive",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for _, n, d, c in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "Loan mix by industry",
                      "evidence_class": "Derived", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "module": "tzbank.chapters.ch02_credit_allocation", "run_log_stage": "part1"}
    files.append(write_json(m, path("manifests", root) / "ch02.json"))

    return files
