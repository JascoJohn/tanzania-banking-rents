"""Analysis exports for the report text (prompt 17 Part B; D79): pipeline tables the owner cites.

Each export reshapes figures that already exist in pipeline tables (chapters 2, 4, 7, 9, 12, 14) into the form the
text needs; nothing new is estimated except where stated (the B8 currency sensitivity is a named chapter 12 scenario,
D66). Tables ``outputs/tables/ch13_text_*.csv``, charts ``outputs/figures/ch13_text_*.png`` (house colours; report
variants through the chart theme), manifest ``outputs/manifests/ch13_text.json`` (dashboard chapter 13 page), so every
figure carries a findings-pack id (CH13.TEXT_*; pack v1.9).

- B1 ``ch13_text_loan_mix``: sector shares FY2023 and FY2025 (system: FY2023 and latest), change in points.
- B2 ``ch13_text_fee_mix``: fee income by type (shares), fee income (TZS), fees / total income; NMB 'other' fees.
- B3 ``ch13_text_implied_growth``: price, COE, base g, guard g, value at the guard and at base g; NMB HIST post-split.
- B4 ``ch13_text_thesis_breaks``: rows r0-r9 with current value, threshold, margin and the direction of a breach.
- B5 ``ch13_text_yield_minus_tbill``: dividend yield less the bill rate, with the bill rate used.
- B6 ``ch13_text_real_excess``: real ROE less real COE, FY2010-FY2025.
- B7 ``ch13_text_rore``: CRDB RORE primary and raise-adjusted, k = 3 and 5, and on the windows both series share.
- B8 ``ch13_text_usd_ytd``: TZS/USD Dec 2025 to 30 Oct 2026, 2026 YTD returns, expected USD returns with the realised
  move in place of the 3.84% history assumption.
- A5 ``ch13_split_restated``: NMB per-share figures dated before the 1:10 split, restated (annexes print these).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path
from tzbank.io import write_csv, write_json, write_text

BID = {"CRDB": "B01", "NMB": "B02"}
NAME = {v: k for k, v in BID.items()}
SYSTEM = "#8c8c86"


def _t(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, low_memory=False, dtype={"bank_id": str})


def _foot(src: str, ev: str) -> str:
    return f"Source: {src}. Evidence class: {ev}."


# --- B1 -----------------------------------------------------------------------------------------------------------------
def loan_mix(root: Path) -> pd.DataFrame:
    lm = _t(root, "ch02_loan_mix.csv")
    rows = []
    for book, g in lm.groupby("book"):
        years = sorted(g["fiscal_year"].unique())
        base = 2023 if 2023 in years else years[0]
        latest = max(years)
        a = g[g["fiscal_year"] == base].set_index("category")["share"]
        z = g[g["fiscal_year"] == latest].set_index("category")["share"]
        for cat in sorted(set(a.index) | set(z.index)):
            s0, s1 = a.get(cat, np.nan), z.get(cat, np.nan)
            rows.append({"book": book, "category": cat, "base_fy": int(base), "share_base": s0, "latest_fy": int(latest),
                         "share_latest": s1, "change_pp": (s1 - s0) * 100 if pd.notna(s0) and pd.notna(s1) else np.nan})
    return pd.DataFrame(rows).assign(evidence_class="Derived")


def loan_mix_figure(t: pd.DataFrame, figd: Path) -> Path:
    order = ["households and personal", "agriculture", "trade", "manufacturing", "transport and communication",
             "hotels and tourism", "financial institutions", "other"]
    books = [("NMB", COLORS["B02"]), ("CRDB", COLORS["B01"]), ("NMB + CRDB", "#06452E"),
             ("BoT commercial banks", SYSTEM)]
    fig, ax = plt_subplots(figsize=(8.0, 4.4))
    y = np.arange(len(order))
    h = 0.2
    for i, (b, c) in enumerate(books):
        g = t[t["book"] == b].set_index("category").reindex(order)
        fy = int(g["latest_fy"].dropna().iloc[0])
        ax.barh(y + (i - 1.5) * h, g["share_latest"] * 100, height=h, color=c, label=f"{b} FY{fy}")
        ax.scatter(g["share_base"] * 100, y + (i - 1.5) * h, color="#1f1f1d", s=6, zorder=3,
                   label="FY2023" if i == 0 else None)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    style(ax, "Loan book by sector: latest year (bars) and FY2023 (dots)", "")
    ax.set_xlabel("percent of loans", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    return save(fig, figd / "ch13_text_loan_mix.png", _foot("bank IFRS 7 industry notes; BoT credit by economic "
                                                            "activity (ch02_loan_mix)", "Derived"))


# --- B2 -----------------------------------------------------------------------------------------------------------------
FEE_TYPES = ["transactions", "cards", "digital", "loan_fees", "fx", "other"]


FEES = Path("data/raw/manual_entries/fees/fee_transcribed.csv")


def net_fees(root: Path) -> pd.Series:
    """Net fee and commission income per bank-year (TZS), from the page-referenced transcription of the annual reports
    (edit pack A B27, prompt 20): the printed net line, else gross plus (negative) expense."""
    f = root / FEES
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    d["v"] = d["value"] * np.where(d["unit"].str.contains("thousand"), 1e3, 1e6)
    w = d.pivot_table(index=["bank", "fy"], columns="field", values="v", aggfunc="first")
    net = w.get("net_fee_commission_income")
    alt = w.get("fee_commission_income_gross") + w.get("fee_commission_expense")
    return net.combine_first(alt) if net is not None else alt


def governance(root: Path) -> pd.DataFrame:
    """Related-party lending and management pay as shares of capital, profit and lending (prompt 24, 3c.14; Blueprint
    ch. 13 governance). Inputs transcribed with page references in data/raw/manual_entries/governance_FY2025.csv.
    Definitions differ: CRDB's key management is the Board, MD, chief officers and directors reporting to the MD;
    NMB uses the IAS 24 wording and excludes Rabobank secondees (paid under the management contract)."""
    f = path("raw", root) / "manual_entries" / "governance_FY2025.csv"
    if not f.exists():
        return pd.DataFrame(columns=["bank", "measure", "value", "unit", "evidence_class"])
    g = pd.read_csv(f)
    rows = []
    for b, x in g.groupby("bank"):
        v = x.set_index("item")["value_tzs_m"].astype(float)
        cap_basis = x.set_index("item").loc["total_capital", "basis"]
        rp = v["loans_directors_kmp"] + v["loans_director_companies"]
        for m, val, unit in (
                ("related-party lending (directors, key management and their companies), TZS m", rp, "TZS m"),
                ("of which to companies associated with directors, TZS m", v["loans_director_companies"], "TZS m"),
                ("related-party lending / gross loans", rp / v["gross_loans"], "ratio"),
                (f"related-party lending / total regulatory capital ({cap_basis})", rp / v["total_capital"], "ratio"),
                ("key management compensation, TZS m", v["kmp_compensation"], "TZS m"),
                ("key management compensation / attributable profit", v["kmp_compensation"] / v["profit_attributable"],
                 "ratio"),
                ("directors' fees, TZS m", v["directors_fees"], "TZS m"),
                ("directors' fees / attributable profit", v["directors_fees"] / v["profit_attributable"], "ratio")):
            rows.append({"bank": b, "fiscal_year": 2025, "measure": m, "value": float(val), "unit": unit,
                         "evidence_class": "Derived" if unit == "ratio" else "Observed"})
    return pd.DataFrame(rows)


def agri_support(root: Path) -> pd.DataFrame:
    """CRDB agricultural support by category (prompt 24, 3c.13): concessional funding, guarantees, realised credit-risk
    transfer (claims paid). As disclosed in the FY2025 integrated report; blanks are not disclosed."""
    f = path("raw", root) / "manual_entries" / "crdb_agri_support_FY2025.csv"
    if not f.exists():
        return pd.DataFrame(columns=["category", "item", "value", "unit", "evidence_class"])
    d = pd.read_csv(f)
    return d.assign(disclosed=d["value"].notna(), evidence_class="Observed")


def fee_bases(root: Path) -> pd.DataFrame:
    """Issue 70 (prompt 21): CRDB FY2025 fee totals on their three bases (unaudited BoT, audited BoT, IFRS gross)."""
    f = root / "data" / "raw" / "manual_entries" / "fees" / "crdb_fee_bases.csv"
    if not f.exists():
        return pd.DataFrame(columns=["evidence_class"])
    d = pd.read_csv(f)
    d["value_tzs"] = d["value_tzs_m"] * 1e6
    return d.assign(evidence_class="Observed")


def fee_mix(root: Path) -> pd.DataFrame:
    c = _t(root, "ch04_4_1_composition.csv")
    nf = net_fees(root)
    c = c[c["bank_id"].isin(["B01", "B02"])].copy()
    o = _t(root, "ch04_4_1_nmb_other_fees.csv").groupby("fiscal_year")["other_total"].first()
    rows = []
    for r in c.itertuples():
        net = nf.get((NAME[r.bank_id], int(r.fiscal_year)), np.nan) if len(nf) else np.nan
        x = {"bank": NAME[r.bank_id], "fiscal_year": int(r.fiscal_year), "total_income": r.total_income,
             # B27: one basis every year: net fees / (NII + non-interest income, which holds net fees)
             "net_fee_income_tzs": net, "fee_share_of_income": net / r.total_income if r.total_income else np.nan,
             # gross IFRS fee-note total (the type mix); gross / total income mixes bases and is kept only as a flag
             "fee_income_tzs": r.total_income * r.fee_share_of_income if pd.notna(r.fee_share_of_income) else np.nan,
             "gross_fee_to_income_mixed_basis": r.fee_share_of_income}
        for f in FEE_TYPES:
            x[f"share_{f}"] = getattr(r, f"share_fee_income_{f}")
        x["nmb_other_fees_tzs"] = o.get(r.fiscal_year, np.nan) if r.bank_id == "B02" else np.nan
        rows.append(x)
    t = pd.DataFrame(rows)
    keep = t[[f"share_{f}" for f in FEE_TYPES] + ["fee_income_tzs"]].notna().any(axis=1)
    return t[keep].sort_values(["bank", "fiscal_year"]).assign(evidence_class="Derived")


def fee_mix_figure(t: pd.DataFrame, figd: Path) -> Path:
    g = t[t["bank"] == "CRDB"].set_index("fiscal_year")
    g = g[g[[f"share_{f}" for f in FEE_TYPES]].notna().any(axis=1)]
    lab = {"transactions": "transactions and accounts", "cards": "cards", "digital": "mobile, internet and agency",
           "loan_fees": "loan-related", "fx": "FX", "other": "other"}
    col = {"transactions": "#2a78d6", "cards": "#eda100", "digital": "#1baf7a", "loan_fees": "#6250d6",
           "fx": "#e87ba4", "other": "#9a9a94"}
    fig, (a1, a2) = plt_subplots(2, 1, figsize=(7.5, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    bottom = np.zeros(len(g))
    for f in FEE_TYPES:
        v = g[f"share_{f}"].fillna(0).to_numpy() * 100
        if v.any():
            a1.bar(g.index, v, bottom=bottom, color=col[f], label=lab[f])
            bottom += v
    style(a1, "CRDB: fee income by type", "percent of fee income")
    a1.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    a2.bar(g.index, g["fee_income_tzs"] / 1e9, color=COLORS["B01"], alpha=0.8, label="gross fee income, IFRS note "
           "(TZS bn)")
    a3 = a2.twinx()
    a3.plot(g.index, g["fee_share_of_income"] * 100, color="#1f1f1d", marker="o", ms=3, lw=1.2,
            label="net fees, % of total income")
    a3.set_ylim(0, max(40, float(g["fee_share_of_income"].max()) * 120))
    style(a2, "CRDB: fee income and its share of total income", "TZS bn")
    a3.set_ylabel("percent of income", fontsize=8)
    a3.spines["top"].set_visible(False)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left")
    return save(fig, figd / "ch13_text_fee_mix.png", _foot(
        "CRDB fee notes (IFRS, gross; inputs workbook; ch04_4_1_composition), type mapping D44 (LOW); net fees from "
        "the annual reports (fee_transcribed.csv) over NII + non-interest income. The BoT-format quarterly line 6.2 is "
        "used only for quarterly comparisons", "Derived"))


# --- B3 -----------------------------------------------------------------------------------------------------------------
def _split(root: Path, bank: str) -> tuple[float, pd.Timestamp | None]:
    ca = pd.read_csv(root / "config" / "corporate_actions.csv")
    x = ca[(ca["ticker"] == bank) & (ca["action"] == "split")]
    return (float(x["ratio"].iloc[0]), pd.Timestamp(x["effective_date"].iloc[0])) if len(x) else (1.0, None)


def implied_growth(root: Path) -> pd.DataFrame:
    ig = _t(root, "ch14_implied_growth.csv")
    v = _t(root, "ch09_valuations.csv").set_index(["valuation", "bank"])
    rows = []
    for r in ig.itertuples():
        x = v.loc[(r.valuation, r.bank)]
        ratio, eff = _split(root, r.bank)
        pre = eff is not None and pd.Timestamp(x["date"]) < eff
        f = ratio if pre else 1.0
        rows.append({"valuation": r.valuation, "date": x["date"], "bank": r.bank, "price": r.price, "coe": r.coe,
                     "base_g": r.base_g, "guard_g": r.g_upper_bound_guard, "value_at_guard_g": r.value_at_upper_bound,
                     "value_at_base_g": float(x["value_per_share"]), "per_share_basis":
                     f"pre-split (1:{int(ratio)} split {eff:%d %b %Y})" if pre else "as traded",
                     "price_post_split": r.price / f, "value_at_guard_g_post_split": r.value_at_upper_bound / f,
                     "value_at_base_g_post_split": float(x["value_per_share"]) / f, "status": r.status})
    return pd.DataFrame(rows).assign(evidence_class="Scenario")


# --- B4 -----------------------------------------------------------------------------------------------------------------
BREAKS = {  # indicator prefix: (trigger, breach when current goes ...)
    "T-bill rate": ("Net deposit franchise turns negative", "below"),
    "cost of risk": ("Credit cost normalises above the cycle mean", "above"),
    "deposit beta": ("Depositors start to reprice", "above"),
    "binding capital headroom": ("Capital headroom erodes", "below"),
    "price vs base value": ("Price falls below the base value (Sell no longer holds)", "below"),
}


def thesis_breaks(root: Path) -> pd.DataFrame:
    t = _t(root, "ch14_thesis_breaks.csv")
    rows = []
    for i, r in enumerate(t.itertuples()):
        trig, side = next(v for k, v in BREAKS.items() if r.indicator.startswith(k))
        margin = (r.current - r.threshold) if side == "below" else (r.threshold - r.current)
        rows.append({"row": f"r{i}", "bank": r.bank, "indicator": r.indicator, "trigger": trig, "current": r.current,
                     "threshold": r.threshold, "breach_when": f"current {side} threshold", "margin_to_breach": margin,
                     "status": r.status, "source_table": r.source})
    return pd.DataFrame(rows).assign(evidence_class="Derived")


# --- B5 / B6 ------------------------------------------------------------------------------------------------------------
def yield_minus_tbill(root: Path) -> pd.DataFrame:
    y = _t(root, "ch14_yield_minus_tbill.csv")
    return y[["bank", "fiscal_year", "dividend_yield", "tbill", "yield_minus_tbill", "tbill_series"]].assign(
        evidence_class="Derived")


def series_figure(t: pd.DataFrame, col: str, figd: Path, stem: str, title: str, ylabel: str, foot: str,
                  extra: str | None = None) -> Path:
    fig, ax = plt_subplots(figsize=(7.5, 3.8))
    for b in ("NMB", "CRDB"):
        g = t[t["bank"] == b].sort_values("fiscal_year")
        ax.plot(g["fiscal_year"], g[col] * 100, color=COLORS[BID[b]], lw=2, marker="o", ms=3, label=b)
    if extra:
        g = t[t["bank"] == "NMB"].sort_values("fiscal_year")
        ax.plot(g["fiscal_year"], g[extra] * 100, color=SYSTEM, lw=1, ls="--", label="T-bill rate (market)")
    ax.axhline(0, color="#1f1f1d", lw=0.6)
    style(ax, title, ylabel)
    ax.legend(fontsize=7, frameon=False)
    return save(fig, figd / f"{stem}.png", foot)


def real_excess(root: Path) -> pd.DataFrame:
    r = _t(root, "ch14_real_roe_coe.csv")
    r = r[r["fiscal_year"].between(2010, 2025)]
    return r[["bank", "fiscal_year", "roe", "coe", "inflation_imf", "real_roe", "real_coe", "real_excess"]].assign(
        evidence_class="Derived")


# --- B7 -----------------------------------------------------------------------------------------------------------------
def rore(root: Path) -> pd.DataFrame:
    s = _t(root, "ch07_rore_summary.csv")
    w = _t(root, "ch07_rore_windows.csv")
    rows = []
    for k in (3, 5):
        g = w[(w["bank_id"] == "B01") & (w["k"] == k)]
        both = g.dropna(subset=["rore_primary", "rore_adjusted"])
        ex = both[~both["stress_start_flag"].astype(bool)]
        for ser in ("rore_primary", "rore_adjusted"):
            x = s[(s["bank_id"] == "B01") & (s["k"] == k) & (s["series"] == ser)].iloc[0]
            stress = g.loc[g["stress_start_flag"].astype(bool), ser].dropna()
            med = float(x["median"])
            rows.append({"bank": "CRDB", "k": k, "series": ser, "label": x["label"], "n_windows": int(x["n_windows"]),
                         "median": med, "n_excl_stress_start": int(x["n_excl_stress_start"]),
                         "median_excl_stress_start": x["median_excl_stress_start"],
                         "stress_windows_below_median": int((stress < med).sum()),
                         "stress_windows_above_median": int((stress > med).sum()),
                         "latest_window": x["latest_window"], "latest": x["latest"],
                         "common_windows": len(both), "common_median": both[ser].median(),
                         "common_windows_excl_stress": len(ex), "common_median_excl_stress": ex[ser].median()})
    return pd.DataFrame(rows).assign(evidence_class="Derived")


# --- B8 -----------------------------------------------------------------------------------------------------------------
def usd_ytd(root: Path) -> pd.DataFrame:
    from tzbank.chapters.ch12_usd_implementation import _cfg

    cfg = _cfg(root)
    dep = _t(root, "ch12_depreciation.csv")
    e = _t(root, "ch12_expected_usd.csv")
    sp = _t(root, "ch12_returns_spans.csv")
    fx = _t(root, "ch12_fx.csv").assign(date=lambda d: pd.to_datetime(d["date"]))
    f0 = float(fx.loc[fx["date"] == "2025-12-31", "tzs_per_usd"].iloc[0])
    f1 = float(fx.sort_values("date")["tzs_per_usd"].iloc[-1])
    d1 = fx.sort_values("date")["date"].iloc[-1]
    move = f1 / f0 - 1
    rows = [{"bank": "", "measure": "TZS per USD, 31 Dec 2025", "value": f0, "unit": "TZS"},
            {"bank": "", "measure": f"TZS per USD, {d1:%d %b %Y}", "value": f1, "unit": "TZS"},
            {"bank": "", "measure": "TZS depreciation, 31 Dec 2025 to date", "value": move, "unit": "ratio"},
            {"bank": "", "measure": "effect on a USD holder of a TZS asset (1 / (1 + d) - 1)",
             "value": 1 / (1 + move) - 1, "unit": "ratio"}]
    cur = dep[dep["valuation"] == "CURR"].set_index("scenario")["depreciation"]
    for s, lab in (("history", "history assumption (base)"), ("realised_ytd", "realised 2026 move as year 1"),
                   ("realised_ytd_annualised", "realised 2026 move annualised as year 1")):
        rows.append({"bank": "", "measure": f"depreciation: {lab}", "value": float(cur[s]), "unit": "ratio"})
    for b in ("NMB", "CRDB"):
        x = sp[(sp["bank"] == b) & sp["period"].str.contains("tested-call")]
        if len(x):
            rows += [{"bank": b, "measure": "2026 YTD total return in TZS (31 Dec 2025 - 31 Oct 2026)",
                      "value": float(x["tzs_return"].iloc[0]), "unit": "ratio"},
                     {"bank": b, "measure": "2026 YTD total return in USD (31 Dec 2025 - 31 Oct 2026)",
                      "value": float(x["usd_return"].iloc[0]), "unit": "ratio"}]
        g = e[(e["valuation"] == "CURR") & (e["bank"] == b) & (e["position_usd"] == cfg["base_position_usd"])
              & np.isclose(e["participation"], cfg["base_participation"])].set_index("scenario")
        for s, lab in (("history", "history assumption (base)"), ("realised_ytd", "realised 2026 move as year 1"),
                       ("realised_ytd_annualised", "realised 2026 move annualised as year 1")):
            if s in g.index:
                rows.append({"bank": b, "measure": f"expected 12-month USD return: {lab}",
                             "value": float(g.loc[s, "expected_usd_return"]), "unit": "ratio"})
    return pd.DataFrame(rows).assign(evidence_class="Scenario")


# --- A5 -----------------------------------------------------------------------------------------------------------------
SPLIT_COLS = {"ch09_valuations.csv": ["rim_per_share", "ddm_per_share", "value_per_share", "price", "target_price_12m",
                                      "expected_dps_12m", "bear_value_per_share", "bull_value_per_share"],
              "ch11_scenarios.csv": ["value_per_share", "price", "eps_2025", "eps_2026", "eps_2027", "eps_2028"],
              "ch11_monte_carlo.csv": ["price", "p5", "p10", "p50", "p90", "p95", "mean"]}


def split_restated(root: Path) -> pd.DataFrame:
    ratio, eff = _split(root, "NMB")
    rows = []
    for name, cols in SPLIT_COLS.items():
        d = _t(root, name)
        d = d[(d["bank"] == "NMB") & (d["valuation"] == "HIST")]
        key = "scenario" if "scenario" in d else "variant" if "variant" in d else None
        for r in d.to_dict("records"):
            for c in cols:
                if c in r and pd.notna(r[c]):
                    rows.append({"source_table": name.removesuffix(".csv"), "valuation": "HIST", "bank": "NMB",
                                 "case": r[key] if key else "", "column": c, "reported": r[c], "ratio": ratio,
                                 "post_split": r[c] / ratio, "split_effective": f"{eff:%Y-%m-%d}"})
    return pd.DataFrame(rows).assign(evidence_class="Derived")


NOTE = """# Report text exports (chapter 13 support, prompt 17 Part B)

Generated by `tzbank.chapters.ch13_text_exports` (stage `market_ext`, after chapter 14).

## Research question
The figures the report text cites (blueprint chapters 2, 4, 5, 7, 10-12), reshaped from existing pipeline tables so
every one has a findings-pack id.

## Variable definitions
See the module docstring and `docs/VARIABLE_DICTIONARY.md`: shares are decimals of the total; `change_pp` and
`margin_to_breach` are in points (decimals for rates); `*_post_split` divide per-share values dated before the NMB
1:10 split (24 Aug 2026) by 10.

## Sources
ch02_loan_mix; ch04_4_1_composition, ch04_4_1_nmb_other_fees; ch14_implied_growth, ch09_valuations;
ch14_thesis_breaks; ch14_yield_minus_tbill; ch14_real_roe_coe; ch07_rore_summary, ch07_rore_windows;
ch12_fx, ch12_depreciation, ch12_expected_usd, ch12_returns_spans; config/corporate_actions.csv.

## Validation test
`tests/test_text_exports.py`.

## Results
{results}

## Evidence class
Derived (reshaped observed/derived figures), Scenario (implied growth, USD expectations). Interpretation is in the
report text, not here.

## Limitations
- Loan mix: the system series ends in {sys_fy}; bank books are FY2023 and FY2025 (Group, IFRS 7 notes).
- Fee types are label-matched (D44, LOW confidence); NMB 'other' fees are a printed-line total.
- B7 medians at k = 3 equal their ex-stress medians because the four stress-start windows split two below and two
  above the median; removing them leaves the middle of the distribution where it was (not a filter fault).
- B8 uses the realised 10-month move as the next 12 months' depreciation (and an annualised variant); it is a
  sensitivity, never the base (the base stays the history CAGR).
"""


def write_text_exports(root: Path) -> list[Path]:
    tab, figd = path("tables", root), path("figures", root)
    lm, fm = loan_mix(root), fee_mix(root)
    ig, tb, ym, rx = implied_growth(root), thesis_breaks(root), yield_minus_tbill(root), real_excess(root)
    ro, usd, sr = rore(root), usd_ytd(root), split_restated(root)
    tables = [(lm, "ch13_text_loan_mix.csv", "B1. Loan mix by sector, FY2023 vs latest, change in points", "Derived"),
              (fm, "ch13_text_fee_mix.csv", "B2. Fee income by type, fee income (TZS), fees / income; NMB other fees",
               "Derived"),
              (ig, "ch13_text_implied_growth.csv", "B3. Implied-growth guard: value at the COE - 4 pt guard and at "
               "base g (NMB 31 Dec 2025 also post-split)", "Scenario"),
              (tb, "ch13_text_thesis_breaks.csv", "B4. Thesis-break monitor: trigger, current, threshold, margin",
               "Derived"),
              (ym, "ch13_text_yield_minus_tbill.csv", "B5. Dividend yield less the T-bill rate", "Derived"),
              (rx, "ch13_text_real_excess.csv", "B6. Real ROE less real COE, FY2010-FY2025", "Derived"),
              (ro, "ch13_text_rore.csv", "B7. CRDB RORE, primary vs raise-adjusted, and on common windows", "Derived"),
              (usd, "ch13_text_usd_ytd.csv", "B8. Currency 2026 YTD and the realised-move USD sensitivity",
               "Scenario"),
              (sr, "ch13_split_restated.csv", "A5. NMB per-share figures restated for the 1:10 split", "Derived"),
              (fee_bases(root), "ch13_text_fee_bases.csv", "Issue 70: CRDB FY2025 fee totals on three bases",
               "Observed"),
              (governance(root), "ch13_text_governance.csv", "Related-party lending and management pay as shares of "
               "capital, profit and lending, FY2025 (prompt 24)", "Derived"),
              (agri_support(root), "ch13_text_agri_support.csv", "CRDB agricultural support: concessional funding, "
               "guarantees, claims paid (prompt 24)", "Observed")]
    files, man_t, man_f = [], [], []
    nf = path("notes", root) / "ch13_text.md"
    for df, name, desc, ev in tables:
        f = write_csv(df, tab / name, require_evidence=True)
        files.append(f)
        man_t.append({"path": f.relative_to(root).as_posix(), "description": desc, "evidence_class": ev,
                      "note": nf.relative_to(root).as_posix()})
    figs = [(loan_mix_figure(lm, figd), lm, "B1. Loan book by sector, latest year and FY2023"),
            (fee_mix_figure(fm, figd), fm[fm["bank"] == "CRDB"], "B2. CRDB fee mix and fee income"),
            (series_figure(ym, "yield_minus_tbill", figd, "ch13_text_yield_minus_tbill",
                           "Dividend yield less the T-bill rate", "percentage points",
                           _foot("ch14_yield_minus_tbill (BoT overall T-bill rate, annual)", "Derived"), "tbill"), ym,
             "B5. Dividend yield less the T-bill rate"),
            (series_figure(rx, "real_excess", figd, "ch13_text_real_excess", "Real ROE less real cost of equity",
                           "percentage points", _foot("ch14_real_roe_coe (IMF inflation)", "Derived")), rx,
             "B6. Real excess return")]
    for png, data, desc in figs:
        d = write_csv(data, png.with_name(f"{png.stem}_data.csv"), require_evidence=True)
        files += [png, d]
        man_f.append({"path": png.relative_to(root).as_posix(), "data": d.relative_to(root).as_posix(),
                      "description": desc, "evidence_class": "Derived", "note": nf.relative_to(root).as_posix()})
    res = []
    c = lm.set_index(["book", "category"])
    res.append(f"- B1: CRDB agriculture {c.loc[('CRDB', 'agriculture'), 'share_base']:.1%} (FY2023) -> "
               f"{c.loc[('CRDB', 'agriculture'), 'share_latest']:.1%} (FY2025); NMB trade "
               f"{c.loc[('NMB', 'trade'), 'share_base']:.1%} -> {c.loc[('NMB', 'trade'), 'share_latest']:.1%}; "
               f"NMB + CRDB households {c.loc[('NMB + CRDB', 'households and personal'), 'share_base']:.1%} -> "
               f"{c.loc[('NMB + CRDB', 'households and personal'), 'share_latest']:.1%}.")
    res.append(f"- B3: {len(ig)} rows; B4: {len(tb)} triggers, none breached: "
               f"{bool((tb['status'] == 'not breached').all())}; B7: {len(ro)} rows; B8: {len(usd)} measures; "
               f"A5: {len(sr)} restated values.")
    sys_fy = int(lm.loc[lm["book"] == "BoT commercial banks", "latest_fy"].max())
    files.append(write_text(NOTE.format(results="\n".join(res), sys_fy=sys_fy), nf))
    m = {"chapter": "13_text", "title": "Report text exports", "status": "complete", "hypotheses": [],
         "tables": man_t, "figures": man_f, "notes": [nf.relative_to(root).as_posix()],
         "inputs": ["outputs/tables/ch02_loan_mix.csv", "outputs/tables/ch04_4_1_composition.csv",
                    "outputs/tables/ch14_implied_growth.csv", "outputs/tables/ch14_thesis_breaks.csv",
                    "outputs/tables/ch14_yield_minus_tbill.csv", "outputs/tables/ch14_real_roe_coe.csv",
                    "outputs/tables/ch07_rore_summary.csv", "outputs/tables/ch12_expected_usd.csv"],
         "module": "tzbank.chapters.ch13_text_exports", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_text.json"))
    return files
