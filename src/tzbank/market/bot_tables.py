"""Processed BoT tables (prompt 02f section 3; blueprint §2 rates and cost of funds, §6 sector totals, chapters 1-3,
4.2 and 6).

Built from ``data/interim/bot_long.csv`` (the owner's research database merged under D40, plus template files) and,
below the owner's precedence, the code-extracted BoT Annual Report tables (``data/interim/bot_code_extracted.csv``,
D39; only values that passed their gates and are not already held):

- ``rates_quarterly.csv``: quarter-end values (March, June, September, December) of every rate series;
- ``rates_annual.csv``: December values of every rate series (owner's annual rows are December values, D40), plus the
  code-extracted financial-year averages as separately labelled series;
- ``sector_totals.csv``: commercial-bank assets, liabilities, deposits and lending (December), financial soundness
  indicators and sector structure (end-June), and code-extracted licensed-bank counts and soundness ratios;
- ``credit_by_activity.csv``: commercial-bank lending by economic activity (December levels and shares).

Rates are decimals (CLAUDE.md 3.4); money stays in TZS million as published, with the unit column. One row per
series and period; ``precedence`` shows the source tier.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

RATE_TOPICS = {"INTEREST_RATES", "TBILL_AUCTION_YIELDS", "DISCOUNT_AND_RESERVE_RATIO"}
SECTOR_TOPICS = {"BANK_ASSETS", "BANK_LIABILITIES", "BANK_DEPOSITS", "LENDING_AND_SECURITIES",
                 "FINANCIAL_SOUNDNESS", "SECTOR_STRUCTURE"}
OUT_COLUMNS = ["topic", "series_id", "series_name", "period_end", "year", "frequency_basis", "value", "unit",
               "source_document", "page_table", "precedence", "evidence_class"]


def load(root: Path) -> pd.DataFrame:
    b = pd.read_csv(path("interim", root) / "bot_long.csv", low_memory=False)
    b["period_end"] = pd.to_datetime(b["period_end"], errors="coerce")
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    b["topic"] = b["series_id"].str.extract(r"^BOTDB_(.*?)__", expand=False).fillna(b["category"].str.upper())
    b["base_id"] = b["series_id"].str.replace(r"_(MONTHLY|QUARTERLY|ANNUAL|EVENT|FY_JUNE)$", "", regex=True)
    b["precedence"] = b["notes"].astype(str).str.extract(r"\((v2|v1 only)\)", expand=False).map(
        {"v2": "owner database v2", "v1 only": "owner database v1"}).fillna("template file")
    return b[b["value"].notna() & b["period_end"].notna()]


def _pick(df: pd.DataFrame, months: set[int], basis: str) -> pd.DataFrame:
    """One value per base series and period end: sub-annual rows first, then annual rows."""
    d = df[df["period_end"].dt.month.isin(months)].copy()
    d["_rank"] = d["frequency"].map({"monthly": 0, "quarterly": 1, "annual": 2, "event": 3}).fillna(4)
    d = d.sort_values(["base_id", "period_end", "_rank"]).drop_duplicates(["base_id", "period_end"])
    return pd.DataFrame({"topic": d["topic"].str.lower(), "series_id": d["base_id"], "series_name": d["series_name"],
                         "period_end": d["period_end"].dt.date, "year": d["period_end"].dt.year,
                         "frequency_basis": basis, "value": d["value"], "unit": d["unit"],
                         "source_document": d["source_document"], "page_table": d["page_table"],
                         "precedence": d["precedence"], "evidence_class": "Observed"})


def code_rows(root: Path, kind: str) -> pd.DataFrame:
    """Gated, new code-extracted values (D39) as processed rows."""
    f = path("interim", root) / "bot_code_extracted.csv"
    if not f.exists():
        return pd.DataFrame(columns=OUT_COLUMNS)
    c = pd.read_csv(f, dtype={"table": str, "report_fy": str, "period": str})
    c = c[c["used"].astype(str).eq("True") & c["db_match"].eq("new") & c["value"].notna()]
    rates = c["title"].str.contains("Interest|Bond", case=False)
    c = c[rates] if kind == "rates" else c[~rates]
    if c.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    c = c.copy()
    noyear = ~c["period"].astype(str).str.contains(r"\d{4}")
    c.loc[noyear, "row_label"] = c.loc[noyear, "row_label"] + " - " + c.loc[noyear, "period"].astype(str)
    c.loc[noyear, "period"] = c.loc[noyear, "report_fy"]  # e.g. Licensed Banks columns: as at the report's June
    fy = c["period"].astype(str).str.extract(r"(\d{4})/(\d\d)")
    yr = pd.to_numeric(fy[0], errors="coerce") + 1
    yr = yr.fillna(pd.to_numeric(c["period"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"))
    month = fy[0].notna().map({True: 6, False: 12})
    ok = yr.notna()
    c, yr, month = c[ok], yr[ok].astype(int), month[ok]
    pe = pd.to_datetime(dict(year=yr, month=month, day=30 * (month == 6) + 31 * (month == 12)), errors="coerce")
    sid = ("BOTCODE_" + (c["title"] + "_" + c["row_label"]).str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True)
           .str.strip("_").str[:90])
    return pd.DataFrame({
        "topic": "code_" + kind, "series_id": sid, "series_name": c["title"] + ": " + c["row_label"],
        "period_end": pe.dt.date, "year": pe.dt.year,
        "frequency_basis": [("BoT financial-year figure" if m == 6 else "calendar-year figure") + f" ({p})"
                            for m, p in zip(month, c["period"], strict=True)],
        "value": c["value"] / 100 if kind == "rates" else c["value"],
        "unit": "ratio" if kind == "rates" else c["unit"],
        "source_document": "BoT Annual Report " + c["report_fy"] + " (" + c["file"] + ")",
        "page_table": "table " + c["table"] + ", p." + c["pdf_page"].astype(str),
        "precedence": "code-extracted (D39, below owner)", "evidence_class": "Observed"})[
        lambda d: d["year"].notna()].drop_duplicates(["series_id", "period_end"], keep="last")


def build(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or find_root()
    b = load(root)
    rates = b[b["topic"].isin(RATE_TOPICS) | b["category"].isin(["rates", "reserve_requirements"])]
    sector = b[b["topic"].isin(SECTOR_TOPICS)]
    credit = b[b["topic"].eq("CREDIT_BY_ACTIVITY")]
    out = {
        "rates_quarterly": _pick(rates[rates["frequency"].isin(["monthly", "quarterly"])], {3, 6, 9, 12},
                                 "quarter-end"),
        "rates_annual": pd.concat([_pick(rates[rates["frequency"] != "event"], {12}, "December value"),
                                   code_rows(root, "rates")], ignore_index=True),
        "sector_totals": pd.concat([
            _pick(sector[~sector["topic"].isin(["FINANCIAL_SOUNDNESS", "SECTOR_STRUCTURE"])], {12},
                  "December value"),
            _pick(sector[sector["topic"].isin(["FINANCIAL_SOUNDNESS", "SECTOR_STRUCTURE"])], {6}, "end-June value"),
            code_rows(root, "sector")], ignore_index=True),
        "credit_by_activity": _pick(credit, {12}, "December value"),
    }
    out["rates_annual"] = pd.concat([out["rates_annual"], policy_rate(out["rates_annual"], root)], ignore_index=True)
    return {k: v.reindex(columns=OUT_COLUMNS).sort_values(["series_id", "period_end"]).reset_index(drop=True)
            for k, v in out.items()}


CBR_START = 2024  # BoT adopted the Central Bank Rate framework in January 2024 (D42, corrected in prompt 03b)


def policy_rate(annual: pd.DataFrame, root: Path) -> pd.DataFrame:
    """D42 policy-rate series: BoT discount rate (December) before the Central Bank Rate, the CBR after.

    The CBR is held only in the inputs workbook's Macroeconomic Data (valuation dates); years without either stay
    missing."""
    d = annual[annual["series_id"] == "BOTDB_INTEREST_RATES__DOM_DISCOUNT"]
    d = d[d["year"] < CBR_START].assign(series_id="POLICY_RATE_D42",
                                        series_name="Policy rate (D42): BoT discount rate before the CBR",
                                        precedence=d["precedence"] + "; D42 policy-rate series")
    rows = [d]
    c = path("interim", root) / "bot_code_extracted.csv"
    if c.exists():
        x = pd.read_csv(c, dtype={"period": str})
        x = x[(x["title"] == "Central Bank Rate (text)") & x["used"].astype(str).eq("True")]
        x = x[x["period"].str.match(r"^\d{4}-12$")].drop_duplicates("period")
        pe = pd.to_datetime(x["period"] + "-31")
        rows.append(pd.DataFrame({
            "topic": "interest_rates", "series_id": "POLICY_RATE_D42",
            "series_name": "Policy rate (D42): Central Bank Rate (BoT Annual Report text)", "period_end": pe.dt.date,
            "year": pe.dt.year, "frequency_basis": "rate in force throughout the BoT financial year containing December",
            "value": x["value"] / 100, "unit": "ratio",
            "source_document": "BoT Annual Report " + x["report_fy"] + " (" + x["file"] + ")",
            "page_table": "text, p." + x["pdf_page"].astype(str), "precedence": "code-extracted (D39); D42 policy-rate",
            "evidence_class": "Observed"}))
    f = path("interim", root) / "inputs_macroeconomic_data.csv"
    if f.exists():
        m = pd.read_csv(f)
        m = m[m["Variable"].astype(str).str.contains("Central Bank Rate") & m["value_numeric"].notna()]
        if len(m):
            pe = pd.to_datetime(m["As At (Valuation Date)"])
            rows.append(pd.DataFrame({
                "topic": "interest_rates", "series_id": "POLICY_RATE_D42",
                "series_name": "Policy rate (D42): Central Bank Rate", "period_end": pe.dt.date, "year": pe.dt.year,
                "frequency_basis": "rate in force at the valuation date (" + m["Observation / Auction Date"].astype(
                    str) + ")", "value": m["value_numeric"], "unit": "ratio",
                "source_document": "inputs workbook, Macroeconomic Data (" + m["Doc ID"].astype(str) + ")",
                "page_table": m["Row ID"], "precedence": "inputs workbook; D42 policy-rate series",
                "evidence_class": "Observed"}))
    return pd.concat(rows, ignore_index=True)


def summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        rows.append({"table": name, "series": df["series_id"].nunique(), "rows": len(df),
                     "first_year": int(df["year"].min()) if len(df) else None,
                     "last_year": int(df["year"].max()) if len(df) else None, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def long_bond_fallback(root: Path) -> dict[int, dict]:
    """Year -> December 10-year Treasury bond rate (BoT rate structure A3.13), the longest tenor the rate table shows
    for 2008-2011 (D34 has no auction of 15 years or more before 2013)."""
    f = path("interim", root) / "bot_long.csv"
    if not f.exists():
        return {}
    b = pd.read_csv(f, low_memory=False)
    b = b[b["series_id"].astype(str).str.match(r"BOTDB_INTEREST_RATES__DOM_TREASURY_BOND_(\d+)_YEAR_(QUARTERLY|"
                                                r"MONTHLY|ANNUAL)$")]
    b["tenor"] = b["series_id"].str.extract(r"BOND_(\d+)_YEAR", expand=False).astype(int)
    b["pe"] = pd.to_datetime(b["period_end"])
    b = b[b["pe"].dt.month == 12]
    b["_r"] = b["frequency"].map({"monthly": 0, "quarterly": 1, "annual": 2}).fillna(3)
    b["_t"] = b["series_id"].str.extract(r"BOND_(\d+)_YEAR", expand=False)
    b = b.sort_values("_r").drop_duplicates(["_t", "pe"])  # sub-annual December rows first, then annual
    out = {}
    for y, g in b.groupby(b["pe"].dt.year):
        by_t = dict(zip(g["tenor"], g["value"].astype(float), strict=False))
        top = g[g["tenor"] == g["tenor"].max()].iloc[0]
        out[int(y)] = {"waytm": float(top["value"]), "tenor": int(top["tenor"]), "source": top["source_document"],
                       "page": top["page_table"],
                       "spread_15_10": by_t[15] - by_t[10] if 15 in by_t and 10 in by_t else None}
    return out


def write_bot_tables(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    if not (path("interim", root) / "bot_long.csv").exists():
        return []
    t = build(root)
    p = path("processed", root)
    files = [write_csv(df, p / f"{k}.csv", require_evidence=True) for k, df in t.items()]
    files.append(write_csv(summary(t), path("reports", root) / "bot_tables_summary.csv", require_evidence=True))
    return files


__all__ = ["build", "long_bond_fallback", "write_bot_tables", "re"]
