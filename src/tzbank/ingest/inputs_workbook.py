"""NMB/CRDB valuation inputs workbook ingest (prompt 01 sections 3.2-3.4; CLAUDE.md section 3.3).

Cells holding Excel formulas are discarded on read: publication dates, eligibility flags and split-adjusted market
fields are recomputed here from the Sources sheet, the valuation dates in config/settings.yaml and
config/corporate_actions.csv. Excel's cached formula results are never read.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.config import find_root, path, valuation_dates
from tzbank.io import write_csv

SHEETS = ["Historical Financials", "Interim Results", "Quarterly Results", "Regulatory Capital", "Market Data", "Macroeconomic Data",
          "Peer Banks", "Assumptions", "Sources", "Phase1 Source Map", "Daily Market Data"]

# Columns that are formulas in the workbook; recomputed in Python and therefore dropped from the raw read.
FORMULA_COLUMNS = {"Publication Date", "Eligible: Historical", "Eligible: Current", "Adj. Low", "Adj. High",
                   "Adj. Close", "Adj. Volume", "Implied VWAP (Adj.)", "Traded Day"}

# Period column used for the "period end on or before the valuation date" test.
PERIOD_COLUMN = {"Historical Financials": "Period End", "Interim Results": "Period End",
                 "Quarterly Results": "Period End",
                 "Regulatory Capital": "Period End", "Market Data": "Period End",
                 "Macroeconomic Data": "Observation / Auction Date", "Daily Market Data": "Date"}


def slug(sheet: str) -> str:
    """'Historical Financials' -> 'historical_financials'."""
    return re.sub(r"[^a-z0-9]+", "_", sheet.lower()).strip("_")


def interim_file(sheet: str, root: Path | None = None) -> Path:
    """data/interim/inputs_<slug>.csv for a workbook sheet."""
    return path("interim", root or find_root()) / f"inputs_{slug(sheet)}.csv"


def read_sheet(wb: openpyxl.Workbook, sheet: str) -> tuple[pd.DataFrame, int, set[str]]:
    """Read one sheet with formulas blanked. Returns (frame, formula cells discarded, columns holding formulas)."""
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else None for h in rows[0]]
    keep = [i for i, h in enumerate(header) if h]
    data, n_formula, fcols = [], 0, set()
    for r in rows[1:]:
        vals = []
        for i in keep:
            v = r[i] if i < len(r) else None
            if isinstance(v, str) and v.startswith("="):
                n_formula += 1
                fcols.add(header[i])
                v = _FORMULA if header[i] == "Value" else None
            vals.append(v)
        if any(v is not None for v in vals):
            data.append(vals)
    df = pd.DataFrame(data, columns=[header[i] for i in keep], dtype=object)
    return df, n_formula, fcols


_FORMULA = "__FORMULA__"

# Market Data rows whose Value is a live Excel formula over Daily Market Data; recomputed in Python.
MARKET_STAT_VARIABLES = {"52-Week High", "52-Week Low", "12-Month Traded Value", "12-Month Traded Volume",
                         "Days with Trades (12 months)"}


def market_window_stat(daily: pd.DataFrame, bank_id: str, variable: str, as_at: pd.Timestamp) -> float:
    """52-week / 12-month statistic over the 364 days ending ``as_at`` (same window as the workbook)."""
    d = daily[(daily["Bank ID"] == bank_id) & (daily["Date"] > as_at - pd.Timedelta(days=364))
              & (daily["Date"] <= as_at)]
    traded = d[pd.to_numeric(d["Volume (as traded)"], errors="coerce") > 0]
    if variable == "52-Week High":
        return float(traded["adj_high"].max())
    if variable == "52-Week Low":
        return float(traded["adj_low"].min())
    if variable == "12-Month Traded Value":
        return float(pd.to_numeric(d["Turnover (TZS)"], errors="coerce").sum())
    if variable == "12-Month Traded Volume":
        return float(d["adj_volume"].sum())
    if variable == "Days with Trades (12 months)":
        return float(len(traded))
    raise KeyError(variable)


def recompute_market_stats(md: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Replace formula Values on the Market Data sheet with Python recomputations."""
    md = md.copy()
    md["value_recomputed_in_python"] = md["value_raw"].eq(_FORMULA)
    bad = md["value_recomputed_in_python"] & ~md["Variable"].isin(MARKET_STAT_VARIABLES)
    if bad.any():
        raise ValueError(f"Market Data: unexpected formula rows {md.loc[bad, 'Row ID'].tolist()}")
    for i in md.index[md["value_recomputed_in_python"]]:
        md.at[i, "value_numeric"] = market_window_stat(daily, md.at[i, "Bank ID"], md.at[i, "Variable"],
                                                       pd.Timestamp(md.at[i, "Period End"]))
        md.at[i, "value_raw"] = repr(md.at[i, "value_numeric"])
    return md


def publication_dates(doc_ids: pd.Series, sources: pd.DataFrame) -> pd.Series:
    """Look up each Doc ID's publication date in Sources (one date per document). Blueprint §6; CLAUDE.md §3.3."""
    col = "publication_date" if "publication_date" in sources.columns else "Publication Date"
    lut = dict(zip(sources["Doc ID"], pd.to_datetime(sources[col], errors="coerce"), strict=False))
    return pd.to_datetime(doc_ids.map(lambda d: lut.get(d) if isinstance(d, str) else pd.NaT))


def publication_date_types(doc_ids: pd.Series, sources: pd.DataFrame) -> pd.Series:
    """Each Doc ID's publication_date_type (actual / estimated / lower_bound / missing)."""
    if "publication_date_type" not in sources.columns:
        return pd.Series(None, index=doc_ids.index, dtype=object)
    lut = dict(zip(sources["Doc ID"], sources["publication_date_type"], strict=False))
    return doc_ids.map(lambda d: lut.get(d) if isinstance(d, str) else None)


def eligibility(pub: pd.Series, period: pd.Series | None, valuation_date: pd.Timestamp,
                pub_type: pd.Series | None = None) -> pd.Series:
    """'Y' when the publication date (and period end, if given) are on or before the valuation date; else 'N'.

    Rows with no publication date, or only a lower bound (D35), are 'N'. Decision D1 cut-off rule.
    """
    ok = pub.notna() & (pub <= valuation_date)
    if pub_type is not None:
        ok &= ~pub_type.eq("lower_bound")
    if period is not None:
        per = pd.to_datetime(period, errors="coerce")
        ok &= per.isna() | (per <= valuation_date)
    return ok.map({True: "Y", False: "N"})


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in {"Period End", "Date", "As At (Valuation Date)", "Observation / Auction Date"}:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Value" in df.columns:
        df["value_raw"] = df["Value"].map(lambda v: None if v is None else (v.date().isoformat() if hasattr(v, "date")
                                                                             else str(v)))
        df["value_numeric"] = pd.to_numeric(
            df["Value"].map(lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None),
            errors="coerce")
        df = df.drop(columns=["Value"])
    return df


def read_inputs_workbook(file: Path | None = None, root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the ten sheets, recompute publication dates and eligibility. Blueprint §6 (inputs workbook v0.6)."""
    root = root or find_root()
    file = file or path("inputs_workbook", root)
    wb = openpyxl.load_workbook(file, read_only=True, data_only=False)
    vd = valuation_dates(root)
    frames: dict[str, pd.DataFrame] = {}
    formula_counts: dict[str, int] = {}
    for sheet in SHEETS:
        df, nf, fcols = read_sheet(wb, sheet)
        formula_counts[sheet] = nf
        unexpected = fcols - FORMULA_COLUMNS - ({"Value"} if sheet == "Market Data" else set())
        if unexpected:
            raise ValueError(f"{sheet}: formulas in unexpected columns {sorted(unexpected)}")
        df = df.drop(columns=sorted(fcols - {"Value"}))
        frames[sheet] = _typed(df)
    wb.close()
    sources = frames["Sources"]
    from tzbank.pit.pubdates import resolve_sources

    for c in ["Publication Date", "Period End", "Actual Publication Date", "FS Board Approval Date",
              "Auditor Report Date", "Legacy Publication Date (pre-D16)", "Estimated Publication Date",
              "Lower-Bound Publication Date"]:
        if c in sources.columns:
            sources[c] = pd.to_datetime(sources[c], errors="coerce")
    # D16: the publication date is recomputed from the evidence columns, never taken from the typed column.
    sources = resolve_sources(sources) if "Period End" in sources.columns else sources.assign(
        publication_date=sources["Publication Date"], publication_date_type="legacy")
    frames["Sources"] = sources
    for sheet, df in frames.items():
        if sheet in {"Sources", "Assumptions"}:
            continue
        if sheet == "Daily Market Data":
            # Daily DSE prices are public on the trading day; the document-level date of DSE-DAILY-2025-2026
            # (2026-10-30) predates rows up to 2026-11-05, so the row date is the publication date.
            pub = pd.to_datetime(df["Date"])
            ptype = None
        else:
            col = "Mapped Doc ID" if sheet == "Phase1 Source Map" else "Doc ID"
            pub = publication_dates(df[col], sources)
            ptype = publication_date_types(df[col], sources)
        df["publication_date"] = pub
        per = df[PERIOD_COLUMN[sheet]] if sheet in PERIOD_COLUMN else None
        df["eligible_historical"] = eligibility(pub, per, vd["historical"], ptype)
        df["eligible_current"] = eligibility(pub, per, vd["current"], ptype)
    frames["Daily Market Data"] = recompute_daily(frames["Daily Market Data"], root=root)
    frames["Market Data"] = recompute_market_stats(frames["Market Data"], frames["Daily Market Data"])
    frames["_formula_counts"] = pd.DataFrame({"sheet": list(formula_counts), "formula_cells_discarded":
                                              list(formula_counts.values())})
    return frames


def recompute_daily(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Recompute split-adjusted daily fields from config/corporate_actions.csv (prompt 01 section 3.4)."""
    from tzbank.clean.splits import apply_split_adjustment, load_corporate_actions

    df = df.rename(columns={"Split Factor": "split_factor_workbook"})
    actions = load_corporate_actions(root)
    df = apply_split_adjustment(df, actions, date_col="Date", bank_col="Bank ID",
                                price_cols={"Low (TZS)": "adj_low", "High (TZS)": "adj_high", "Close (TZS)": "adj_close",
                                            "Open (TZS)": "adj_open"},
                                volume_cols={"Volume (as traded)": "adj_volume"})
    turnover = pd.to_numeric(df["Turnover (TZS)"], errors="coerce")
    df["implied_vwap_adj"] = (turnover / df["adj_volume"]).where(df["adj_volume"] > 0)
    df["traded_day"] = (pd.to_numeric(df["Volume (as traded)"], errors="coerce") > 0).astype(int)
    return df


def write_inputs_interim(root: Path | None = None) -> list[Path]:
    """Write one data/interim/inputs_<sheet>.csv per sheet."""
    root = root or find_root()
    frames = read_inputs_workbook(root=root)
    out = []
    for sheet in SHEETS:
        out.append(write_csv(frames[sheet], interim_file(sheet, root)))
    out.append(write_csv(frames["_formula_counts"], path("interim", root) / "inputs_formula_cells_discarded.csv"))
    return out


def load_interim(sheet: str, root: Path | None = None) -> pd.DataFrame:
    """Read an interim inputs CSV with date columns parsed."""
    df = pd.read_csv(interim_file(sheet, root), dtype={"Doc ID": str, "Row ID": str, "Bank ID": str, "value_raw": str},
                     low_memory=False)
    for c in ["Period End", "Date", "publication_date", "Publication Date", "As At (Valuation Date)",
              "Observation / Auction Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df
