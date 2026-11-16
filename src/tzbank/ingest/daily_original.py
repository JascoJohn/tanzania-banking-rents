"""Original daily trading file (NMB_CRDB_52wk_and_trading_value.xlsx): ingest and cross-check against the inputs
workbook's Daily Market Data sheet (blueprint chapters 1 and 12; data layer). Formula cells are ignored and the
split adjustment is recomputed from config/corporate_actions.csv.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.clean.splits import apply_split_adjustment, load_corporate_actions
from tzbank.config import find_root, path
from tzbank.io import write_csv

SHEETS = {"NMB Data": ("B02", "NMB"), "CRDB Data": ("B01", "CRDB")}
COLS = ["Date", "Open (TZS)", "Low (TZS)", "High (TZS)", "Close (TZS)", "Turnover (TZS)", "Volume"]


def read_original(file: Path | None = None, root: Path | None = None) -> pd.DataFrame:
    """Typed values only (formula columns dropped), one row per bank-date."""
    wb = openpyxl.load_workbook(file or path("daily_trading_original", root), read_only=True, data_only=False)
    parts = []
    for sheet, (bank, ticker) in SHEETS.items():
        rows = list(wb[sheet].iter_rows(values_only=True))
        hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
        idx = [hdr.index(c) for c in COLS]
        data = [[r[i] for i in idx] for r in rows[1:] if r[0] is not None]
        df = pd.DataFrame(data, columns=COLS)
        df["bank_id"], df["ticker"] = bank, ticker
        parts.append(df)
    wb.close()
    out = pd.concat(parts, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    out = apply_split_adjustment(out, load_corporate_actions(root), "Date", "bank_id",
                                 {"Close (TZS)": "adj_close", "Low (TZS)": "adj_low", "High (TZS)": "adj_high"},
                                 {"Volume": "adj_volume"})
    out["evidence_class"] = "Observed"
    return out.sort_values(["bank_id", "Date"]).reset_index(drop=True)


def crosscheck(original: pd.DataFrame, workbook_daily: pd.DataFrame) -> pd.DataFrame:
    """Rows where close, turnover or volume differ between the two daily sources (or a date is in one only)."""
    w = workbook_daily.rename(columns={"Bank ID": "bank_id", "Volume (as traded)": "Volume"})
    m = original.merge(w[["bank_id", "Date", "Close (TZS)", "Turnover (TZS)", "Volume"]], on=["bank_id", "Date"],
                       how="outer", suffixes=("_original", "_workbook"), indicator=True)
    diff = (m["_merge"] != "both")
    for c in ["Close (TZS)", "Turnover (TZS)", "Volume"]:
        a = pd.to_numeric(m[f"{c}_original"], errors="coerce")
        b = pd.to_numeric(m[f"{c}_workbook"], errors="coerce")
        diff |= (m["_merge"] == "both") & ~((a == b) | (a.isna() & b.isna()))
    out = m[diff].rename(columns={"_merge": "present_in"})
    out["present_in"] = out["present_in"].astype(str).replace({"left_only": "original_only",
                                                               "right_only": "workbook_only"})
    return out.reset_index(drop=True)


def write_daily_original(root: Path | None = None) -> list[Path]:
    """Write data/interim/daily_trading_original.csv."""
    root = root or find_root()
    return [write_csv(read_original(root=root), path("interim", root) / "daily_trading_original.csv")]
