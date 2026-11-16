"""Processed tables (prompt 01 section 7; CLAUDE.md sections 3.5 and 3.8).

Precedence (CLAUDE.md §3.8): for NMB/CRDB the inputs workbook (audited primary, Group) replaces Phase 1 for the
same canonical variable, period and basis. The other five banks use Phase 1 only. Every replaced Phase 1 row is
written to reports/validation/provenance_replacements.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.clean.crosswalk import load_crosswalk
from tzbank.clean.normalise import PRIMARY_ENTITIES, load_canonical
from tzbank.config import bank_basis, banks, find_root, path, settings
from tzbank.ingest.inputs_workbook import load_interim
from tzbank.io import write_csv

LONG_COLUMNS = [
    "bank_id", "period_end", "period_type", "fiscal_year", "canonical_name", "category", "entity", "basis",
    "basis_rule", "value", "value_split_adj", "value_text", "unit", "status", "provenance", "doc_id",
    "source_document", "publication_date", "is_restated", "forecast_base_historical", "forecast_base_current",
    "evidence_class", "confidence", "source_sheet", "source_row_id", "source_variable", "definition", "notes",
    "unit_flag",
]
MATCH_KEY = ["bank_id", "canonical_name", "period_type", "period_end", "basis"]
# Variables kept on Bank-solo basis for a bank regardless of its reporting basis (decision D2).
BANK_BASIS_OVERRIDES = {"B02": {"tier1_ratio", "total_capital_ratio", "tier1_capital", "tier2_capital",
                                "total_regulatory_capital", "tier1_deductions", "rwa_credit", "rwa_market",
                                "rwa_operational", "rwa_total"}}


def apply_precedence(canon: pd.DataFrame, focus: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop Phase 1 rows for focus banks where the inputs workbook holds a value for the same key.

    Returns (kept rows, replaced Phase 1 rows with the replacing value attached).
    """
    iw = canon[(canon["provenance"] == "inputs_workbook") & canon["value"].notna()
               & canon["entity"].isin(PRIMARY_ENTITIES)]
    iw_orig = iw[~iw["is_restated"]].drop_duplicates(MATCH_KEY)
    iw_keys = set(map(tuple, iw[MATCH_KEY].astype(str).to_numpy()))
    p1 = canon["provenance"] == "phase1"
    key = canon[MATCH_KEY].astype(str).apply(tuple, axis=1)
    replaced_mask = p1 & canon["bank_id"].isin(focus) & key.isin(iw_keys)
    replaced = canon[replaced_mask].merge(
        iw_orig[[*MATCH_KEY, "value", "doc_id", "source_row_id"]].rename(
            columns={"value": "replacing_value", "doc_id": "replacing_doc_id", "source_row_id": "replacing_row_id"}),
        on=MATCH_KEY, how="left")
    replaced["relative_difference"] = (replaced["value"] - replaced["replacing_value"]) / replaced[
        "replacing_value"].abs()
    replaced["rule"] = "CLAUDE.md 3.8: inputs workbook over Phase 1 for NMB/CRDB"
    return canon[~replaced_mask].copy(), replaced


def bank_period_long(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build bank_period_long and the provenance-replacement list."""
    root = root or find_root()
    canon = load_canonical(root)
    # CLAUDE.md 3.8 for NMB/CRDB; D26 extends it to B03-B07 lines extracted into the inputs workbook (prompt 02).
    kept, replaced = apply_precedence(canon, settings(root)["precedence_banks"])
    kept = kept.sort_values(["bank_id", "canonical_name", "period_end", "period_type", "basis", "entity",
                             "publication_date", "source_row_id"], na_position="last")
    return kept[LONG_COLUMNS].reset_index(drop=True), replaced


def bank_year_wide(long: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """One row per bank-year (FY, bank's own entity, original not restated), canonical variables as columns.

    Basis rule: each bank's reporting basis (config/banks.yaml), except D2 overrides. Where only another basis is
    available the value is used and named in ``non_default_basis_vars`` (CLAUDE.md §3.5 flag column).
    """
    root = root or find_root()
    default = bank_basis(root)
    # D73: a year-end closing price held as a Market Data "AS AT" row dated 31 December is that fiscal year's
    # share_price_close (inputs workbook over Phase 1 through prov_rank below).
    ye = long[(long["canonical_name"] == "share_price_close") & (long["period_type"] == "AS AT")
              & (pd.to_datetime(long["period_end"]).dt.strftime("%m-%d") == "12-31")].copy()
    ye["period_type"] = "FY"
    long = pd.concat([long, ye], ignore_index=True)
    fy = long[(long["period_type"] == "FY") & long["entity"].isin(PRIMARY_ENTITIES) & ~long["is_restated"]
              & long["value"].notna() & long["category"].isna()].copy()
    want = []
    for b, c in zip(fy["bank_id"], fy["canonical_name"], strict=False):
        want.append("BANK" if c in BANK_BASIS_OVERRIDES.get(b, set()) else default[b])
    fy["preferred"] = np.array(want) == fy["basis"].to_numpy()
    fy["prov_rank"] = fy["provenance"].map({"inputs_workbook": 0, "phase1": 1})
    # Within Phase 1, a REPORTED value is preferred to one COMPUTED from secondary data.
    fy["status_rank"] = fy["status"].map({"REPORTED": 0}).fillna(1)
    fy = fy.sort_values(["bank_id", "fiscal_year", "canonical_name", "preferred", "prov_rank", "status_rank",
                         "publication_date", "source_row_id"],
                        ascending=[True, True, True, False, True, True, True, True], na_position="last")
    pick = fy.drop_duplicates(["bank_id", "fiscal_year", "canonical_name"])
    wide = pick.pivot(index=["bank_id", "fiscal_year"], columns="canonical_name", values="value")
    flags = (pick[~pick["preferred"]].groupby(["bank_id", "fiscal_year"])["canonical_name"]
             .apply(lambda s: ";".join(f"{v}:{b}" for v, b in zip(s, pick.loc[s.index, "basis"], strict=False))))
    prov = (pick.groupby(["bank_id", "fiscal_year"])["provenance"].apply(lambda s: ";".join(sorted(set(s)))))
    wide = wide.reset_index()
    wide.columns.name = None
    idx = pd.MultiIndex.from_frame(wide[["bank_id", "fiscal_year"]])
    wide["non_default_basis_vars"] = flags.reindex(idx).to_numpy()
    wide["provenance"] = prov.reindex(idx).to_numpy()
    wide["reporting_basis"] = wide["bank_id"].map(default)
    beg = {b["bank_id"]: bool(b.get("bank_equals_group", False)) for b in banks(root)}
    wide["bank_equals_group"] = wide["bank_id"].map(beg)  # D23
    wide["evidence_class"] = "Observed"
    front = ["bank_id", "fiscal_year", "reporting_basis", "bank_equals_group", "provenance", "non_default_basis_vars", "evidence_class"]
    return wide[front + sorted(c for c in wide.columns if c not in front)]


def market_daily(root: Path | None = None) -> pd.DataFrame:
    """Daily DSE data with Python-recomputed split adjustment and eligibility."""
    d = load_interim("Daily Market Data", root)
    out = pd.DataFrame({
        "date": d["Date"], "bank_id": d["Bank ID"], "ticker": d["Ticker"], "open": d["Open (TZS)"],
        "low": d["Low (TZS)"], "high": d["High (TZS)"], "close": d["Close (TZS)"], "turnover": d["Turnover (TZS)"],
        "volume": d["Volume (as traded)"], "split_factor": d["split_factor"],
        "split_factor_workbook": d["split_factor_workbook"], "adj_open": d["adj_open"], "adj_low": d["adj_low"],
        "adj_high": d["adj_high"], "adj_close": d["adj_close"], "adj_volume": d["adj_volume"],
        "implied_vwap_adj": d["implied_vwap_adj"], "traded_day": d["traded_day"], "data_flag": d.get("Data Flag"),
        "doc_id": d["Doc ID"], "publication_date": d["publication_date"],
        "eligible_historical": d["eligible_historical"], "eligible_current": d["eligible_current"],
    })
    out["evidence_class"] = np.where(out["split_factor"] != 1, "Derived", "Observed")
    return out.sort_values(["bank_id", "date"]).reset_index(drop=True)


def macro(root: Path | None = None) -> pd.DataFrame:
    """Macroeconomic Data with canonical names."""
    cw = load_crosswalk(root)
    m = load_interim("Macroeconomic Data", root)
    lut = dict(zip(cw.loc[cw["source_sheets"] == "Macroeconomic Data", "inputs_name"],
                   cw.loc[cw["source_sheets"] == "Macroeconomic Data", "canonical_name"], strict=False))
    m.insert(2, "canonical_name", m["Variable"].map(lut))
    m["evidence_class"] = np.where(m["Status"].astype(str).str.upper().eq("COMPUTED"), "Derived", "Observed")
    from tzbank.ingest.web_collected import fill_macro

    return fill_macro(m, root)  # D64: TO COLLECT rows filled from cited web sources (workbook keeps precedence)


def _subset(long: pd.DataFrame, sheet: str) -> pd.DataFrame:
    return long[long["source_sheet"] == sheet].reset_index(drop=True)


def write_processed(root: Path | None = None) -> list[Path]:
    """Write every processed table and the provenance-replacement report."""
    root = root or find_root()
    out_dir = path("processed", root)
    long, replaced = bank_period_long(root)
    wide = bank_year_wide(long, root)
    files = [
        write_csv(long, out_dir / "bank_period_long.csv", require_evidence=True),
        write_csv(wide, out_dir / "bank_year_wide.csv", require_evidence=True),
        write_csv(market_daily(root), out_dir / "market_daily.csv", require_evidence=True),
        write_csv(macro(root), out_dir / "macro.csv", require_evidence=True),
        write_csv(_subset(long, "Interim Results"), out_dir / "interim_long.csv", require_evidence=True),
        write_csv(_subset(long, "Quarterly Results"), out_dir / "quarterly_long.csv", require_evidence=True),
        write_csv(_subset(long, "Regulatory Capital"), out_dir / "capital_long.csv", require_evidence=True),
        write_csv(load_interim("Sources", root), out_dir / "sources.csv"),
        write_csv(load_interim("Assumptions", root), out_dir / "assumptions.csv"),
        write_csv(replaced, path("validation", root) / "provenance_replacements.csv"),
    ]
    return files
