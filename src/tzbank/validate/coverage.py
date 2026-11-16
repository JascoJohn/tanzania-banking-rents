"""Coverage and conflict checks (prompt 01 sections 6.3-6.4; CLAUDE.md sections 3.7-3.8)."""

from __future__ import annotations

import pandas as pd

from tzbank.clean.normalise import PRIMARY_ENTITIES

VALUE_STATUSES = {"REPORTED", "COMPUTED", "CALCULATED", "DERIVED"}
INPUT_KEY = ["Bank ID", "Entity", "Basis", "Period Type", "Period End", "Variable", "Doc ID"]


def duplicate_keys(sheets: dict[str, pd.DataFrame], phase1: pd.DataFrame) -> pd.DataFrame:
    """Duplicate keys in source tables. Restatements differ by Doc ID and are not duplicates.

    ``critical`` is True when two or more rows sharing a key carry a value (ambiguous data). Duplicated empty
    placeholders (e.g. two TO COLLECT rows) are listed as non-critical. Phase 1 keys include the source's stated
    definition, because Phase 1 records some ratios under one name with different definitions (e.g. LDR on gross
    vs net loans).
    """
    rows = []
    for sheet, df in sheets.items():
        k = [c for c in INPUT_KEY if c in df.columns]
        if sheet == "Daily Market Data":
            k = ["Bank ID", "Date"]
        elif sheet == "Macroeconomic Data":
            k = ["Variable", "As At (Valuation Date)", "Doc ID"]
        elif sheet == "Peer Banks":
            k = ["Peer", "As At (Valuation Date)", "Variable"]
        elif sheet in {"Sources"}:
            k = ["Doc ID"]
        elif sheet in {"Assumptions"}:
            k = ["ID"]
        elif sheet == "Phase1 Source Map":
            k = ["Bank ID", "Phase 1 Source Label"]
        d = df[df.duplicated(k, keep=False)].copy()
        has_value = d["value_numeric"].notna() if "value_numeric" in d.columns else pd.Series(True, index=d.index)
        n_vals = has_value.groupby([d[c].astype(str) for c in k]).transform("sum")
        for i, r in d.iterrows():
            rows.append({"table": sheet, "key": "|".join(str(r[c]) for c in k), "row_id": r.get("Row ID"),
                         "has_value": bool(has_value[i]), "critical": bool(n_vals[i] > 1)})
    k = ["bank_id", "year", "sheet", "variable", "category", "definition"]
    d = phase1[phase1["sheet"] != "RAW_DATA"]
    d = d[d.duplicated(k, keep=False)].copy()
    n_vals = d["value"].notna().groupby([d[c].astype(str) for c in k]).transform("sum")
    for i, r in d.iterrows():
        rows.append({"table": "phase1_long", "key": "|".join(str(r[c]) for c in k),
                     "row_id": f"{r['sheet']}:{r['row_number']}", "has_value": bool(pd.notna(r["value"])),
                     "critical": bool(n_vals[i] > 1)})
    return pd.DataFrame(rows, columns=["table", "key", "row_id", "has_value", "critical"])


def values_without_source(sheets: dict[str, pd.DataFrame], phase1: pd.DataFrame) -> pd.DataFrame:
    """Rows with a value but no Doc ID / Source Document.

    ``critical`` is True when the row's status claims an extracted value (REPORTED/COMPUTED/...); a coded 0 with
    status NOT FOUND is listed but not critical.
    """
    rows = []
    for sheet, df in sheets.items():
        if "value_numeric" not in df.columns or "Doc ID" not in df.columns:
            continue
        has = df["value_numeric"].notna() | df["value_raw"].notna()
        bad = df[has & df["Doc ID"].isna()]
        for _, r in bad.iterrows():
            rows.append({"table": sheet, "row_id": r["Row ID"], "variable": r.get("Variable"),
                         "status": r.get("Status"), "critical": str(r.get("Status")).upper() in VALUE_STATUSES})
    bad = phase1[phase1["value"].notna() & phase1["source_document"].isna()]
    for _, r in bad.iterrows():
        rows.append({"table": "phase1_long", "row_id": f"{r['sheet']}:{r['row_number']}", "variable": r["variable"],
                     "status": r["status"], "critical": str(r["status"]).upper() in VALUE_STATUSES})
    return pd.DataFrame(rows, columns=["table", "row_id", "variable", "status", "critical"])


def doc_ids_missing_from_sources(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Doc IDs cited in data sheets but not registered in Sources."""
    known = set(sheets["Sources"]["Doc ID"].dropna())
    rows = []
    for sheet, df in sheets.items():
        col = "Mapped Doc ID" if sheet == "Phase1 Source Map" else "Doc ID"
        if col not in df.columns or sheet == "Sources":
            continue
        for d in sorted(set(df[col].dropna()) - known):
            rows.append({"table": sheet, "doc_id": d, "rows": int((df[col] == d).sum())})
    return pd.DataFrame(rows, columns=["table", "doc_id", "rows"])


def publication_date_coverage(canon: pd.DataFrame) -> pd.DataFrame:
    """Share of rows with a value that carry a publication date, by provenance and bank."""
    d = canon[canon["value"].notna()]
    g = d.groupby(["provenance", "bank_id"])
    out = pd.DataFrame({"rows_with_value": g.size(), "with_publication_date": g["publication_date"].apply(
        lambda s: s.notna().sum())}).reset_index()
    out["coverage"] = out["with_publication_date"] / out["rows_with_value"]
    return out


def phase1_vs_inputs(canon: pd.DataFrame, focus: list[str], threshold: float) -> pd.DataFrame:
    """NMB/CRDB variables in both sources (same canonical, FY year, basis): differences above ``threshold``.

    Compared against the inputs workbook's original vintage; ``matches_other_vintage`` notes when a restated
    vintage agrees within the threshold. Conflicts are reported, not resolved.
    """
    key = ["bank_id", "canonical_name", "period_end", "basis"]
    p1 = canon[(canon["provenance"] == "phase1") & canon["bank_id"].isin(focus) & canon["value"].notna()
               & (canon["period_type"] == "FY") & canon["category"].isna()]
    iw = canon[(canon["provenance"] == "inputs_workbook") & canon["value"].notna() & (canon["period_type"] == "FY")
               & canon["entity"].isin(PRIMARY_ENTITIES)]
    m = p1.merge(iw[[*key, "value", "is_restated", "doc_id", "source_row_id"]], on=key, suffixes=("_phase1", "_inputs"))
    if m.empty:
        return pd.DataFrame()
    m["relative_difference"] = (m["value_phase1"] - m["value_inputs"]) / m["value_inputs"].abs().where(
        m["value_inputs"] != 0)
    m["abs_rel"] = m["relative_difference"].abs()
    orig = m[~m["is_restated_inputs"]].copy()
    restated = m[m["is_restated_inputs"]]
    close_restated = restated[restated["abs_rel"] <= threshold].groupby(key)["source_row_id_inputs"].first()
    orig["matches_other_vintage"] = orig.set_index(key).index.map(close_restated)
    conflicts = orig[(orig["abs_rel"] > threshold) | (orig["value_inputs"].eq(0) & orig["value_phase1"].ne(0))]
    cols = ["bank_id", "canonical_name", "fiscal_year", "basis", "value_phase1", "value_inputs", "relative_difference",
            "source_document", "source_row_id_phase1", "doc_id_inputs", "source_row_id_inputs",
            "matches_other_vintage"]
    out = conflicts[cols].sort_values(["bank_id", "canonical_name", "fiscal_year"]).reset_index(drop=True)
    out["resolution"] = "UNRESOLVED (reported only; precedence rule CLAUDE.md 3.8 uses the inputs workbook)"
    explained = out["matches_other_vintage"].notna()
    out.loc[explained, "resolution"] = ("EXPLAINED: Phase 1 holds a restated comparative, kept as a separate "
                                        "restatement row (" + out.loc[explained, "matches_other_vintage"].astype(str)
                                        + "; D5/D21); the original is used for the original-vintage panel")
    out["evidence_class"] = "Derived"
    return out


def phase1_vs_inputs_compared(canon: pd.DataFrame, focus: list[str]) -> int:
    """Number of Phase 1 / inputs value pairs compared."""
    key = ["bank_id", "canonical_name", "period_end", "basis"]
    p1 = canon[(canon["provenance"] == "phase1") & canon["bank_id"].isin(focus) & canon["value"].notna()
               & (canon["period_type"] == "FY")]
    iw = canon[(canon["provenance"] == "inputs_workbook") & canon["value"].notna() & (canon["period_type"] == "FY")
               & canon["entity"].isin(PRIMARY_ENTITIES) & ~canon["is_restated"]]
    return len(p1.merge(iw[key].drop_duplicates(), on=key))
