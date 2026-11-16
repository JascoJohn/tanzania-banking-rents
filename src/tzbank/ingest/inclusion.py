"""Financial inclusion panel (prompt 02c; decisions D30-D33; blueprint chapters 1, 2, 4.4, 4.7, 6 context).

The owner's curated workbook ``Tanzania_Financial_Inclusion_Panel_2018-2025.xlsx`` is a raw input (D30). The
pipeline reads its long-format observation sheet ``Panel_Long`` and the supporting sheets; the Excel formula views
(``Panel_Wide``, ``Panel_Vintage``) are never used as data. Formula columns are recomputed in Python:

- ``Std_Value`` = ``Reported_Value`` x ``Factor_to_Std``;
- ``Is_Latest_Vintage`` (D31): an included observation is the latest vintage when no other included observation of
  the same series and year comes from a more recent report.

``Panel_Wide``'s cached values are read only to check that the Python rule reproduces them (every mismatch is
reported). Series with a basis suffix (_A, _B, _R25) stay separate (D32).
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

WORKBOOK = "Tanzania_Financial_Inclusion_Panel_2018-2025.xlsx"
LONG_COLUMNS = ["Obs_ID", "Series_ID", "Domain", "Indicator", "Breakdown", "Year", "Reported_Value", "Reported_Unit",
                "Factor_to_Std", "Std_Value", "Std_Unit", "Source_Report", "Source_Table_or_Section", "Page_(printed)",
                "PDF_Page", "Evidence_Type", "Include_in_Panel", "Is_Latest_Vintage", "Notes"]
BASIS_SUFFIX = re.compile(r"_(A|B|R25)$")
# header row (1-based) of each supporting sheet
SUPPORT = {"Revision_Check": 5, "Data_Issues": 4, "Text_Facts": 4, "Source_Register": 3}
STANDALONE = ["Region_Branches", "Region_ATMs", "Region_BankAgents", "Region_Microfinance", "Region_TASAF",
              "Region_POS", "TMX_Trading", "LGA_Loans", "Credit_Guarantee", "Literacy_Themes"]


# blueprint chapters per Panel_Long domain number (crosswalk; prompt 02c section 6)
DOMAIN_CHAPTERS = {1: "1", 2: "1", 3: "1", 4: "1;2", 5: "1", 6: "1", 7: "1", 8: "1;4", 9: "6", 10: "2", 11: "2",
                   12: "1", 13: "1", 14: "1", 15: "1", 16: "4", 17: "1", 18: "2", 19: "2", 20: "1", 21: "1", 22: "1;6",
                   23: "6", 24: "1"}


def canonical_name(series_id: str) -> str:
    """Crosswalk name of an inclusion series: ``incl_`` + the owner's Series_ID in lower case."""
    return "incl_" + str(series_id).lower()


def workbook_path(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / WORKBOOK


def _strip(v: object) -> object:
    return None if isinstance(v, str) and v.startswith("=") else v


def read_long(file: Path | None = None) -> pd.DataFrame:
    """Panel_Long with formula columns recomputed (Std_Value, Is_Latest_Vintage)."""
    wb = openpyxl.load_workbook(file or workbook_path(), read_only=True, data_only=False)
    rows = list(wb["Panel_Long"].iter_rows(values_only=True))
    wb.close()
    df = pd.DataFrame([[_strip(v) for v in r] for r in rows[1:] if r[0]], columns=list(rows[0]))
    if list(df.columns) != LONG_COLUMNS:
        raise ValueError(f"Panel_Long columns changed: {list(df.columns)}")
    for c in ["Year", "Reported_Value", "Factor_to_Std", "Source_Report", "Include_in_Panel", "PDF_Page"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Std_Value"] = df["Reported_Value"] * df["Factor_to_Std"]
    df["Is_Latest_Vintage"] = latest_vintage(df)
    df["basis_suffix"] = df["Series_ID"].str.extract(BASIS_SUFFIX)[0]
    df["evidence_class"] = "Observed"
    return df


def latest_vintage(df: pd.DataFrame) -> pd.Series:
    """D31: 1 for the included observation from the most recent report per series-year, else 0."""
    inc = df[df["Include_in_Panel"] == 1]
    newest = inc.groupby(["Series_ID", "Year"])["Source_Report"].transform("max")
    flag = pd.Series(0, index=df.index)
    flag[inc.index[inc["Source_Report"] == newest]] = 1
    return flag


def panel(df: pd.DataFrame) -> pd.DataFrame:
    """Latest-vintage panel (long): one value per series-year. If two included rows tie on the newest report, the
    values are summed, as the workbook's SUMIFS does, and the tie is flagged."""
    lv = df[df["Is_Latest_Vintage"] == 1]
    g = lv.groupby(["Series_ID", "Year"], as_index=False)
    out = g.agg(Domain=("Domain", "first"), Indicator=("Indicator", "first"), Breakdown=("Breakdown", "first"),
                Std_Unit=("Std_Unit", "first"), value=("Std_Value", "sum"), source_report=("Source_Report", "first"),
                obs_ids=("Obs_ID", lambda s: ";".join(s)), n_obs=("Obs_ID", "size"),
                source_ref=("Source_Table_or_Section", "first"), pdf_page=("PDF_Page", "first"))
    out.insert(0, "canonical_name", out["Series_ID"].map(canonical_name))
    out["doc_id"] = "BOT-FIR-" + out["source_report"].astype("Int64").astype(str)  # Sources (prompt 02c section 5)
    out["tie_flag"] = out["n_obs"] > 1
    out["basis_suffix"] = out["Series_ID"].str.extract(BASIS_SUFFIX)[0]
    out["definition_break_note"] = out["basis_suffix"].map(
        lambda s: f"basis '{s}': separate series by design (D32); never chained with the other basis" if
        isinstance(s, str) else "")
    out["evidence_class"] = "Observed"
    return out


def read_wide_cached(file: Path | None = None) -> pd.DataFrame:
    """Panel_Wide cached values (comparison target only, D31)."""
    wb = openpyxl.load_workbook(file or workbook_path(), read_only=True, data_only=True)
    rows = list(wb["Panel_Wide"].iter_rows(min_row=4, values_only=True))
    wb.close()
    hdr = [str(h) if h is not None else "" for h in rows[0]]
    years = {i: int(h) for i, h in enumerate(hdr) if re.fullmatch(r"\d{4}", h)}
    out = []
    for r in rows[1:]:
        if not r[0]:
            continue
        for i, y in years.items():
            v = r[i]
            out.append({"Series_ID": r[0], "Year": y, "wide_value": None if v in ("", None) else float(v)})
    return pd.DataFrame(out)


def vintage_check(pan: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Python panel vs Panel_Wide, every series-year; status MATCH / MISMATCH / only_python / only_workbook."""
    m = wide.merge(pan[["Series_ID", "Year", "value", "source_report", "obs_ids"]], on=["Series_ID", "Year"],
                   how="outer")
    a, b = m["value"], m["wide_value"]
    both = a.notna() & b.notna()
    close = both & ((a - b).abs() <= 1e-9 + 1e-9 * b.abs())
    m["status"] = "both empty"
    m.loc[close, "status"] = "MATCH"
    m.loc[both & ~close, "status"] = "MISMATCH"
    m.loc[a.notna() & b.isna(), "status"] = "only_python"
    m.loc[a.isna() & b.notna(), "status"] = "only_workbook"
    m["evidence_class"] = "Derived"
    return m[m["status"] != "both empty"].sort_values(["status", "Series_ID", "Year"]).reset_index(drop=True)


def _sheet_frame(ws, header_row: int) -> pd.DataFrame:
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(rows[header_row - 1])]
    data = [[_strip(v) for v in r] for r in rows[header_row:] if any(v is not None for v in r)]
    return pd.DataFrame(data, columns=hdr[: len(data[0])] if data else hdr)


def _standalone(ws) -> pd.DataFrame:
    """A standalone sheet: header at row 4, plus a year row at 5 when present (regional panels)."""
    rows = list(ws.iter_rows(values_only=True))
    h4 = list(rows[3]) if len(rows) > 3 else []
    h5 = list(rows[4]) if len(rows) > 4 else []
    two_row = sum(1 for v in h5 if re.fullmatch(r"\d{4}", str(v or ""))) >= 3
    if two_row:
        group, names = None, []
        for i, (a, b) in enumerate(zip(h4, h5, strict=False)):
            group = a if a not in (None, "") else group
            names.append(str(b) if i == 0 else f"{group} | {b}")
        start = 5
    else:
        names = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(h4)]
        start = 4
    data = [[_strip(v) for v in r[: len(names)]] for r in rows[start:] if any(v is not None for v in r)]
    df = pd.DataFrame(data, columns=names)
    df.insert(0, "source_note", " ".join(str(x) for x in rows[1] if x) if len(rows) > 1 else "")
    return df.dropna(axis=1, how="all")


def write_inclusion(root: Path | None = None) -> list[Path]:
    """Interim and processed inclusion files plus the D31 vintage check."""
    root = root or find_root()
    f = workbook_path(root)
    long = read_long(f)
    i, p, v = path("interim", root), path("processed", root), path("validation", root)
    files = [write_csv(long, i / "inclusion_long.csv", require_evidence=True)]
    wb = openpyxl.load_workbook(f, read_only=True, data_only=False)
    for name, hr in SUPPORT.items():
        files.append(write_csv(_sheet_frame(wb[name], hr), i / f"inclusion_{name.lower()}.csv"))
    for name in STANDALONE:
        files.append(write_csv(_standalone(wb[name]), i / f"inclusion_{name.lower()}.csv"))
    wb.close()
    pan = panel(long)
    issues = pd.read_csv(i / "inclusion_data_issues.csv")
    pan["known_source_issues"] = pan["Series_ID"].map(lambda s: issues_for(s, long, issues))
    from tzbank.pit.pubdates import attach_doc_dates

    pan = attach_doc_dates(pan, root)
    from tzbank.ingest.adjudication import adjudicate

    pan = adjudicate(pan, "inclusion", root, long)
    from tzbank.extract.inclusion_nps_supplement import merge_supplements

    pan = merge_supplements(pan, "inclusion", root)  # D39: code-extracted, below the owner's values
    files.append(write_csv(pan, p / "inclusion_panel.csv", require_evidence=True))
    owner = pan[pan["source_precedence"] == "owner panel"]
    files.append(write_csv(vintage_check(owner, read_wide_cached(f)), v / "inclusion_vintage_check.csv",
                           require_evidence=True))
    return files


def issues_for(series_id: str, long: pd.DataFrame, issues: pd.DataFrame) -> str:
    """Data_Issues IDs whose description mentions the series' indicator or ID (attached to the series)."""
    ind = long.loc[long["Series_ID"] == series_id, "Indicator"].iloc[0] if (long["Series_ID"] == series_id).any() else ""
    keys = {series_id.lower(), str(ind).lower()}
    words = [w for w in re.split(r"[^a-z]+", str(ind).lower()) if len(w) > 4][:3]
    hits = []
    for _, r in issues.iterrows():
        text = f"{r.get('Area', '')} {r.get('Description', '')}".lower()
        if any(k and k in text for k in keys) or (words and all(w in text for w in words)):
            hits.append(str(r["Issue_ID"]))
    return ";".join(hits)
