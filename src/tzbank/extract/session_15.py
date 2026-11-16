"""D18 workbook session for prompt 15 (v0.17): quarterly statements and CRDB FY2010-15 year-end prices (D73).

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_15
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

1. New sheet "Quarterly Results" (same columns and formulas as Interim Results): every value accepted by
   ``tzbank.extract.quarterly`` (arithmetic gates, page-recorded) for the periods not already held. Interim Results
   keeps H1 2025-26 and 9M 2024-26; fourth-quarter statements are not written (the audited annual statements are held).
   Q1 / H1 / 9M values are cumulative for the period (BoT format), Group when the statement shows Group and Bank.
   Doc IDs ``<T>-QR-<YYYY>Q<q>`` are added to Sources; the publication date is the DSE filing date when the listing is
   dated after the quarter end, otherwise an estimate (quarter end + 30 days; the held 2024-26 statements were posted
   28-30 days after the quarter end), labelled in Sources.
2. Market Data: CRDB closing price at 31 December FY2010-15 from the annual reports (page-checked; issue 46).
   FY2015: the report (p.23) and the DSE daily record both give TZS 405; Phase 1 holds 380 (conflict resolved by the
   inputs-workbook precedence rule; written to reports/validation by the market_ext stage).
"""

from __future__ import annotations

import datetime as dt

import openpyxl
import pandas as pd
from openpyxl.formula.translate import Translator

from tzbank.config import find_root, path
from tzbank.extract import quarterly as Q
from tzbank.extract.workbook import _append, _header, _last_row, _next_id

VERSION = "v0.17"
SHEET = "Quarterly Results"
BANK = {"B01": "CRDB Bank", "B02": "NMB Bank"}
PTYPE = {1: "Q1", 2: "H1", 3: "9M"}
# CRDB closing price at 31 December, TZS per share, as printed in the annual report (file, printed page, quote)
CRDB_PRICES = [
    (2010, 115.0, "CRDB/2010.pdf p.21 ('Tshs 115'); also 2011.pdf p.28 '(2010: TZS 115)'"),
    (2011, 172.5, "CRDB/2011.pdf p.28"),
    (2012, 150.0, "CRDB/2012.pdf p.66"),
    (2013, 280.0, "CRDB/2013.pdf p.82"),
    (2014, 430.0, "CRDB/2014.pdf p.75"),
    (2015, 405.0, "CRDB/2015.pdf p.23; DSE daily close 31 Dec 2015 = 405 (data/library/dse)"),
]


def held_periods(root) -> set[tuple[str, str]]:
    ir = pd.read_csv(path("interim", root) / "inputs_interim_results.csv")
    return {(b, str(pd.Timestamp(p).date())) for b, p in zip(ir["Bank ID"], ir["Period End"], strict=False)}


def quarterly_rows(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ledger rows for the Quarterly Results sheet, Sources rows)."""
    e = Q.extract_all(root)
    a = e[e["status"].eq("accepted") & e["item"].isin(list(Q.WORKBOOK_VARS))].copy()
    a = a[a["quarter"].isin([1, 2, 3])]
    held = held_periods(root)
    a = a[[(b, p) not in held for b, p in zip(a["bank_id"], a["period_end"], strict=False)]]
    a = a.sort_values(["text_source"]).drop_duplicates(["bank_id", "period_end", "item"])  # text layer before OCR
    a = a.sort_values(["bank_id", "period_end", "item"])
    a["doc_id"] = [f"{t}-QR-{p[:4]}Q{int(q)}" for t, p, q in zip(a["ticker"], a["period_end"], a["quarter"], strict=False)]
    led = pd.DataFrame({
        "Bank ID": a["bank_id"], "Bank": a["bank_id"].map(BANK),
        "Entity": a["two_entities"].map({True: "Group", False: "Bank"}),
        "Basis": a["two_entities"].map({True: "GROUP", False: "BANK"}),
        "Period Type": a["quarter"].map(PTYPE), "Period End": pd.to_datetime(a["period_end"]),
        "Variable": a["item"].map(Q.WORKBOOK_VARS), "Value": a["value"],
        "Unit": ["ratio" if i == "nim_reported" else "TZS" for i in a["item"]], "Status": "REPORTED",
        "Doc ID": a["doc_id"],
        "Page / Note": [f"DSE filing p.{int(p)} ({s}); line '{lab}'; {c}" for p, s, lab, c in
                        zip(a["page"], a["text_source"], a["label"], a["column"], strict=False)],
        "Confidence": a["text_source"].map({"text layer": "HIGH", "OCR copy": "MEDIUM"}),
        "Notes": ["BoT regulatory format (not the IFRS presentation); cumulative for the period. Gate: " + g
                  + (f"; scale {u}" if u else "") for g, u in zip(a["gate"], a["unit_note"], strict=False)],
    })
    docs = a.drop_duplicates("doc_id")
    src = []
    for r in docs.itertuples():
        pe = pd.Timestamp(r.period_end)
        listed = pd.Timestamp(r.doc_date)
        if listed > pe + pd.Timedelta(days=5) and r.period_basis.startswith("dates printed") and \
                "mislabelled" not in r.period_basis:
            pub, how, est, basis = listed, "DSE listing date of the filing (data/library/dse/documents_index.csv)", \
                None, None
        else:
            pub = pe + pd.Timedelta(days=30)
            how = "ESTIMATED: quarter end + 30 days (DSE listing dated at the quarter end, not the filing date)"
            est, basis = pub, ("held 2024-26 quarterly statements were posted 28-30 days after the quarter end "
                               "(Sources *-INT-* rows)")
        src.append({"Doc ID": r.doc_id, "Owner": r.ticker,
                    "Document Title": f"{r.ticker} quarterly statement (BoT format), period ended {pe.date()}",
                    "Document Type": "Quarterly statement", "Period Covered": f"Q{int(r.quarter)}{pe.year}",
                    "Publication Date": pub.to_pydatetime(), "How Publication Date Established": how,
                    "Primary / Secondary": "PRIMARY",
                    "URL": f"https://dse.co.tz/en/main-market/company-profile/{r.ticker}/financial-reports",
                    "Local File Path": r.local_file,
                    "Notes": f"DSE '{r.title}' listed {r.doc_date}; sha256 {r.sha256}; {r.period_basis}.",
                    "Period End": pe.to_pydatetime(),
                    "Actual Publication Date": pub.to_pydatetime() if est is None else None,
                    "Estimated Publication Date": est.to_pydatetime() if est is not None else None,
                    "Estimate Basis": basis})
    return led, pd.DataFrame(src)


def apply(root=None) -> dict[str, int]:
    root = root or find_root()
    led, src = quarterly_rows(root)
    f = path("inputs_workbook", root)
    wb = openpyxl.load_workbook(f)
    ir = wb["Interim Results"]
    if SHEET in wb.sheetnames:
        raise RuntimeError(f"{SHEET} already exists; this session runs once")
    ws = wb.create_sheet(SHEET, index=wb.sheetnames.index("Interim Results") + 1)
    cols = [c.value for c in ir[1]]
    for j, name in enumerate(cols, start=1):
        ws.cell(1, j).value = name
        ws.column_dimensions[ws.cell(1, j).column_letter].width = ir.column_dimensions[
            ir.cell(1, j).column_letter].width
    tmpl = {name: ir.cell(2, j).value for j, name in enumerate(cols, start=1)}
    for i, r in enumerate(led.to_dict("records"), start=1):
        row = i + 1
        vals = r | {"Row ID": f"QR-{i:04d}", "Period End": r["Period End"].to_pydatetime()}
        for j, name in enumerate(cols, start=1):
            if name in vals:
                v = vals[name]
                ws.cell(row, j).value = None if isinstance(v, float) and pd.isna(v) else v
            elif isinstance(tmpl[name], str) and tmpl[name].startswith("="):
                origin = ir.cell(2, j).coordinate
                ws.cell(row, j).value = Translator(tmpl[name], origin=origin).translate_formula(
                    ws.cell(row, j).coordinate)
        ws.cell(row, cols.index("Period End") + 1).number_format = "yyyy-mm-dd"
    # Sources
    s = wb["Sources"]
    h = _header(s)
    have = {s.cell(r, 1).value for r in range(2, _last_row(s) + 1)}
    n_src = 0
    for r in src.to_dict("records"):
        if r["Doc ID"] in have:
            continue
        rr = _last_row(s) + 1
        for k, v in r.items():
            if k in h and v is not None and not (isinstance(v, float) and pd.isna(v)):
                s.cell(rr, h[k]).value = v
        n_src += 1
    # Market Data: CRDB year-end prices FY2010-15
    md = wb["Market Data"]
    n_md = 0
    existing = {(md.cell(r, 2).value, md.cell(r, 7).value, md.cell(r, 8).value) for r in range(2, _last_row(md) + 1)}
    for fy, price, page in CRDB_PRICES:
        pe = dt.datetime(fy, 12, 31)
        if ("B01", pe, "Closing Price") in existing:
            continue
        n = _next_id(md, "MD")
        _append(md, {"Row ID": f"MD-{n:04d}", "Bank ID": "B01", "Bank": "CRDB Bank", "Entity": "Group",
                     "Basis": "GROUP", "Period Type": "AS AT", "Period End": pe, "Variable": "Closing Price",
                     "Value": price, "Unit": "TZS", "Status": "REPORTED", "Doc ID": f"CRDB-AR-FY{fy}",
                     "Page / Note": page, "Confidence": "HIGH",
                     "Notes": "Share price at 31 December as printed in the annual report (prompt 15 item 4d; issue 46)."
                     + (" Phase 1 holds 380; the report and the DSE daily close give 405 (inputs workbook prevails)."
                        if fy == 2015 else "")})
        n_md += 1
    rm = wb["Read Me"]
    r0 = _last_row(rm) + 1
    for k, line in enumerate([
        f"{VERSION} ({dt.date.today():%Y-%m-%d}, prompt 15, D18 working workbook)",
        f"Quarterly Results (new sheet): {len(led)} values from {led['Doc ID'].nunique()} NMB / CRDB quarterly "
        "statements (BoT format, DSE filings) for periods not held in Interim Results; arithmetic-gated extraction "
        "(tzbank.extract.quarterly), page recorded.",
        f"Sources: {n_src} quarterly statements (<T>-QR-<YYYY>Q<q>). Market Data: CRDB closing price 31 Dec "
        "FY2010-15 from the annual reports (FY2015 405, not Phase 1's 380)."]):
        rm.cell(r0 + k, 1).value = line
    wb.save(f)
    return {"quarterly_rows": len(led), "documents": int(led["Doc ID"].nunique()), "sources": n_src,
            "market_data": n_md}


if __name__ == "__main__":
    print(apply())
