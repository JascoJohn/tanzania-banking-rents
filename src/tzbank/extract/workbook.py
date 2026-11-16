"""Write reviewed extraction rows into the working inputs workbook (decision D18; prompt 02 sections 1-2).

Run only inside a D18 update session: ``backup()`` first (copy to backups/inputs/ + commit), then ``update()``,
then the MANIFEST update, ``run all`` and a commit. Formula columns of appended rows copy the sheet's own formulas
(translated to the new row) so the workbook stays usable in Excel; the pipeline never reads their cached values.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.formula.translate import Translator

from tzbank.config import banks, find_root, path

BANK_NAMES = {b["bank_id"]: b["name"] for b in banks()}
CAPITAL_SHEET_VARS = {"Total RWA", "Tier 1 Capital"}
SOURCE_EVIDENCE_COLUMNS = ["Period End", "Actual Publication Date", "FS Board Approval Date", "Auditor Report Date",
                           "Date Evidence", "Legacy Publication Date (pre-D16)", "Estimated Publication Date",
                           "Estimate Basis"]


def backup(root: Path | None = None) -> Path:
    """Copy the working workbook to backups/inputs/NMB_CRDB_Valuation_Inputs_<YYYYMMDD-HHMM>.xlsx (D18)."""
    root = root or find_root()
    src = path("inputs_workbook", root)
    dst = root / "backups" / "inputs" / f"{src.stem}_{dt.datetime.now():%Y%m%d-%H%M}.xlsx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _header(ws) -> dict[str, int]:
    return {c.value: c.column for c in ws[1] if c.value}


def _last_row(ws, col: int = 1) -> int:
    r = ws.max_row
    while r > 1 and ws.cell(r, col).value in (None, ""):
        r -= 1
    return r


def _append(ws, values: dict[str, object], template_row: int = 2) -> int:
    """Append one row; columns missing from ``values`` copy row-2 formulas translated to the new row."""
    h = _header(ws)
    r = _last_row(ws) + 1
    for name, col in h.items():
        if name in values:
            ws.cell(r, col).value = values[name]
            continue
        tmpl = ws.cell(template_row, col).value
        if isinstance(tmpl, str) and tmpl.startswith("="):
            src_ref = ws.cell(template_row, col).coordinate
            ws.cell(r, col).value = Translator(tmpl, origin=src_ref).translate_formula(ws.cell(r, col).coordinate)
    return r


def _next_id(ws, prefix: str) -> int:
    ids = [str(ws.cell(r, 1).value) for r in range(2, _last_row(ws) + 1)]
    nums = [int(i.split("-")[1]) for i in ids if i.startswith(prefix + "-")]
    return max(nums) + 1 if nums else 1


def doc_id_for(bank_id: str, file: str, fiscal_year: int, condensed: bool) -> str:
    """Doc ID for a B03-B07 document (or an NMB/CRDB file not yet in Sources)."""
    ticker = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB",
              "B07": "MUCOBA"}[bank_id]
    return f"{ticker}-{'FSPUB' if condensed else 'AR'}-FY{fiscal_year}"


def update(ledger: pd.DataFrame, sources: pd.DataFrame, source_updates: pd.DataFrame, readme_lines: list[str],
           root: Path | None = None) -> dict[str, int]:
    """Append ledger rows (HF / Regulatory Capital), new Sources rows, Sources evidence columns, Read Me lines."""
    root = root or find_root()
    f = path("inputs_workbook", root)
    wb = openpyxl.load_workbook(f)
    counts = {"Historical Financials": 0, "Regulatory Capital": 0, "Sources": 0}
    hf, rc, src = wb["Historical Financials"], wb["Regulatory Capital"], wb["Sources"]

    # Sources: evidence columns (D16), then existing-row updates, then new documents
    h = _header(src)
    for name in SOURCE_EVIDENCE_COLUMNS:
        if name not in h:
            c = src.max_column + 1
            src.cell(1, c).value = name
            h[name] = c
    row_of = {src.cell(r, 1).value: r for r in range(2, _last_row(src) + 1)}
    for _, u in source_updates.iterrows():
        r = row_of.get(u["Doc ID"])
        if r is None:
            continue
        for k, v in u.items():
            if k != "Doc ID" and k in h and not (isinstance(v, float) and pd.isna(v)):
                src.cell(r, h[k]).value = v.to_pydatetime() if isinstance(v, pd.Timestamp) else v
    for _, s in sources.iterrows():
        if s["Doc ID"] in row_of:
            continue
        r = _last_row(src) + 1
        for k, v in s.items():
            if k in h and not (isinstance(v, float) and pd.isna(v)):
                src.cell(r, h[k]).value = v.to_pydatetime() if isinstance(v, pd.Timestamp) else v
        row_of[s["Doc ID"]] = r
        counts["Sources"] += 1

    placeholders = {}
    for ws in (hf, rc):
        hh = _header(ws)
        for r in range(2, _last_row(ws) + 1):
            if ws.cell(r, hh["Value"]).value in (None, "") and str(ws.cell(r, hh["Status"]).value).upper() in {
                    "TO COLLECT", "NOT DISCLOSED"}:
                pe = ws.cell(r, hh["Period End"]).value
                key = (ws.title, ws.cell(r, hh["Bank ID"]).value, ws.cell(r, hh["Basis"]).value,
                       ws.cell(r, hh["Period Type"]).value, pe.year if hasattr(pe, "year") else None,
                       ws.cell(r, hh["Variable"]).value)
                placeholders.setdefault(key, r)
    existing: dict = {}
    held_docs: dict = {}
    for ws in (hf, rc):
        hh = _header(ws)
        for r in range(2, _last_row(ws) + 1):
            v = ws.cell(r, hh["Value"]).value
            if v in (None, ""):
                continue
            pe = ws.cell(r, hh["Period End"]).value
            k = (ws.title, ws.cell(r, hh["Bank ID"]).value, ws.cell(r, hh["Basis"]).value,
                 ws.cell(r, hh["Period Type"]).value, pe.year if hasattr(pe, "year") else None,
                 ws.cell(r, hh["Variable"]).value)
            existing.setdefault(k, (r, v))
            held_docs.setdefault(k, set()).add(ws.cell(r, hh["Doc ID"]).value)
    counts["placeholders_filled"] = 0
    counts["skipped_already_held"] = 0
    skipped = []
    for _, x in ledger.iterrows():
        cap = x["variable"] in CAPITAL_SHEET_VARS
        ws, prefix = (rc, "RC") if cap else (hf, "HF")
        small = x["bank_id"] not in {"B01", "B02"}
        d17 = cap and x["bank_id"] == "B01" and 2009 <= int(x["fiscal_year"]) <= 2014
        basis = "BANK" if (small or (cap and x["bank_id"] == "B02") or d17) else "GROUP"
        entity = "Bank solo" if basis == "BANK" else "Group"
        notes = [x.get("notes") if isinstance(x.get("notes"), str) else "",
                 f"Extracted by tzbank.extract (gate: {x['gate']}); label '{x['label']}'",
                 "bank_equals_group=True (no subsidiaries found in the held reports, D23)" if small else "",
                 "Bank-solo capital accepted for CRDB FY2009-FY2014 (D17)" if d17 else ""]
        method = x.get("extraction_method") if isinstance(x.get("extraction_method"), str) else "gated"
        notes.append(f"extraction_method={method}")
        if x.get("ocr") is True or str(x.get("ocr")) == "True":
            notes.append("read from the OCR text layer; value passed an arithmetic gate")
        status = "COMPUTED" if str(x["label"]).startswith("sum of") or " + " in str(x["label"]) else "REPORTED"
        vals = {"Bank ID": x["bank_id"], "Bank": BANK_NAMES[x["bank_id"]], "Entity": entity, "Basis": basis,
                "Period Type": "FY", "Period End": dt.datetime(int(x["fiscal_year"]), 12, 31),
                "Variable": x["variable"], "Value": float(x["value"]),
                "Unit": "count" if str(x["variable"]).startswith("Number of ") else "TZS",
                "Status": status, "Doc ID": x["doc_id"], "Page / Note": f"PDF p.{int(x['page'])}",
                "Confidence": x["confidence"], "Notes": " | ".join(n for n in notes if n)}
        key = (ws.title, x["bank_id"], basis, "FY", int(x["fiscal_year"]), x["variable"])
        if key in held_docs and x["doc_id"] in held_docs[key]:
            counts["skipped_already_held"] += 1  # same line from the same document is already in the workbook
            continue
        if key in existing:
            r0, v0 = existing[key]
            later = int(str(x["doc_id"])[-4:]) > int(x["fiscal_year"]) if str(x["doc_id"])[-4:].isdigit() else False
            if later and v0 and abs(x["value"] - v0) / abs(v0) > 0.005:
                # D21: a later report's comparative that differs from the held original is a restatement row
                vals["Notes"] = (f"RESTATED in {x['doc_id']} comparative (original {v0:,.0f} held in row {r0}) | "
                                 + vals["Notes"])
                vals["Row ID"] = f"{prefix}-{_next_id(ws, prefix):04d}"
                _append(ws, vals)
                held_docs[key].add(x["doc_id"])
                counts["restated_rows_added"] = counts.get("restated_rows_added", 0) + 1
                continue
            # the workbook already holds this line: keep it, record the extraction as a cross-check only
            skipped.append({"sheet": ws.title, "row": r0, "bank_id": x["bank_id"], "fiscal_year": x["fiscal_year"],
                            "variable": x["variable"], "workbook_value": v0, "extracted_value": x["value"],
                            "rel_diff": (x["value"] - v0) / abs(v0) if v0 else None, "file": x["file"],
                            "page": x["page"]})
            counts["skipped_already_held"] += 1
            continue
        r = placeholders.pop(key, None) if x.get("row_type", "original") == "original" else None
        if r is not None:
            hh = _header(ws)
            old_note = ws.cell(r, hh["Notes"]).value
            for k, v in vals.items():
                ws.cell(r, hh[k]).value = v
            if old_note:
                ws.cell(r, hh["Notes"]).value = f"{vals['Notes']} | previous note: {old_note}"
            counts["placeholders_filled"] += 1
            continue
        vals["Row ID"] = f"{prefix}-{_next_id(ws, prefix):04d}"
        _append(ws, vals)
        counts[ws.title] += 1

    if skipped:
        pd.DataFrame(skipped).to_csv(root / "data" / "extraction" / "already_held_crosscheck.csv", index=False,
                                     lineterminator="\n")
    me = wb["Read Me"]
    r = me.max_row + 2
    for line in readme_lines:
        me.cell(r, 1).value = line
        r += 1
    wb.save(f)
    return counts


def correct(fixes: list[dict], readme_lines: list[str], root: Path | None = None) -> int:
    """Correct extraction errors in place (D18 session): each fix names bank_id, fiscal_year, variable, basis,
    old_value, value, doc_id, page and reason. The held cell must carry ``old_value`` (else KeyError, nothing is
    saved); the previous value and the reason are kept in Notes. Not a restatement (D5/D21): the same document's figure
    was mis-read, so the wrong number has no standing as an original."""
    root = root or find_root()
    f = path("inputs_workbook", root)
    wb = openpyxl.load_workbook(f)
    ws = wb["Historical Financials"]
    h = _header(ws)
    done = 0
    for x in fixes:
        hit = None
        for r in range(2, _last_row(ws) + 1):
            pe = ws.cell(r, h["Period End"]).value
            v = ws.cell(r, h["Value"]).value
            if (ws.cell(r, h["Bank ID"]).value == x["bank_id"] and ws.cell(r, h["Variable"]).value == x["variable"]
                    and ws.cell(r, h["Basis"]).value == x["basis"] and ws.cell(r, h["Period Type"]).value == "FY"
                    and getattr(pe, "year", None) == x["fiscal_year"] and isinstance(v, (int, float))
                    and abs(v - x["old_value"]) < 0.5):
                hit = r
                break
        if hit is None:
            raise KeyError(f"no held cell for {x}")
        old_note = ws.cell(hit, h["Notes"]).value or ""
        ws.cell(hit, h["Value"]).value = float(x["value"])
        ws.cell(hit, h["Doc ID"]).value = x["doc_id"]
        ws.cell(hit, h["Page / Note"]).value = f"PDF p.{int(x['page'])}"
        ws.cell(hit, h["Status"]).value = "REPORTED"
        ws.cell(hit, h["Confidence"]).value = "HIGH"
        ws.cell(hit, h["Notes"]).value = (f"CORRECTED {x['session']}: previously {x['old_value']:,.0f} ({x['reason']}) "
                                          f"| previous note: {old_note}")
        done += 1
    me = wb["Read Me"]
    r = me.max_row + 2
    for line in readme_lines:
        me.cell(r, 1).value = line
        r += 1
    wb.save(f)
    return done
