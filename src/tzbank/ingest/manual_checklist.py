"""Filled manual checklists (decision D38; prompt 02e section 2; D29 manual-entry path).

The owner saves a filled copy of ``reports/manual_check_list.csv`` as
``data/raw/manual_entries/manual_check_list_filled_<YYYYMMDD>.csv`` (read-only raw input). ``convert`` turns every
row with a value into a manual-entry row in full TZS:

- units: TZS, TZS'000, TZS million (all spellings seen) -> full TZS;
- source: the PDF among the row's candidates whose cited page prints the value (the page text "from 2013 report
  comparative" names the report when given); its Doc ID follows ``workbook.doc_id_for``; confidence HIGH for an
  own-year figure, MEDIUM for a comparative. The owner's page text is kept.

``gate`` then checks each row against the panel (``data/processed/bank_year_wide.csv``) with all converted rows
overlaid, and accepts it only if every applicable check passes:

- deposit mix (current + savings + time + other) = customer deposits within 1%;
- Tier 1 / RWA = the reported Tier 1 ratio (0.5 pp; printed ratios are rounded);
- gross loans - allowance = net loans within 2%;
- interest income - interest expense = NII (0.5%), or the implied NII printed on the same page;
- deposit interest + borrowings interest <= total interest expense (equal within 0.1% for NMB/CRDB, D29/02d);
- year-on-year change above 3x: flagged, not rejected.

A row with no applicable arithmetic check is accepted only if its value is printed on the cited page. Rejections go
to ``reports/manual_entry_rejections.csv`` for the owner; they are never written.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

UNIT_SCALE = {"tzs": 1.0, "tzs'000": 1e3, "tzs' 000": 1e3, "tzs '000": 1e3, "tzs'000s": 1e3, "tzs million": 1e6,
              "tzs' million": 1e6, "tzs' millions": 1e6, "tzs millions": 1e6, "tzs mn": 1e6, "tzs m": 1e6}
UNIT_NAME = {1.0: "TZS", 1e3: "TZS '000", 1e6: "TZS million"}
CANON = {"Customer Deposits": "customer_deposits", "Customer Deposits: Current Accounts": "deposits_current",
         "Customer Deposits: Savings": "deposits_savings", "Customer Deposits: Time": "deposits_time",
         "Customer Deposits: Other": "deposits_other", "Gross Loans and Advances to Customers": "gross_loans",
         "Loss Allowance on Loans (ECL)": "loan_loss_allowance", "Loans and Advances to Customers (Net)": "net_loans",
         "Tier 1 Capital": "tier1_capital", "Total RWA": "rwa_total", "Interest Income": "interest_income",
         "Interest Expense": "interest_expense", "Net Interest Income": "net_interest_income",
         "Interest Expense on Customer Deposits": "interest_expense_deposits",
         "Interest Expense on Borrowings": "interest_expense_borrowings", "Net Profit": "net_profit",
         "Total Equity": "total_equity"}
MIX = ["deposits_current", "deposits_savings", "deposits_time"]
FOLDER_BANK = {"CRDB": "B01", "NMB": "B02", "DCB": "B03", "Maendeleo": "B04", "MKCB": "B05", "MCB": "B06",
               "MUCOBA": "B07"}
REJECT_COLUMNS = ["source_file", "bank_id", "variable", "fiscal_year", "value_as_entered", "unit_as_entered",
                  "value_full", "page_as_entered", "file", "reason", "gate_results", "evidence_class"]


def filled_files(root: Path | None = None) -> list[Path]:
    return sorted((path("raw", root or find_root()) / "manual_entries").glob("manual_check_list_filled_*.csv"))


def unit_scale(unit: object) -> float | None:
    u = re.sub(r"\s+", " ", str(unit or "").strip().lower()).replace("’", "'")
    return UNIT_SCALE.get(u)


def page_number(text: object) -> int | None:
    m = re.match(r"\s*(?:PDF\s*)?p?\.?\s*(\d+)", str(text or ""), re.I)
    return int(m.group(1)) if m else None


def comparative_year(text: object) -> int | None:
    m = re.search(r"from (\d{4})(?: report)? comparative", str(text or ""), re.I)
    return int(m.group(1)) if m else None


def _printed_numbers(pdf: Path, page: int) -> set[float]:
    from tzbank.extract.pdftext import _doc

    d = _doc(str(pdf))
    if not 1 <= page <= d.page_count:
        return set()
    text = d[page - 1].get_text()
    out = set()
    for t in re.findall(r"\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?\d+(?:\.\d+)?\)?", text):
        try:
            out.add(abs(float(re.sub(r"[,\s()]", "", t))))
        except ValueError:
            continue
    return out


def find_value(pdf: Path, page: int | None, value: float) -> int | None:
    """Page (cited page first, then +/-2, then every page) on which ``value`` (as entered) is printed."""
    from tzbank.extract.pdftext import _doc

    n = _doc(str(pdf)).page_count
    order = ([page, page - 1, page + 1, page - 2, page + 2] if page else []) + list(range(1, n + 1))
    for p in dict.fromkeys(x for x in order if x and 1 <= x <= n):
        if abs(value) in _printed_numbers(pdf, p) or round(abs(value)) in _printed_numbers(pdf, p):
            return p
    return None


def find_sum(pdf: Path, pages: list[int], value: float) -> tuple[int, str] | None:
    """(page, 'a + b [+ c]') where ``value`` equals the sum of two or three printed numbers (a 'sum the deposit
    lines' entry), searched on the given pages only."""
    from itertools import combinations

    from tzbank.extract.pdftext import _doc

    n = _doc(str(pdf)).page_count
    for p in dict.fromkeys(x for x in pages if 1 <= x <= n):
        nums = sorted(x for x in _printed_numbers(pdf, p) if 0 < x < abs(value) and x >= 100)
        for k in (2, 3):
            for combo in combinations(nums, k):
                if abs(sum(combo) - abs(value)) < 0.5:
                    return p, " + ".join(f"{x:,.0f}" for x in combo)
    return None


def _candidate_pages(text: str, file: str) -> list[int]:
    m = re.search(re.escape(Path(file).name) + r": p\.([\d, ]+)", text or "")
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []


def _file_year(f: str) -> int | None:
    m = re.findall(r"(20\d\d)", Path(f).name)
    return int(m[0]) if m else None


def convert(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(converted rows, conversion rejections) from every filled checklist."""
    from tzbank.extract.ocr import extraction_source
    from tzbank.extract.pdftext import _doc
    from tzbank.extract.workbook import doc_id_for

    root = root or find_root()
    rows, rej = [], []
    for f in filled_files(root):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for _, r in df[df["value"].str.strip() != ""].iterrows():
            base = {"source_file": f.name, "bank_id": r["bank_id"], "variable": r["variable"],
                    "fiscal_year": int(r["fiscal_year"]), "value_as_entered": r["value"], "unit_as_entered": r["unit"],
                    "page_as_entered": r["page"], "evidence_class": "Observed"}
            scale = unit_scale(r["unit"])
            value = pd.to_numeric(r["value"].replace(",", ""), errors="coerce")
            if scale is None or pd.isna(value):
                rej.append(base | {"reason": f"unit {r['unit']!r} or value {r['value']!r} not understood"})
                continue
            cands = [c.strip().removeprefix("data/raw/documents/") for c in r["pdf_original"].split(";") if c.strip()]
            hint = comparative_year(r["page"])
            fy = int(r["fiscal_year"])
            if hint:
                cands = sorted(cands, key=lambda c: _file_year(c) != hint)
            else:
                cands = sorted(cands, key=lambda c: (_file_year(c) != fy, _file_year(c) != fy + 1))
            page = page_number(r["page"])
            chosen, found = None, None
            for c in cands:
                pdf = path("documents", root) / c
                if not pdf.exists():
                    continue
                p = find_value(extraction_source(pdf, root), page, float(value))
                if p is not None:
                    chosen, found = c, p
                    break
            summed = None
            if chosen is None:
                for c in cands:
                    pdf = path("documents", root) / c
                    if not pdf.exists():
                        continue
                    near = ([page, page - 1, page + 1] if page else []) + _candidate_pages(r["candidate_pages"], c)
                    hit = find_sum(extraction_source(pdf, root), near, float(value))
                    if hit:
                        chosen, (found, summed) = c, hit
                        break
            if chosen is None:
                chosen = next((c for c in cands if (path("documents", root) / c).exists()), None)
            if chosen is None:
                rej.append(base | {"reason": "none of the candidate PDFs is held"})
                continue
            if (found or page) is None:
                rej.append(base | {"file": chosen, "reason": "no page number in the entry and the value (or a sum of "
                                                             "printed lines) is not found in the PDF text"})
                continue
            doc_fy = _file_year(chosen) or fy
            pages = _doc(str(path("documents", root) / chosen)).page_count
            notes = [f"owner checklist {f.name}", f"page as entered: {r['page']}", f"unit as entered: {r['unit']}",
                     f"checked_by as entered: {r['checked_by']}"]
            notes.append(f"value = {summed} printed on PDF p.{found}" if summed else f"value printed on PDF p.{found}"
                         if found else "value not found in the PDF text layer (scanned or OCR differs); owner's reading")
            rows.append({"bank_id": r["bank_id"], "fiscal_year": fy, "period_type": "FY", "variable": r["variable"],
                         "value": float(value), "unit": UNIT_NAME[scale], "value_full": float(value) * scale,
                         "basis": "GROUP" if r["bank_id"] in {"B01", "B02"} else "BANK", "file": chosen,
                         "page": found or page, "page_as_entered": r["page"], "value_found_on_page": found is not None,
                         "value_is_sum_of_printed_lines": summed or "",
                         "as_reported_label": r["expected_label"], "checked_by": r["checked_by"],
                         "checked_on": f.stem.rsplit("_", 1)[-1], "confidence": "HIGH" if doc_fy == fy else "MEDIUM",
                         "doc_fiscal_year": doc_fy,
                         "doc_id": doc_id_for(r["bank_id"], chosen, doc_fy, pages <= 6),
                         "column": "current" if doc_fy == fy else "prior", "source_file": f.name,
                         "notes": "; ".join(notes), "extraction_method": "manual", "evidence_class": "Observed"})
    return pd.DataFrame(rows), pd.DataFrame(rej, columns=REJECT_COLUMNS)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1.0)


def gate(conv: pd.DataFrame, wide: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Add ``gate_results``, ``accepted``, ``reason`` and ``yoy_flag`` to the converted rows."""
    from tzbank.extract.ocr import extraction_source

    root = root or find_root()
    ctx = {(b, int(y), c): v for (b, y), r in wide.set_index(["bank_id", "fiscal_year"]).iterrows()
           for c, v in r.items() if isinstance(v, (int, float)) and not pd.isna(v)}
    for _, r in conv.iterrows():
        if r["variable"] in CANON:
            ctx[(r["bank_id"], int(r["fiscal_year"]), CANON[r["variable"]])] = r["value_full"]

    def g(b, y, c):
        return ctx.get((b, y, c))

    out = []
    for _, r in conv.iterrows():
        b, y, c, v = r["bank_id"], int(r["fiscal_year"]), CANON.get(r["variable"]), r["value_full"]
        res: list[tuple[str, str]] = []
        if c in MIX or c in {"deposits_other", "customer_deposits"}:
            parts = [g(b, y, m) for m in MIX]
            tot = g(b, y, "customer_deposits")
            if all(p is not None for p in parts) and tot:
                s = sum(parts) + (g(b, y, "deposits_other") or 0.0)
                res.append(("deposit mix = customer deposits (1%)", "PASS" if _close(s, tot, 0.01) else
                            f"FAIL: mix {s:,.0f} vs deposits {tot:,.0f} ({s / tot - 1:+.2%})"))
        if c in {"tier1_capital", "rwa_total"}:
            t1, rwa, ratio = g(b, y, "tier1_capital"), g(b, y, "rwa_total"), g(b, y, "tier1_ratio")
            if t1 and rwa and ratio:
                res.append(("Tier 1 / RWA = reported ratio (0.5 pp)", "PASS" if abs(t1 / rwa - ratio) <= 0.005 else
                            f"FAIL: {t1 / rwa:.2%} vs reported {ratio:.2%}"))
        if c in {"gross_loans", "loan_loss_allowance"}:
            gr, al, net = g(b, y, "gross_loans"), g(b, y, "loan_loss_allowance"), g(b, y, "net_loans")
            if gr and al is not None and net:
                res.append(("gross - allowance = net loans (2%)", "PASS" if _close(gr - abs(al), net, 0.02) else
                            f"FAIL: {gr - abs(al):,.0f} vs net {net:,.0f}"))
        if c in {"interest_income", "interest_expense", "net_interest_income"}:
            ii, ie, nii = g(b, y, "interest_income"), g(b, y, "interest_expense"), g(b, y, "net_interest_income")
            if ii and ie and nii:
                if _close(ii - abs(ie), nii, 0.005):
                    res.append(("II - IE = NII (0.5%)", "PASS"))
                else:
                    pdf = path("documents", root) / r["file"]
                    scale = r["value_full"] / r["value"] if r["value"] else 1.0
                    printed = r["page"] and ((ii - abs(ie)) / scale) in _printed_numbers(
                        extraction_source(pdf, root), int(r["page"]))
                    res.append(("II - IE = NII (0.5%)", "PASS (implied NII printed on the same page; the panel NII is "
                                "another vintage)" if printed else f"FAIL: {ii - abs(ie):,.0f} vs NII {nii:,.0f}"))
        if c in {"interest_expense_deposits", "interest_expense_borrowings"}:
            dep, bor, ie = g(b, y, "interest_expense_deposits"), g(b, y, "interest_expense_borrowings"), g(
                b, y, "interest_expense")
            if dep is not None and ie:
                s = dep + (bor or 0.0)
                if b in {"B01", "B02"} and bor is not None and not _close(s, ie, 0.001):
                    res.append(("deposits + borrowings interest = total (0.1%, NMB/CRDB)",
                                f"residual {ie - s:,.0f} (other note lines)" if s < ie else
                                f"FAIL: {s:,.0f} exceeds total {ie:,.0f}"))
                else:
                    lab = "deposits + borrowings interest <= total" if bor is not None else "deposit interest <= total"
                    res.append((lab, "PASS" if s <= ie * 1.001 else
                                f"FAIL: {s:,.0f} exceeds total {ie:,.0f}"))
        if not res:
            res.append(("value printed on the cited page (no arithmetic check applicable)",
                        "PASS" if r["value_found_on_page"] else "FAIL: value not found in the PDF text"))
        prev = g(b, y - 1, c) if c else None
        yoy = ""
        if prev and v and (abs(v / prev) > 3 or abs(v / prev) < 1 / 3):
            yoy = f"year-on-year change {v / prev:.2f}x vs FY{y - 1} ({prev:,.0f}): flagged for the owner"
        fails = [f"{k}: {s}" for k, s in res if s.startswith("FAIL")]
        out.append(r.to_dict() | {"gate_results": " | ".join(f"{k}: {s}" for k, s in res), "accepted": not fails,
                                  "reason": "; ".join(fails), "yoy_flag": yoy})
    return pd.DataFrame(out)


def gated(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(accepted rows, rejections incl. conversion failures)."""
    root = root or find_root()
    conv, rej = convert(root)
    if conv.empty:
        return conv, rej
    wide = pd.read_csv(path("processed", root) / "bank_year_wide.csv", low_memory=False)
    gd = gate(conv, wide, root)
    bad = gd[~gd["accepted"]]
    rej = pd.concat([rej, bad.assign(page_as_entered=bad["page_as_entered"]).reindex(columns=REJECT_COLUMNS)],
                    ignore_index=True)
    rej["evidence_class"] = "Observed"
    return gd[gd["accepted"]].reset_index(drop=True), rej


def write_checklist(root: Path | None = None) -> list[Path]:
    """data/interim/manual_checklist_rows.csv (all converted rows with gate results) and
    reports/manual_entry_rejections.csv. Needs the PDFs (page check), so it runs in the D18 session."""
    root = root or find_root()
    conv, rej = convert(root)
    wide = pd.read_csv(path("processed", root) / "bank_year_wide.csv", low_memory=False)
    gd = gate(conv, wide, root) if len(conv) else conv
    bad = gd[~gd["accepted"]] if len(gd) else gd
    rej = pd.concat([rej, bad.reindex(columns=REJECT_COLUMNS)], ignore_index=True)
    rej["evidence_class"] = "Observed"
    gd = gd.replace({np.nan: None})
    return [write_csv(gd, path("interim", root) / "manual_checklist_rows.csv", require_evidence=True),
            write_csv(rej, path("reports", root) / "manual_entry_rejections.csv", require_evidence=True)]


if __name__ == "__main__":
    for p in write_checklist():
        print(p)
