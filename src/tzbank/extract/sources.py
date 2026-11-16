"""Sources evidence for D16 publication dates (prompt 02 sections 2.1, 2.4, 2.5).

Reclassifies every existing Sources date by how it was established (approval / auditor / posting / AGM notice),
adds dates found in the held reports (``tzbank.extract.dates``), and builds Sources rows for documents newly cited
by the extraction ledger. Output feeds ``tzbank.extract.workbook.update``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.dates import find_dates
from tzbank.extract.ocr import extraction_source

BANK_DOC_TYPES = {"Annual report", "Audited FS abstract", "Interim", "Condensed annual"}


def period_end(period_covered: str, doc_type: str) -> pd.Timestamp | None:
    """'FY2020' -> 2020-12-31; '9M2024' -> 2024-09-30; 'H12025' -> 2025-06-30."""
    p = str(period_covered or "")
    if m := re.fullmatch(r"FY(\d{4})", p):
        return pd.Timestamp(f"{m.group(1)}-12-31")
    if m := re.fullmatch(r"9M(\d{4})", p):
        return pd.Timestamp(f"{m.group(1)}-09-30")
    if m := re.fullmatch(r"H1(\d{4})", p):
        return pd.Timestamp(f"{m.group(1)}-06-30")
    return None


def classify_existing(how: str) -> str:
    """What an existing Sources date is, from 'How Publication Date Established'."""
    h = str(how or "").lower()
    if "agm notice" in h:
        return "agm_notice"
    if "annual report" in h and "board approval" in h or "integrated report" in h and "board approval" in h:
        return "report_approval"  # D24: not the FS approval
    if "board approval" in h or "board-approval" in h or "approval date" in h or "signature date" in h:
        return "fs_approval"
    if "auditor report date" in h:
        return "auditor"
    if any(k in h for k in ["posting", "dse filing", "file timestamp", "announcement", "notice", "dsm", "release",
                            "published", "dated", "statement date", "auction date", "data date", "report date",
                            "update", "same day", "decision announced", "posted", "download"]):
        return "actual"
    return "unknown"


def local_file(doc_id: str, local_path: str, root: Path) -> Path | None:
    """Held PDF for a Sources row: same file name under data/raw/documents/<bank>/, else <bank>/<year>.pdf."""
    docs = path("documents", root)
    name = Path(str(local_path or "").replace("\\", "/")).name
    for f in docs.rglob(name) if name else []:
        return f
    m = re.match(r"(CRDB|NMB)-AR-FY(\d{4})", doc_id)
    if m:
        f = docs / m.group(1) / f"{m.group(2)}.pdf"
        return f if f.exists() else None
    return None


def existing_updates(sources: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Evidence columns for every existing Sources row."""
    root = root or find_root()
    rows = []
    for _, s in sources.iterrows():
        old = pd.to_datetime(s.get("Publication Date"), errors="coerce")
        kind = classify_existing(s.get("How Publication Date Established"))
        pe = period_end(s.get("Period Covered"), s.get("Document Type"))
        u = {"Doc ID": s["Doc ID"], "Period End": pe, "Legacy Publication Date (pre-D16)": old}
        is_bank = s.get("Document Type") in BANK_DOC_TYPES
        evidence = [f"pre-D16 date {old.date() if pd.notna(old) else 'none'} was: {kind}"]
        if kind in {"actual", "agm_notice"} or not is_bank:
            u["Actual Publication Date"] = old
        elif kind == "fs_approval":
            u["FS Board Approval Date"] = old
        elif kind == "auditor":
            u["Auditor Report Date"] = old
        how = str(s.get("How Publication Date Established") or "")
        if m := re.search(r"auditor's report(?:/FS approval)? (?:date )?(\d{1,2} \w{3} \d{4})", how):
            u.setdefault("Auditor Report Date", pd.to_datetime(m.group(1), errors="coerce"))
            if "FS approval" in how:
                u.setdefault("FS Board Approval Date", pd.to_datetime(m.group(1), errors="coerce"))
        if m := re.search(r"board approval (\d{1,2} \w{3} \d{4})", how):
            u.setdefault("FS Board Approval Date", pd.to_datetime(m.group(1), errors="coerce"))
        if is_bank and str(s.get("Document Type")) != "Interim":
            f = local_file(s["Doc ID"], s.get("Local File Path"), root)
            if f is not None and pe is not None:
                found = find_dates(extraction_source(f, root), pe.year)
                for key, col in [("fs_board_approval_date", "FS Board Approval Date"),
                                 ("auditor_report_date", "Auditor Report Date")]:
                    if key in found and pd.isna(u.get(col)):
                        u[col] = found[key]
                        evidence.append(f"{col} from {f.parent.name}/{f.name} p.{found[key.replace('date', 'page')]}")
                if "agm_notice_date" in found and pd.isna(u.get("Actual Publication Date")):
                    u["Actual Publication Date"] = found["agm_notice_date"]
                    evidence.append(f"actual: dated AGM notice in {f.parent.name}/{f.name} p.{found['agm_notice_page']}"
                                    f" ('{found['agm_notice_quote']}')")
        u["Date Evidence"] = "; ".join(evidence)
        rows.append(u)
    return pd.DataFrame(rows)


def new_sources(ledger: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Sources rows for documents the ledger cites that Sources does not hold yet."""
    root = root or find_root()
    rows = []
    for (doc_id, file), g in ledger.groupby(["doc_id", "file"]):
        f = path("documents", root) / file
        fy = int(g["doc_fiscal_year"].iloc[0])
        found = find_dates(extraction_source(f, root), fy) if f.exists() else {}
        condensed = "FSPUB" in doc_id
        ev = [f"held file {file}"]
        for k in ["fs_board_approval", "auditor_report", "agm_notice"]:
            if f"{k}_date" in found:
                ev.append(f"{k} p.{found[f'{k}_page']}: '{found[f'{k}_quote'][:80]}'")
        rows.append({
            "Doc ID": doc_id, "Owner": doc_id.split("-")[0],
            "Document Title": f"{doc_id.split('-')[0]} " + (
                "condensed audited statements (DSE/newspaper publication)" if condensed
                else "annual report and audited financial statements") + f" FY{fy}",
            "Document Type": "Condensed annual" if condensed else "Annual report", "Period Covered": f"FY{fy}",
            "Primary / Secondary": "PRIMARY", "Local File Path": f"data/raw/documents/{file}",
            "How Publication Date Established": "D16: actual if a dated AGM notice is in the report; otherwise "
                                                "regulatory estimate from the FS board approval date",
            "Period End": pd.Timestamp(f"{fy}-12-31"),
            "Actual Publication Date": found.get("agm_notice_date"),
            "FS Board Approval Date": found.get("fs_board_approval_date"),
            "Auditor Report Date": found.get("auditor_report_date"),
            "Date Evidence": "; ".join(ev),
            "Notes": "Added by prompt 02 extraction (tzbank.extract).",
        })
    return pd.DataFrame(rows)
