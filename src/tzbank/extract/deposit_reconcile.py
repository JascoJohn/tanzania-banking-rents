"""Deposit-interest reconciliation for NMB and CRDB (prompt 02d section 1; blueprint §4.2 funding, §2 cost of deposits).

Rule (prompt 02d 1.3): for NMB and CRDB a deposit-interest line is accepted only if the interest-expense note of the
document it comes from reconciles, i.e. the note lines sum to that document's audited total interest expense within
0.1% (``tzbank.extract.ledger.ie_note_components``). The weaker "deposit interest <= total interest expense" check of
the first extraction pass no longer admits a value.

``reconcile()`` re-reads every NMB/CRDB deposit-interest row of the working workbook from its own document and column
(current year, or the prior-year comparative of a later report) and classifies it:

- ``reconciled``: the row equals the reconciled note value;
- ``corrected``: the note reconciles but gives a different value (the row is corrected in a D18 session);
- ``not reconciled``: the note cannot be made to sum to the total (the value is withdrawn to the manual list).

Needs the git-ignored PDFs, so it runs in the D18 session (``tzbank.extract.session_02d``), not in ``run all``; its
output ``reports/validation/deposit_interest_reconciliation.csv`` is committed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.ledger import DEP_TYPES, ie_note_components
from tzbank.extract.pdftext import text_pages
from tzbank.io import write_csv

DEPOSIT_VARS = ["Interest Expense on Customer Deposits", *DEP_TYPES]
NOTE_VARS = [*DEPOSIT_VARS, "Interest Expense on Borrowings"]
TOL = 0.001


def doc_files(sources: pd.DataFrame, root: Path) -> dict[str, Path]:
    """Doc ID -> held PDF (the Sources 'Local File Path' basename inside data/raw/documents/<bank>/)."""
    out = {}
    for _, s in sources.iterrows():
        did, lp = str(s["Doc ID"]), s.get("Local File Path")
        m = re.match(r"(NMB|CRDB)-AR-FY\d{4}$", did)
        if not m or not isinstance(lp, str):
            continue
        f = path("documents", root) / m.group(1) / re.split(r"[\\/]", lp)[-1]
        if f.exists():
            out[did] = f
    return out


def targets(cands: pd.DataFrame, hf: pd.DataFrame, bank_id: str, fy: int, file: Path) -> list[float]:
    """Candidate audited totals for one document column: that document's income-statement value(s), then the
    workbook's held total interest expense for the year."""
    rel = f"{file.parent.name}/{file.name}"
    c = cands[(cands["variable"] == "Interest Expense") & (cands["file"] == rel) & (cands["fiscal_year"] == fy)]
    w = hf[(hf["Bank ID"] == bank_id) & (hf["Variable"] == "Interest Expense") & (hf["Period End"].dt.year == fy)
           & hf["value_numeric"].notna()]
    vals = [abs(float(v)) for v in [*c["value"], *w["value_numeric"]] if pd.notna(v)]
    return list(dict.fromkeys(round(v) for v in vals))


def note_parts(file: Path, col: int, tgts: list[float]) -> tuple[dict | None, float | None]:
    """First target the note reconciles to: ({variable: value}, target)."""
    if not text_pages(file):
        return None, None
    for t in tgts:
        hit = ie_note_components(file, t, col)
        if hit:
            parts = {k: v for k, (v, _) in hit["parts"].items()}
            types = [parts[k] for k in DEP_TYPES if k in parts]
            if types and "Interest Expense on Customer Deposits" not in parts:
                parts["Interest Expense on Customer Deposits"] = sum(types)
            parts["_page"] = hit["page"]
            return parts, t
    return None, None


def reconcile(root: Path | None = None) -> pd.DataFrame:
    """One row per NMB/CRDB deposit-interest (and borrowings) workbook row with a value."""
    from tzbank.ingest.inputs_workbook import load_interim

    root = root or find_root()
    hf = load_interim("Historical Financials", root)
    files = doc_files(load_interim("Sources", root), root)
    cands = pd.read_csv(root / "data" / "extraction" / "candidates_all.csv", low_memory=False,
                        usecols=["bank_id", "fiscal_year", "variable", "value", "file"])
    rows = hf[hf["Bank ID"].isin(["B01", "B02"]) & hf["Variable"].isin(NOTE_VARS) & hf["value_numeric"].notna()]
    cache: dict = {}
    out = []
    for _, r in rows.iterrows():
        fy, did = int(r["Period End"].year), str(r["Doc ID"])
        f = files.get(did)
        doc_fy = int(did[-4:]) if did[-4:].isdigit() else None
        rec = {"row_id": r["Row ID"], "bank_id": r["Bank ID"], "fiscal_year": fy, "variable": r["Variable"],
               "doc_id": did, "workbook_value": float(r["value_numeric"]), "workbook_gate": _gate(r["Notes"])}
        if f is None or doc_fy is None or doc_fy - fy not in (0, 1):
            out.append(rec | {"status": "not reconciled", "detail": "document not held or not the own-year/next-year "
                                                                   "report"})
            continue
        col = doc_fy - fy
        key = (str(f), col, fy)
        if key not in cache:
            cache[key] = note_parts(f, col, targets(cands, hf, r["Bank ID"], fy, f))
        parts, tgt = cache[key]
        rec |= {"file": f"{f.parent.name}/{f.name}", "column": "current" if col == 0 else "prior"}
        if parts is None:
            out.append(rec | {"status": "not reconciled",
                              "detail": "note lines do not sum to the document's total interest expense (0.1%)"})
            continue
        v = parts.get(r["Variable"])
        rec |= {"page": parts["_page"], "audited_total": tgt, "reconciled_value": v,
                "deposit_interest_reconciled": parts.get("Interest Expense on Customer Deposits")}
        if v is None:
            out.append(rec | {"status": "not reconciled", "detail": "line not in the reconciled note"})
        elif abs(v - rec["workbook_value"]) <= TOL * max(abs(v), 1):
            out.append(rec | {"status": "reconciled", "detail": "note lines sum to the audited total (0.1%)"})
        else:
            out.append(rec | {"status": "corrected", "detail": f"workbook {rec['workbook_value']:,.0f} -> "
                                                               f"reconciled {v:,.0f}"})
    return cross_report(pd.DataFrame(out))


# Notes whose layout the row parser cannot read but whose lines were read from the page and re-checked here: every
# value must be printed on the page and the lines must sum to the printed total (0.1%). (file, page, unit, fiscal year,
# document Doc ID) -> lines, total, layout note.
CR, NM = "Interest Expense on Current Accounts", "Interest Expense on Customer Deposits"
SV, TD, BR = "Interest Expense on Savings Deposits", "Interest Expense on Time Deposits", "Interest Expense on Borrowings"
_NMB21 = ({TD: 62_486, CR: 27_698, SV: 9_413, "due to other banks": 932, BR: 24_468, "NMB bond": 8_335,
           "lease liabilities": 2_274}, 135_606,
          "amounts are printed one row below their labels (the first amount row under the 'Deposits from customers' "
          "heading belongs to time deposits). Confirmed: the FY2020 column read the same way gives deposit interest "
          "106,289m, the held FY2020 value, and the FY2022 report's FY2022 column read the same way equals the FY2023 "
          "report's reconciled comparative (time 78,346m, current 32,958m, savings 11,843m)")
LAYOUT_VERIFIED = {
    ("CRDB/2010.pdf", 58, "000", 2010, "CRDB-AR-FY2010"): (
        {CR: 1_194_847, SV: 6_677_004, TD: 27_025_013, BR: 136_936}, 35_033_800,
        "Group column; 'Loans and other facilities' is the borrowing line"),
    ("CRDB/2010.pdf", 58, "000", 2009, "CRDB-AR-FY2010"): (
        {CR: 1_337_528, SV: 5_682_607, TD: 28_426_823, BR: 229_530}, 35_676_488,
        "Group column; 'Loans and other facilities' is the borrowing line"),
    ("CRDB/2012.pdf", 133, "million", 2011, "CRDB-AR-FY2012"): (
        {CR: 1_879, SV: 8_469, TD: 24_484, "inter-bank borrowing": 44, "subordinated debt": 96}, 34_972,
        "Group column; the subordinated-debt amounts are not split into columns by the row parser"),
    ("CRDB/2012.pdf", 133, "million", 2012, "CRDB-AR-FY2012"): (
        {CR: 2_036, SV: 9_551, TD: 43_663, "inter-bank borrowing": 119, "subordinated debt": 96}, 55_465,
        "Group column; the subordinated-debt amounts are not split into columns by the row parser"),
    ("NMB/2021.pdf", 317, "million", 2021, "NMB-AR-FY2021"): _NMB21,
    ("NMB/2022.pdf", 168, "million", 2021, "NMB-AR-FY2022"): _NMB21,
}


def check_layout(root: Path, file: str, page: int, unit: str, lines: dict, total: float) -> bool:
    """Every line value and the total are printed on the page, and the lines sum to the total (0.1%)."""
    from tzbank.extract.pdftext import _doc

    text = _doc(str(path("documents", root) / file))[page - 1].get_text()
    nums = {float(t.replace(",", "")) for t in re.findall(r"(?<![\d,.])\d{1,3}(?:,\d{3})*(?![\d,])", text)}
    printed = all(float(v) in nums for v in [*lines.values(), total])
    return printed and abs(sum(lines.values()) - total) <= TOL * total


def layout_verified(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Apply LAYOUT_VERIFIED to rows still not reconciled."""
    scale = {"000": 1e3, "million": 1e6}
    for (file, page, unit, fy, did), (lines, total, note) in LAYOUT_VERIFIED.items():
        if not check_layout(root, file, page, unit, lines, total):
            raise ValueError(f"layout-verified note no longer checks: {file} p.{page} FY{fy}")
        vals = {k: v * scale[unit] for k, v in lines.items()}
        vals[NM] = sum(vals[k] for k in (CR, SV, TD) if k in vals)
        sel = (df["doc_id"] == did) & (df["fiscal_year"] == fy) & (df["status"] == "not reconciled")
        for i in df.index[sel]:
            v = vals.get(df.at[i, "variable"])
            if v is None:
                continue
            same = abs(v - df.at[i, "workbook_value"]) <= TOL * v
            df.loc[i, ["status", "reconciled_value", "audited_total", "page", "file", "deposit_interest_reconciled",
                       "detail"]] = ["reconciled (layout verified)" if same else "corrected", v, total * scale[unit],
                                     page, file, vals[NM],
                                     f"note lines sum to the printed total {total:,} ({unit}); {note}"]
    return df


def cross_report(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """A row whose own document does not reconcile is still accepted when the same value is the reconciled value of
    another held report for that bank-year (the figure is printed and reconciled there): another workbook row, or a
    ledger line admitted by the note reconciliation (``data/extraction/ledger.csv``)."""
    root = root or find_root()
    df = df.drop(columns=["acts_on_workbook", "evidence_class"], errors="ignore")
    ok = df[df["status"].isin(["reconciled", "corrected"])].dropna(subset=["reconciled_value"])
    led = pd.read_csv(root / "data" / "extraction" / "ledger.csv")
    led = led[led["gate"].astype(str).str.contains("sum to audited total")]
    ok = pd.concat([ok, pd.DataFrame({"bank_id": led["bank_id"], "fiscal_year": led["fiscal_year"],
                                      "variable": led["variable"], "reconciled_value": led["value"],
                                      "doc_id": led["file"], "column": led["column"], "page": led["page"]})],
                   ignore_index=True)
    for i, r in df[df["status"] == "not reconciled"].iterrows():
        same = ok[(ok["bank_id"] == r["bank_id"]) & (ok["fiscal_year"] == r["fiscal_year"])
                  & (ok["variable"] == r["variable"])]
        hit = same[(same["reconciled_value"] - r["workbook_value"]).abs() <= TOL * r["workbook_value"]]
        if len(hit):
            h = hit.iloc[0]
            df.loc[i, ["status", "detail", "reconciled_value"]] = [
                "reconciled via another report",
                f"own document does not reconcile; the same value reconciles in {h['doc_id']} ({h['column']} column, "
                f"p.{int(h['page'])})", h["reconciled_value"]]
    df = layout_verified(df, root)
    df["acts_on_workbook"] = df["variable"].isin(DEPOSIT_VARS) & ~df["status"].str.startswith("reconciled")
    df["evidence_class"] = "Observed"
    return df.sort_values(["bank_id", "fiscal_year", "variable", "doc_id"]).reset_index(drop=True)


def _gate(notes: object) -> str:
    m = re.search(r"gate: ([^)]*\)?)\)", str(notes))
    return m.group(1) if m else ("manual" if "manual" in str(notes) else "")


def write_reconciliation(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    return [write_csv(reconcile(root), path("validation", root) / "deposit_interest_reconciliation.csv",
                      require_evidence=True)]


if __name__ == "__main__":
    import sys

    if "--from-csv" in sys.argv:  # re-apply the cross-report rule without re-reading the PDFs
        f = path("validation") / "deposit_interest_reconciliation.csv"
        df = cross_report(pd.read_csv(f))
        df["evidence_class"] = "Observed"
        write_csv(df, f, require_evidence=True)
    else:
        write_reconciliation()
        df = pd.read_csv(path("validation") / "deposit_interest_reconciliation.csv")
    print(df.groupby(["variable", "status"]).size())
