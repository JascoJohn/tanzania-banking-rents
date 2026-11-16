"""D18 workbook session for prompt 02e (v0.12): new reports, manual checklist, extra extraction.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_02e
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

1. CRDB FY2025 full report (crdb-ir-2025-8th-may-2026.pdf): deposit interest by type and borrowings interest from
   Note 13 (lines sum to the audited total interest expense), deposit mix from the Note 45 maturity table (sums to
   customer deposits), liquidity ratio; FY2024 comparatives restated in that report (D21). Every existing FY2025 row
   is re-checked against its cited page (``reports/validation/crdb_fy2025_fullreport_check.csv``).
2. CRDB FY2010 non-interest income (FY2010 report; same definition as FY2011-13).
3. MKCB FY2021: the original publication's NII (net profit, interest income and expense come from the owner's
   checklist) and the FY2022 report's restated FY2021 lines as restatement rows (D21).
4. The owner's filled checklist rows accepted by ``tzbank.ingest.manual_checklist`` (D38).
5. Sources: the 2008 gazette (D36), CRDB-AR-FY2025 local file and FS approval / auditor dates, NMB-AR-FY2023 note.

Every value written here is checked against its printed page before the workbook is opened.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.ocr import extraction_source
from tzbank.extract.pdftext import _doc

VERSION = "v0.12"
TODAY = "2026-11-07"
CRDB25 = "CRDB/crdb-ir-2025-8th-may-2026.pdf"
M = 1_000_000


def printed(file: str, page: int, values: list[float], root: Path) -> bool:
    """Every value (as printed, absolute) appears on the page of the held PDF (OCR copy for scans)."""
    pdf = extraction_source(path("documents", root) / file, root)
    text = _doc(str(pdf))[page - 1].get_text()
    nums = set()
    for t in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text):
        nums.add(float(t.replace(",", "")))
    return all(abs(float(v)) in nums for v in values)


def _row(bank, fy, var, value, file, page, label, gate, conf, notes="", row_type="original", column="current",
         doc_id=None):
    return {"bank_id": bank, "fiscal_year": fy, "variable": var, "value": float(value), "file": file, "page": page,
            "label": label, "column": column, "gate": gate, "gate_passed": True, "confidence": conf, "notes": notes,
            "row_type": row_type, "ocr": False, "extraction_method": "gated", "doc_id": doc_id}


def crdb_2025_rows(root: Path) -> list[dict]:
    """Note 13 (p.287) and Note 45 (p.314), Group, TZS million."""
    note13 = {2025: dict(cur=46_130, sav=40_446, fix=196_032, banks=109_642, bor=116_714, bond=52_134, sub=16_357,
                         eir=577_455, lease=5_819, isl=7_898, total=591_172),
              2024: dict(cur=22_713, sav=34_186, fix=125_457, banks=118_547, bor=85_328, bond=17_612, sub=13_020,
                         eir=416_863, lease=4_755, isl=1_031, total=422_649)}
    mix = {2025: (7_134_720, 5_091_225, 2_704_559, 14_930_504), 2024: (5_330_444, 3_902_157, 1_701_530, 10_934_131)}
    out = []
    for fy, n in note13.items():
        lines = [n[k] for k in ("cur", "sav", "fix", "banks", "bor", "bond", "sub")]
        if not printed(CRDB25, 287, [*lines, n["eir"], n["lease"]], root) or sum(lines) != n["eir"] or \
                n["eir"] + n["lease"] + n["isl"] != n["total"]:
            raise ValueError(f"CRDB FY{fy} Note 13 no longer reconciles")
        col = "current" if fy == 2025 else "prior"
        conf = "HIGH" if fy == 2025 else "MEDIUM"
        gate = (f"note lines sum to audited total interest expense: {n['eir']:,} + lease {n['lease']:,} + Islamic "
                f"distribution {n['isl']:,} = {n['total']:,} (TZS m, p.287 and p.195)")
        isl = (f"Distribution on Islamic deposits and sukuk profit ({n['isl']:,}m) is shown below the conventional note "
               "lines and is not included in deposit interest")
        dep = n["cur"] + n["sav"] + n["fix"]
        for var, v, lab in [("Interest Expense on Current Accounts", n["cur"], "current accounts"),
                            ("Interest Expense on Savings Deposits", n["sav"], "savings accounts"),
                            ("Interest Expense on Time Deposits", n["fix"], "fixed deposits"),
                            ("Interest Expense on Customer Deposits", dep,
                             "sum of current accounts, savings accounts, fixed deposits"),
                            ("Interest Expense on Borrowings", n["bor"] + n["bond"] + n["sub"],
                             "Borrowings (Note 50) + Bonds (Note 49) + Subordinated debt (Note 51)")]:
            out.append(_row("B01", fy, var, v * M, CRDB25, 287, lab, gate, conf, isl, column=col))
        c, s, t, total = mix[fy]
        if not printed(CRDB25, 314, [c, s, t, total], root) or c + s + t != total:
            raise ValueError(f"CRDB FY{fy} Note 45 maturity table no longer sums")
        for var, v, lab in [("Customer Deposits: Current Accounts", c, "Demand deposits (incl. Qardh CASA)"),
                            ("Customer Deposits: Savings", s, "Saving deposits"),
                            ("Customer Deposits: Time", t, "Time and Call Deposits (incl. Mudharabah)")]:
            out.append(_row("B01", fy, var, v * M, CRDB25, 314, lab,
                            f"mix lines sum to customer deposits {total:,} (TZS m, Note 45 maturity table)", conf,
                            column=col))
    # FY2024 income-statement lines restated in the FY2025 report (Islamic income presentation), p.195
    ii, nii, pbt, opex, imp = 1_528_377 + 12_449, 1_118_177, 778_796, 739_568, 95_187
    if not printed(CRDB25, 195, [1_528_377, 12_449, 1_118_177, 778_796, 739_568, 95_187], root):
        raise ValueError("CRDB FY2024 restated SOPL lines not printed on p.195")
    restated = "FY2024 column marked 'Restated' in the FY2025 report (Islamic income presentation)"
    out.append(_row("B01", 2024, "Interest Income", ii * M, CRDB25, 195,
                    "EIR interest income 1,528,377 + other 0 + Islamic financing income 12,449",
                    "II - IE = NII: 1,540,826 - 422,649 = 1,118,177 (TZS m)", "MEDIUM", restated, column="prior"))
    out.append(_row("B01", 2024, "Net Interest Income", nii * M, CRDB25, 195,
                    "Net interest income and net income from Islamic financing and investment products",
                    "II - IE = NII on the statement page", "MEDIUM", restated, column="prior"))
    out.append(_row("B01", 2024, "Non-Interest Income", (pbt + opex + imp - nii) * M, CRDB25, 195,
                    "COMPUTED = profit before tax + operating expenses + impairment - NII (FY2011+ definition)",
                    "income statement lines sum to profit before tax", "MEDIUM", restated, column="prior"))
    if ii - 422_649 != nii:
        raise ValueError("CRDB FY2024 restated II - IE != NII")
    return out


def other_rows(root: Path) -> list[dict]:
    out = []
    # CRDB FY2010 non-interest income (FY2010 AR p.32, TZS '000): PBT + opex + impairment - NII
    pbt, opex, imp, nii = 65_636_760, 63_360_635 + 46_180_800, 20_356_975, 125_005_122
    parts = [46_590_928, 22_080_586, 2_060_367, -201_833]
    if not printed("CRDB/2010.pdf", 32, [pbt, 63_360_635, 46_180_800, imp, nii, *parts], root) or \
            pbt + opex + imp - nii != sum(parts):
        raise ValueError("CRDB FY2010 income statement no longer reconciles")
    out.append(_row("B01", 2010, "Non-Interest Income", (pbt + opex + imp - nii) * 1_000, "CRDB/2010.pdf", 32,
                    "sum of net fee and commission income, net foreign exchange income, other operating income, net "
                    "loss on fair value hedges", "income statement lines sum to profit before tax", "HIGH",
                    "COMPUTED = profit before tax + operating expenses + impairment - NII (FY2011+ definition); "
                    "equals 46,590,928 + 22,080,586 + 2,060,367 - 201,833 (TZS '000). The FY2011 report's restated "
                    "FY2010 lines give the same total (70,530m)."))
    # MKCB FY2021 original publication (MKCB/2021.pdf p.1, OCR, TZS m): NII = 22,820 - 8,192
    if not printed("MKCB/2021.pdf", 1, [22_820, 8_192, 14_628, 1_148], root):
        raise ValueError("MKCB FY2021 condensed lines not printed")
    out.append(_row("B05", 2021, "Net Interest Income", 14_628 * M, "MKCB/2021.pdf", 1,
                    "Net Interest Income (1 minus 2), current year cumulative", "II - IE = NII (22,820 - 8,192)",
                    "HIGH", "Original Q4 2021 BoT publication (prompt 02e section 1.5)"))
    # MKCB FY2021 restated in the FY2022 report (MKCB/2022.pdf p.26, OCR, TZS '000)
    rs = {"Interest Income": 22_075_616, "Interest Expense": 8_787_173, "Net Interest Income": 13_288_443,
          "Profit Before Tax": -2_706_838, "Net Profit": -1_859_510}
    if not printed("MKCB/2022.pdf", 26, list(rs.values()), root) or 22_075_616 - 8_787_173 != 13_288_443:
        raise ValueError("MKCB FY2021 restated lines not printed or do not reconcile")
    for var, v in rs.items():
        out.append(_row("B05", 2021, var, abs(v) * 1_000 if var == "Interest Expense" else v * 1_000,
                        "MKCB/2022.pdf", 26, f"{var} (2021 'Restated' column)",
                        "II - IE = NII on the statement page (22,075,616 - 8,787,173 = 13,288,443)", "MEDIUM",
                        "RESTATED in MKCB-AR-FY2022 comparative (column marked 'Restated'; loan-modification loss "
                        "2,757,741k); original from MKCB-FSPUB-FY2021", row_type="restated", column="prior"))
    return out


def manual_rows(root: Path) -> pd.DataFrame:
    """Accepted owner checklist rows (D38) as ledger rows."""
    from tzbank.ingest.manual_checklist import gated

    acc, _ = gated(root)
    if acc.empty:
        return acc
    notes = acc["notes"] + "; " + acc["gate_results"] + acc["yoy_flag"].fillna("").map(lambda s: f"; {s}" if s else "")
    return pd.DataFrame({
        "bank_id": acc["bank_id"], "fiscal_year": acc["fiscal_year"].astype(int), "variable": acc["variable"],
        "value": acc["value_full"], "file": acc["file"], "page": acc["page"].astype(int), "label":
        acc["as_reported_label"], "column": acc["column"], "gate": "manual entry (D29/D38): " + acc["gate_results"],
        "gate_passed": True, "confidence": acc["confidence"], "notes": notes + " | read by " + acc["checked_by"],
        "row_type": "original", "ocr": False, "extraction_method": "manual", "doc_id": acc["doc_id"]})


def fy2025_check(root: Path) -> pd.DataFrame:
    """Every existing CRDB FY2025 row (CRDB-AR-FY2025) against its cited page in the now-held full report."""
    from tzbank.ingest.inputs_workbook import load_interim

    out = []
    for sheet in ("Historical Financials", "Regulatory Capital"):
        df = load_interim(sheet, root)
        df = df[(df["Doc ID"] == "CRDB-AR-FY2025") & df["value_numeric"].notna()]
        for _, r in df.iterrows():
            v = float(r["value_numeric"])
            pages = [int(p) for p in re.findall(r"p\.?\s?(\d+)", str(r["Page / Note"]))
                     if 1 <= int(p) <= _doc(str(path("documents", root) / CRDB25)).page_count][:2]
            if re.search(r"per Share|Shares(?![a-z])", r["Variable"]):
                unit, target = "as reported", v
            else:
                unit = "%" if abs(v) < 1.5 else "TZS m"
                target = round(v * 100, 1) if unit == "%" else round(v / M)
            hit = None
            for p in pages:
                for q in (p, p - 1, p + 1):
                    if 1 <= q and printed(CRDB25, q, [target], root):
                        hit = q
                        break
                if hit:
                    break
            status = ("printed on the cited page" if hit else "computed (components on the cited page)"
                      if str(r["Status"]) == "COMPUTED" else "not printed as such on the cited page")
            out.append({"sheet": sheet, "row_id": r["Row ID"], "entity": r["Entity"], "variable": r["Variable"],
                        "value": v, "printed_as": target, "unit": unit, "cited": r["Page / Note"],
                        "page_found": hit, "status": status, "evidence_class": "Observed"})
    return pd.DataFrame(out)


def source_updates(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tzbank.ingest.inputs_workbook import load_interim

    src = load_interim("Sources", root)
    nmb = src.loc[src["Doc ID"] == "NMB-AR-FY2023", "Notes"].iloc[0]
    upd = pd.DataFrame([
        {"Doc ID": "CRDB-AR-FY2025", "Local File Path": f"data/raw/documents/{CRDB25}",
         "FS Board Approval Date": pd.Timestamp("2026-03-13"), "Auditor Report Date": pd.Timestamp("2026-03-31"),
         "Date Evidence": "Held locally from 2026-11-07. Statement of financial position (PDF p.196): 'approved and "
                          "authorized for issue by those charged with governance on 13 March 2026'; independent "
                          "auditor's report (PwC) dated 31 March 2026 (p.194); directors' statements signed 30 March "
                          "2026 (pp.191-192). Actual publication 8 May 2026 from the bank's posting (file name)."},
        {"Doc ID": "NMB-AR-FY2023", "Notes": f"{nmb} | A re-saved copy of the same 218-page report was placed on "
                                             "2026-11-07 as 'NMB 2023 Integrated.pdf' (modified 29 May 2025); same "
                                             "document, not a separate source."},
    ])
    new = pd.DataFrame([{
        "Doc ID": "REG-GAZ-2008-50", "Owner": "Government of Tanzania",
        "Document Title": "Gazette of the United Republic of Tanzania No. 50, 12 December 2008 (notice No. 632: GN 368 "
                          "of 2008 published in Subsidiary Legislation Supplement No. 41)",
        "Document Type": "Regulation", "Period Covered": "2008", "Primary / Secondary": "PRIMARY",
        "Local File Path": "data/raw/documents/regulation/tz-government-gazette-dated-2008-12-12-no-50.pdf",
        "Notes": "Evidence for D36: records the publication of the BFI (Publication of Financial Statements) "
                 "Regulations as GN 368 of 2008; the regulation text itself is not held.",
        "Period End": pd.Timestamp("2008-12-12"), "Actual Publication Date": pd.Timestamp("2008-12-12"),
        "Date Evidence": "Gazette date printed on the issue (12 Desemba, 2008)"}])
    return new, upd


def run() -> dict[str, int]:
    from tzbank.extract.session import assign_doc_ids
    from tzbank.extract.sources import new_sources
    from tzbank.ingest.inputs_workbook import load_interim

    root = find_root()
    chk = fy2025_check(root)
    chk.to_csv(path("validation", root) / "crdb_fy2025_fullreport_check.csv", index=False, lineterminator="\n")
    rows = pd.DataFrame(crdb_2025_rows(root) + other_rows(root))
    man = manual_rows(root)
    led = pd.concat([rows, man], ignore_index=True)
    src = load_interim("Sources", root)
    auto = assign_doc_ids(led.drop(columns=["doc_id"]), set(src["Doc ID"]), root)
    led["doc_id"] = led["doc_id"].fillna(auto["doc_id"])
    led["doc_fiscal_year"] = led["doc_id"].str[-4:].astype(int)
    led["doc_is_new"] = ~led["doc_id"].isin(set(src["Doc ID"]))
    new_docs = new_sources(led[led["doc_is_new"]], root)
    gaz, upd = source_updates(root)
    counts = workbook.update(led, pd.concat([new_docs, gaz], ignore_index=True), upd, [], root)

    # Regulatory Capital: CRDB FY2025 liquidity ratio (p.271 and p.52: 'the liquidity ratio stood at 30.3%')
    if not printed(CRDB25, 52, [30.3], root):
        raise ValueError("CRDB FY2025 liquidity ratio not printed on p.52")
    wb = openpyxl.load_workbook(path("inputs_workbook", root))
    rc, me = wb["Regulatory Capital"], wb["Read Me"]
    h = workbook._header(rc)
    filled = 0
    for r in range(2, workbook._last_row(rc) + 1):
        pe = rc.cell(r, h["Period End"]).value
        if (rc.cell(r, h["Bank ID"]).value == "B01" and hasattr(pe, "year") and pe.year == 2025
                and rc.cell(r, h["Variable"]).value == "Liquid Assets Ratio (BoT definition)"
                and rc.cell(r, h["Value"]).value in (None, "")):
            prev = rc.cell(r, h["Notes"]).value
            for k, v in {"Value": 0.303, "Status": "REPORTED", "Doc ID": "CRDB-AR-FY2025",
                         "Page / Note": "PDF p.271 (Note 10.4.2), p.52", "Confidence": "MEDIUM",
                         "Notes": "Liquidity ratio 30.3% (regulatory minimum 20%), Group and Bank, as stated in the "
                                  "FY2025 report; taken as the BoT liquid assets ratio | prompt 02e | previous note: "
                                  f"{prev}"}.items():
                if k in h:
                    rc.cell(r, h[k]).value = v
            filled += 1
    lines = [f"{VERSION} ({TODAY}, prompt 02e, D18 working workbook)",
             f"Historical Financials / Regulatory Capital: {len(rows)} rows from held reports (CRDB FY2025 full report: "
             "deposit interest by type, borrowings interest, deposit mix, FY2024 restated comparatives; CRDB FY2010 "
             "non-interest income; MKCB FY2021 original NII and FY2022-report restatement rows) and "
             f"{len(man)} owner checklist rows (D38, extraction_method=manual); counts {counts}; CRDB FY2025 "
             f"liquidity ratio filled: {filled}.",
             "Sources: REG-GAZ-2008-50 (D36); CRDB-AR-FY2025 local file, FS approval 13 Mar 2026, auditor 31 Mar 2026; "
             f"NMB-AR-FY2023 copy noted; {len(new_docs)} new report documents."]
    r = me.max_row + 2
    for line in lines:
        me.cell(r, 1).value = line
        r += 1
    wb.save(path("inputs_workbook", root))
    # audit trail: append the written rows to the extraction ledger
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp)
    add = led.reindex(columns=old.columns)
    add["evidence_class"] = "Observed"
    pd.concat([old, add], ignore_index=True).to_csv(lp, index=False, lineterminator="\n")
    return counts | {"rows_extracted": len(rows), "manual_rows": len(man), "liquidity_ratio": filled,
                     "new_docs": len(new_docs)}


if __name__ == "__main__":
    print(run())
