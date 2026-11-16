"""D18 workbook session for prompt 03b (v0.15): inputs for chapter 4.4-4.6.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_03b
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

1. NMB interest-earning assets FY2016-FY2017 (fix 3): placements and balances with other banks + government
   securities (loans and receivables + held to maturity) + net loans, the FY2015/FY2018 definition, from the
   consolidated statement of financial position (FY2016 report p.81; FY2017 report p.79), each value re-checked on its
   page; fills the TO COLLECT placeholders so the NMB full DuPont tree covers the 2015-18 stress period.
2. NMB and CRDB net foreign exchange income (4.4): the income-statement line, own-year report first; gate: printed on
   the page and between 0 and the held non-interest income.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.detail_lines import BANKS, _amounts, _clean, full_reports, held
from tzbank.extract.ocr import extraction_source
from tzbank.extract.pdftext import _doc, page_rows, page_unit, text_pages
from tzbank.extract.picker import SCALE
from tzbank.extract.session import assign_doc_ids
from tzbank.extract.session_02e import printed

VERSION = "v0.15"
FX_RX = r"^(net )?(foreign exchange|forex)( trading)? (income|gains?|revaluation gains?)( net)?$|^net (gain|income) from " \
        r"foreign exchange"
IEA = {  # fiscal year -> (file, page, placements, government securities parts, net loans) in TZS million
    2016: ("NMB/2016.pdf", 81, 137_241, [724_197], 2_774_464),
    2017: ("NMB/2017.pdf", 79, 242_731, [422_732, 496_367], 2_771_732),
}


def iea_rows(root: Path) -> list[dict]:
    out = []
    for fy, (file, page, plc, sec, loans) in IEA.items():
        if not printed(file, page, [plc, *sec, loans], root):
            raise ValueError(f"NMB FY{fy} SoFP lines not printed on {file} p.{page}")
        total = plc + sum(sec) + loans
        out.append({"bank_id": "B02", "fiscal_year": fy, "variable": "Interest-Earning Assets", "value": total * 1e6,
                    "file": file, "page": page, "label": "sum of placements, government securities and net loans",
                    "column": "current", "gate": "SoFP lines printed on the page (prompt 03b fix 3)",
                    "gate_passed": True, "confidence": "HIGH", "row_type": "original", "ocr": False,
                    "extraction_method": "gated",
                    "notes": f"COMPUTED = placements and balances with other banks {plc:,} + government securities "
                             f"{' + '.join(f'{x:,}' for x in sec)} + net loans {loans:,} = {total:,} (TZS m); same "
                             "definition as FY2015 and FY2018 (BoT balances and AFS excluded)"})
    return out


def fx_line(pdf: Path, col: int, nonii: float | None) -> dict | None:
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if "foreign exchange" not in t or "profit before" not in t:
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        for half in (None, "left", "right"):
            for r in page_rows(pdf, p, half=half):
                if re.match(FX_RX, _clean(r.label)) and "%" not in r.text:
                    a = _amounts(r)
                    if len(a) > col:
                        v = a[col] * SCALE[unit]
                        if nonii and 0 < v <= 1.05 * nonii:
                            return {"page": p, "value": v, "label": r.label.strip()[:60]}
    return None


def fx_rows(root: Path) -> list[dict]:
    h = held(root)
    out = {}
    for b in BANKS:
        for ry, f in full_reports(b, root).items():
            src = extraction_source(f, root)
            for col, fy in [(0, ry), (1, ry - 1)]:
                if (b, fy) in out and out[(b, fy)]["column"] == "current":
                    continue
                hit = fx_line(src, col, h.get((b, fy, "Non-Interest Income")))
                if hit:
                    out[(b, fy)] = {"bank_id": b, "fiscal_year": fy, "variable": "Net Foreign Exchange Income",
                                    "value": hit["value"], "file": f"{f.parent.name}/{f.name}", "page": hit["page"],
                                    "label": hit["label"], "column": "current" if col == 0 else "prior",
                                    "gate": "income-statement line printed; between 0 and held non-interest income",
                                    "gate_passed": True, "confidence": "HIGH" if col == 0 else "MEDIUM",
                                    "row_type": "original", "ocr": src != f, "extraction_method": "gated",
                                    "notes": "prompt 03b (4.4)"}
    return list(out.values())


def run() -> dict[str, int]:
    import csv

    root = root_ = find_root()
    led = pd.DataFrame(iea_rows(root) + fx_rows(root))
    src = load_sources(root_)
    led = assign_doc_ids(led, set(src["Doc ID"]), root)
    cw = path("crosswalk", root)
    rows = list(csv.reader(cw.open(encoding="utf-8")))
    if not any(r[2] == "Net Foreign Exchange Income" for r in rows[1:] if len(r) > 2):
        d = dict.fromkeys(rows[0], "")
        d.update(canonical_name="fx_income", inputs_name="Net Foreign Exchange Income", unit="TZS", basis="GROUP",
                 definition="Net foreign exchange income (income-statement line)", blueprint_chapters="4",
                 evidence_class="Observed", source="inputs_workbook", source_sheets="Historical Financials",
                 status="mapped")
        with cw.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerow([d[h] for h in rows[0]])
    lines = [f"{VERSION} (2026-11-08, prompt 03b, D18 working workbook)",
             f"Historical Financials: NMB interest-earning assets FY2016-17 (placeholders filled); net foreign "
             f"exchange income for NMB and CRDB ({(led['variable'] == 'Net Foreign Exchange Income').sum()} rows)."]
    counts = workbook.update(led, pd.DataFrame(columns=["Doc ID"]), pd.DataFrame(columns=["Doc ID"]), lines, root)
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp)
    pd.concat([old, led.reindex(columns=old.columns).assign(evidence_class="Observed")], ignore_index=True).to_csv(
        lp, index=False, lineterminator="\n")
    return counts | {"rows": len(led)}


def load_sources(root: Path) -> pd.DataFrame:
    from tzbank.ingest.inputs_workbook import load_interim

    return load_interim("Sources", root)


if __name__ == "__main__":
    print(run())
