"""D18 workbook session for prompt 15c (v0.18): CRDB fee-note lines for FY2015 and FY2025 (figure fix ch04_4_1_fee_mix).

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_15c
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

The prompt 03a extractor (``tzbank.extract.detail_lines``) did not reconcile these two notes: FY2015 prints the
heading as "A Fee and Commission Income" with a misaligned salary-processing row; FY2025 prints the IFRS 15
disaggregation by segment (Corporate / Retail / Treasury / Total) with two sub-totals. The lines below are transcribed
from the Group column (FY2015) and the Group Total column (FY2025), TZS million; the same gate as prompt 03a applies
(the lines sum to the printed gross total), and the categories come from ``detail_lines.categorise`` so the mapping is
identical to every other year (D44; issue 6).
"""

from __future__ import annotations

import pandas as pd

from tzbank.config import find_root
from tzbank.extract import workbook
from tzbank.extract.detail_lines import NOTES, _clean, categorise, held
from tzbank.extract.session import assign_doc_ids
from tzbank.extract.sources import new_sources
from tzbank.ingest.inputs_workbook import load_interim

VERSION = "v0.18"
SCALE = 1e6
# (fiscal year, file, printed page, printed gross total, [(label, value)]) - Group, current year, TZS million
NOTES_15C = [
    (2015, "CRDB/2015.pdf", 106, 156_432, [
        ("Service charge on customer accounts", 42_458), ("Loan application fees", 29_003),
        ("ATM withdrawal charges", 10_668), ("VISA and master card fees", 3_878),
        ("Commission on letters of credit", 8_168), ("Fee on issue of bank cards", 7_521),
        ("Fee on local transfers and drafts", 7_593), ("Point of sale fees", 3_851),
        ("Fee on international telegraphic transfers", 6_179), ("Commission on guarantees and indemnities", 2_030),
        ("Commission on mobile phone services", 13_615), ("Salary processing fees", 5_810),
        ("Bills discounted", 479), ("Penalties", 2_209), ("Other fees and commissions", 12_970)]),
    (2025, "CRDB/crdb-ir-2025-8th-may-2026.pdf", 288, 599_314, [
        ("Service charge on customer accounts", 38_583), ("Commission on mobile phone services", 176_997),
        ("Loan commitment fees", 42_137), ("VISA and master card fees", 13_493),
        ("Commission on letters of credit", 36_790), ("Bank cards-annual fees", 24_109),
        ("Agency banking", 96_635), ("Fees on point of sale", 11_259), ("Custodianship commission", 16_911),
        ("Commission on guarantees", 30_756), ("Salary processing fees", 2_743),
        ("Bancassurance commission", 45_749), ("Fees on local transfers and drafts", 11_252),
        ("Fees on international telegraphic transfers", 5_346), ("Sale of cheque books", 39),
        ("ATM withdrawal charges", 26_225), ("Statement charges", 1_074), ("Other fees and commissions", 19_216)]),
]


def rows(root=None) -> pd.DataFrame:
    """Ledger rows in the prompt 03a layout; raises if a note fails the sum gate."""
    root = root or find_root()
    h = held(root)
    out = []
    for fy, f, page, total, lines in NOTES_15C:
        s = sum(v for _, v in lines)
        if abs(s - total) > 0.001 * total:
            raise ValueError(f"CRDB FY{fy} fee note: lines sum to {s:,} not the printed {total:,}")
        cats = categorise([(_clean(lab), v) for lab, v in lines], NOTES["fee_income"])
        audited = h.get(("B01", fy, "Non-Interest Income"))
        note = (f"note lines sum to the printed total {total:,} (TZS million); lines: "
                + "; ".join(f"{lab.lower()} {v:,}" for lab, v in lines))[:900] + " | prompt 15c transcription"
        base = {"bank_id": "B01", "fiscal_year": fy, "file": f, "page": page, "column": "current",
                "gate": "note lines sum to the printed note total (0.1%)", "gate_passed": True, "confidence": "HIGH",
                "notes": note, "note_kind": "fee_income", "note_total": total * SCALE, "audited_total": audited,
                "ocr": False, "extraction_method": "transcribed (gated)", "row_type": "original"}
        for var, v in cats.items():
            out.append(base | {"variable": var, "value": v * SCALE, "label": "sum of category lines"})
        out.append(base | {"variable": "Fee and Commission Income (Gross, Note Total)", "value": total * SCALE,
                           "label": "printed note total"})
    return pd.DataFrame(out)


def run() -> dict[str, int]:
    root = find_root()
    d = rows(root)
    src = load_interim("Sources", root)
    led = assign_doc_ids(d, set(src["Doc ID"]), root)
    new = new_sources(led[led["doc_is_new"]], root) if led["doc_is_new"].any() else pd.DataFrame(columns=["Doc ID"])
    lines = [f"{VERSION} (2026-11-13, prompt 15c, D18 working workbook)",
             f"Historical Financials: CRDB gross fee income by type for FY2015 and FY2025 ({len(led)} rows), "
             "transcribed from the fee notes (FY2015 p.106 Group; FY2025 p.288 Group Total) that the prompt 03a "
             "extractor did not reconcile; lines sum to the printed totals; categories as prompt 03a (D44)."]
    counts = workbook.update(led, new, pd.DataFrame(columns=["Doc ID"]), lines, root)
    lp = root / "data" / "extraction" / "ledger.csv"
    old = pd.read_csv(lp)
    pd.concat([old, led.reindex(columns=old.columns).assign(evidence_class="Observed")], ignore_index=True).to_csv(
        lp, index=False, lineterminator="\n")
    return counts | {"rows": len(led)}


if __name__ == "__main__":
    print(run())
