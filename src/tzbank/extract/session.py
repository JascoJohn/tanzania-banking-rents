"""One D18 workbook update session for prompt 02: ledger -> Doc IDs -> Sources -> workbook.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.
"""

from __future__ import annotations

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.sources import existing_updates, new_sources
from tzbank.ingest.inputs_workbook import load_interim

VERSION = "v0.9"


def assign_doc_ids(ledger: pd.DataFrame, existing: set[str], root) -> pd.DataFrame:
    """Doc ID per ledger row: the report whose current column holds the value (fiscal year of the document)."""
    led = ledger.copy()
    led["doc_fiscal_year"] = led["fiscal_year"].astype(int) + (led["column"] != "current").astype(int)
    pages = {f: pymupdf.open(path("documents", root) / f).page_count for f in led["file"].unique()}
    ids = []
    for f, y, b in zip(led["file"], led["doc_fiscal_year"], led["bank_id"], strict=False):
        condensed = pages[f] <= 6
        d = workbook.doc_id_for(b, f, int(y), condensed)
        ids.append(d)
    led["doc_id"] = ids
    led["doc_is_new"] = ~led["doc_id"].isin(existing)
    return led


def run() -> dict[str, int]:
    root = find_root()
    ledger = pd.read_csv(root / "data" / "extraction" / "ledger.csv")
    src = load_interim("Sources", root)
    led = assign_doc_ids(ledger, set(src["Doc ID"]), root)
    led.to_csv(root / "data" / "extraction" / "ledger.csv", index=False, lineterminator="\n")
    new = new_sources(led[led["doc_is_new"]], root)
    upd = existing_updates(src, root)
    imf = {"IMF-WEO-HIST": ("2025-10-31", "IMF World Economic Outlook, October 2025 edition (latest public by "
                                          "2025-12-31). October editions are released in mid-October; end of October "
                                          "used as a conservative upper bound. imf.org refused automated access (403)."),
           "IMF-WEO-CURR": ("2026-10-31", "IMF World Economic Outlook, latest edition public by 2026-10-31 (October "
                                          "2026 if released by then, else April 2026); end of October used as a "
                                          "conservative upper bound. imf.org refused automated access (403); confirm "
                                          "the edition and date when the WEO values are collected.")}
    for d, (date, basis) in imf.items():
        upd.loc[upd["Doc ID"] == d, "Estimated Publication Date"] = pd.Timestamp(date)
        upd.loc[upd["Doc ID"] == d, "Estimate Basis"] = basis
    n_banks = led.groupby("bank_id").size().to_dict()
    lines = [f"{VERSION} (2026-11-06, prompt 02b, D18 working workbook)",
             f"Historical Financials / Regulatory Capital: {len(led)} rows extracted from held reports by tzbank.extract "
             f"(gated; audit trail data/extraction/ledger.csv). Rows by bank: {n_banks}.",
             f"Sources: {len(new)} new documents (B03-B07 reports and NMB/CRDB condensed publications); evidence columns "
             "Period End, Actual Publication Date, FS Board Approval Date, Auditor Report Date, Date Evidence and "
             "Legacy Publication Date (pre-D16) added (D16). Publication dates are recomputed in Python.",
             "Second extraction pass (prompt 02b): rows already held are skipped; later-report comparatives that "
             "differ by more than 0.5% are added as restatement rows (D21). IMF WEO sources dated (estimated)."]
    return workbook.update(led, new, upd, lines, root)


if __name__ == "__main__":
    print(run())
