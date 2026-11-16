"""D18 workbook session for prompt 02c section 5: Sources entries for the inclusion and NPS reports (D16).

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_inclusion
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

No report carries a dated foreword or launch notice (searched in the text layer of the first pages), so each date
is an estimate with its basis stated:

- inclusion reports: the BoT document ID in the file name, read as an upload timestamp (YYYYMMDDhhmm...);
- NPS reports: no document ID; the PDF's modification timestamp. This is a lower bound for posting (a file cannot
  be posted before it is last modified); the BoT posting date is to be confirmed.

These series are context for Parts I-II and are not used in the valuations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.extract import workbook
from tzbank.extract.captions import INCLUSION_FILES, NPS_FILES

VERSION = "v0.10"


def _pdf_date(meta: str | None) -> pd.Timestamp | None:
    m = re.match(r"D:(\d{8})", meta or "")
    return pd.Timestamp(m.group(1)) if m else None


def report_sources(root: Path | None = None) -> pd.DataFrame:
    """Sources rows for the three inclusion and four NPS reports."""
    root = root or find_root()
    docs = path("documents", root)
    rows = []
    for kind, files in (("inclusion", INCLUSION_FILES), ("nps", NPS_FILES)):
        for year, f in files.items():
            doc = pymupdf.open(docs / kind / f)
            created, modified = _pdf_date(doc.metadata.get("creationDate")), _pdf_date(doc.metadata.get("modDate"))
            doc.close()
            meta = (f"PDF metadata: created {created.date() if created is not None else 'n/a'}, modified "
                    f"{modified.date() if modified is not None else 'n/a'}; no dated foreword or launch notice found")
            if kind == "inclusion":
                full_id = re.search(r"(20\d+)", f).group(1)
                est = pd.Timestamp(full_id[:8])
                basis = (f"Candidate posting date: BoT document ID in the file name ({full_id})"
                         f" read as an upload timestamp (YYYYMMDD...), consistent with the PDF creation date "
                         f"{created.date() if created is not None else 'n/a'}")
                doc_id, title = f"BOT-FIR-{year}", f"BoT Annual Financial Inclusion Report {year}"
            else:
                est = modified or created
                basis = (f"Lower bound: PDF modification timestamp {est.date()} (the file cannot be posted earlier); "
                         "no document ID in the file name; BoT posting date to be confirmed")
                doc_id, title = f"BOT-NPSAR-{year}", f"BoT National Payment Systems Annual Report {year}"
            rows.append({"Doc ID": doc_id, "Owner": "BoT", "Document Title": title,
                         "Document Type": "Statistical report", "Period Covered": str(year),
                         "Primary / Secondary": "PRIMARY", "Local File Path": f"documents/{kind}/{f}",
                         "Notes": f"Source of the {'inclusion' if kind == 'inclusion' else 'NPS'} panel (prompt 02c, "
                                  "D30); context for Parts I-II, not used in the valuations.",
                         "Period End": pd.Timestamp(year, 12, 31), "Date Evidence": meta,
                         "Estimated Publication Date": est, "Estimate Basis": basis})
    return pd.DataFrame(rows)


def run() -> dict[str, int]:
    root = find_root()
    new = report_sources(root)
    lines = [f"{VERSION} (2026-11-06, prompt 02c, D18 working workbook)",
             f"Sources: {len(new)} new documents, the BoT Annual Financial Inclusion Reports 2023-2025 (BOT-FIR-*) and "
             "National Payment Systems Annual Reports 2022-2025 (BOT-NPSAR-*). Estimated publication dates (D16): "
             "inclusion from the BoT document ID in the file name; NPS from the PDF modification date (lower bound). "
             "No other sheet changed."]
    return workbook.update(pd.DataFrame(columns=["variable"]), new, pd.DataFrame(columns=["Doc ID"]), lines, root)


if __name__ == "__main__":
    print(run())
